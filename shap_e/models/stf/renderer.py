import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from shap_e.models.nn.camera import DifferentiableCamera, DifferentiableProjectiveCamera
from shap_e.models.nn.meta import subdict
from shap_e.models.nn.utils import to_torch
from shap_e.models.query import Query
from shap_e.models.renderer import Renderer, get_camera_from_batch
from shap_e.models.volume import BoundingBoxVolume, Volume
from shap_e.rendering.blender.constants import BASIC_AMBIENT_COLOR, BASIC_DIFFUSE_COLOR
from shap_e.rendering.mc import marching_cubes
from shap_e.rendering.torch_mesh import TorchMesh
from shap_e.rendering.view_data import ProjectiveCamera
from shap_e.util.collections import AttrDict

from .base import Model


class STFRendererBase(ABC):
    @abstractmethod
    def get_signed_distance(
        self,
        position: torch.Tensor,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_texture(
        self,
        position: torch.Tensor,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        pass


class STFRenderer(Renderer, STFRendererBase):
    def __init__(
        self,
        sdf: Model,
        tf: Model,
        volume: Volume,
        grid_size: int,
        texture_channels: Sequence[str] = ("R", "G", "B"),
        channel_scale: Sequence[float] = (255.0, 255.0, 255.0),
        ambient_color: Union[float, Tuple[float]] = BASIC_AMBIENT_COLOR,
        diffuse_color: Union[float, Tuple[float]] = BASIC_DIFFUSE_COLOR,
        specular_color: Union[float, Tuple[float]] = 0.0,
        output_srgb: bool = True,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(volume, BoundingBoxVolume), "cannot sample points in unknown volume"
        self.sdf = sdf
        self.tf = tf
        self.volume = volume
        self.grid_size = grid_size
        self.texture_channels = texture_channels
        self.channel_scale = to_torch(channel_scale).to(device)
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.output_srgb = output_srgb
        self.device = device
        self.to(device)

    def render_views(
        self,
        batch: Dict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        params = self.update(params)
        options = AttrDict() if not options else AttrDict(options)

        sdf_fn = partial(self.sdf.forward_batched, params=subdict(params, "sdf"))
        tf_fn = partial(self.tf.forward_batched, params=subdict(params, "tf"))
        nerstf_fn = None

        return render_views_from_stf(
            batch,
            options,
            sdf_fn=sdf_fn,
            tf_fn=tf_fn,
            nerstf_fn=nerstf_fn,
            volume=self.volume,
            grid_size=self.grid_size,
            channel_scale=self.channel_scale,
            texture_channels=self.texture_channels,
            ambient_color=self.ambient_color,
            diffuse_color=self.diffuse_color,
            specular_color=self.specular_color,
            output_srgb=self.output_srgb,
            device=self.device,
        )

    def get_signed_distance(
        self,
        query: Query,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        return self.sdf(
            query,
            params=subdict(params, "sdf"),
            options=options,
        ).signed_distance

    def get_texture(
        self,
        query: Query,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        return self.tf(
            query,
            params=subdict(params, "tf"),
            options=options,
        ).channels


def render_views_from_stf(
    batch: Dict,
    options: AttrDict[str, Any],
    *,
    sdf_fn: Optional[Callable],
    tf_fn: Optional[Callable],
    nerstf_fn: Optional[Callable],
    volume: BoundingBoxVolume,
    grid_size: int,
    channel_scale: torch.Tensor,
    texture_channels: Sequence[str] = ("R", "G", "B"),
    ambient_color: Union[float, Tuple[float]] = 0.0,
    diffuse_color: Union[float, Tuple[float]] = 1.0,
    specular_color: Union[float, Tuple[float]] = 0.2,
    output_srgb: bool = False,
    device: torch.device = torch.device("cuda"),
) -> AttrDict:
    """
    :param batch: contains either ["poses", "camera"], or ["cameras"]. Can
        optionally contain any of ["height", "width", "query_batch_size"]
    :param options: controls checkpointing, caching, and rendering
    :param sdf_fn: returns [batch_size, query_batch_size, n_output] where
        n_output >= 1.
    :param tf_fn: returns [batch_size, query_batch_size, n_channels]
    :param volume: AABB volume
    :param grid_size: SDF sampling resolution
    :param texture_channels: what texture to predict
    :param channel_scale: how each channel is scaled
    :return: at least
        channels: [batch_size, len(cameras), height, width, 3]
        transmittance: [batch_size, len(cameras), height, width, 1]
        aux_losses: AttrDict[str, torch.Tensor]
    """
    camera, batch_size, inner_shape = get_camera_from_batch(batch)
    inner_batch_size = int(np.prod(inner_shape))
    assert camera.width == camera.height, "only square views are supported"
    assert camera.x_fov == camera.y_fov, "only square views are supported"
    assert isinstance(camera, DifferentiableProjectiveCamera)

    device = camera.origin.device
    device_type = device.type

    TO_CACHE = ["fields", "raw_meshes", "raw_signed_distance", "raw_density", "mesh_mask", "meshes"]
    if options.cache is not None and all(key in options.cache for key in TO_CACHE):
        fields = options.cache.fields
        raw_meshes = options.cache.raw_meshes
        raw_signed_distance = options.cache.raw_signed_distance
        raw_density = options.cache.raw_density
        mesh_mask = options.cache.mesh_mask
    else:
        query_batch_size = batch.get("query_batch_size", batch.get("ray_batch_size", 4096))
        query_points = volume_query_points(volume, grid_size)
        fn = nerstf_fn if sdf_fn is None else sdf_fn
        sdf_out = fn(
            query=Query(position=query_points[None].repeat(batch_size, 1, 1)),
            query_batch_size=query_batch_size,
            options=options,
        )
        raw_signed_distance = sdf_out.signed_distance
        raw_density = None
        if "density" in sdf_out:
            raw_density = sdf_out.density
        with torch.autocast(device_type, enabled=False):
            fields = sdf_out.signed_distance.float()
            raw_signed_distance = sdf_out.signed_distance
            assert (
                len(fields.shape) == 3 and fields.shape[-1] == 1
            ), f"expected [meta_batch x inner_batch] SDF results, but got {fields.shape}"
            fields = fields.reshape(batch_size, *([grid_size] * 3))

            # Force a negative border around the SDFs to close off all the models.
            full_grid = torch.zeros(
                batch_size,
                grid_size + 2,
                grid_size + 2,
                grid_size + 2,
                device=fields.device,
                dtype=fields.dtype,
            )
            full_grid.fill_(-1.0)
            full_grid[:, 1:-1, 1:-1, 1:-1] = fields
            fields = full_grid

            raw_meshes = []
            mesh_mask = []
            for field in fields:
                raw_mesh = marching_cubes(field, volume.bbox_min, volume.bbox_max - volume.bbox_min)
                if len(raw_mesh.faces) == 0:
                    # DDP deadlocks when there are unused parameters on some ranks
                    # and not others, so we make sure the field is a dependency in
                    # the graph regardless of empty meshes.
                    vertex_dependency = field.mean()
                    raw_mesh = TorchMesh(
                        verts=torch.zeros(3, 3, device=device) + vertex_dependency,
                        faces=torch.tensor([[0, 1, 2]], dtype=torch.long, device=device),
                    )
                    # Make sure we only feed back zero gradients to the field
                    # by masking out the final renderings of this mesh.
                    mesh_mask.append(False)
                else:
                    mesh_mask.append(True)
                raw_meshes.append(raw_mesh)
            mesh_mask = torch.tensor(mesh_mask, device=device)

        max_vertices = max(len(m.verts) for m in raw_meshes)

        fn = nerstf_fn if tf_fn is None else tf_fn
        tf_out = fn(
            query=Query(
                position=torch.stack(
                    [m.verts[torch.arange(0, max_vertices) % len(m.verts)] for m in raw_meshes],
                    dim=0,
                )
            ),
            query_batch_size=query_batch_size,
            options=options,
        )

        if "cache" in options:
            options.cache.fields = fields
            options.cache.raw_meshes = raw_meshes
            options.cache.raw_signed_distance = raw_signed_distance
            options.cache.raw_density = raw_density
            options.cache.mesh_mask = mesh_mask

    if output_srgb:
        tf_out.channels = _convert_srgb_to_linear(tf_out.channels)

    # Make sure the raw meshes have colors.
    with torch.autocast(device_type, enabled=False):
        textures = tf_out.channels.float()
        assert len(textures.shape) == 3 and textures.shape[-1] == len(
            texture_channels
        ), f"expected [meta_batch x inner_batch x texture_channels] field results, but got {textures.shape}"
        for m, texture in zip(raw_meshes, textures):
            texture = texture[: len(m.verts)]
            m.vertex_channels = {name: ch for name, ch in zip(texture_channels, texture.unbind(-1))}

    args = dict(
        options=options,
        texture_channels=texture_channels,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        camera=camera,
        batch_size=batch_size,
        inner_batch_size=inner_batch_size,
        inner_shape=inner_shape,
        raw_meshes=raw_meshes,
        tf_out=tf_out,
    )

    try:
        out = _render_with_pytorch3d(**args)
    except ModuleNotFoundError as exc:
        warnings.warn(f"exception rendering with PyTorch3D: {exc}")
        warnings.warn(
            "falling back on native PyTorch renderer, which does not support full gradients"
        )
        out = _render_with_raycast(**args)

    # Apply mask to prevent gradients for empty meshes.
    reshaped_mask = mesh_mask.view([-1] + [1] * (len(out.channels.shape) - 1))
    out.channels = torch.where(reshaped_mask, out.channels, torch.zeros_like(out.channels))
    out.transmittance = torch.where(
        reshaped_mask, out.transmittance, torch.ones_like(out.transmittance)
    )

    if output_srgb:
        out.channels = _convert_linear_to_srgb(out.channels)
    out.channels = out.channels * (1 - out.transmittance) * channel_scale.view(-1)

    # This might be useful information to have downstream
    out.raw_meshes = raw_meshes
    out.fields = fields
    out.mesh_mask = mesh_mask
    out.raw_signed_distance = raw_signed_distance
    out.aux_losses = AttrDict(cross_entropy=cross_entropy_sdf_loss(fields))
    if raw_density is not None:
        out.raw_density = raw_density

    return out


def _render_with_pytorch3d(
    options: AttrDict,
    texture_channels: Sequence[str],
    ambient_color: Union[float, Tuple[float]],
    diffuse_color: Union[float, Tuple[float]],
    specular_color: Union[float, Tuple[float]],
    camera: DifferentiableCamera,
    batch_size: int,
    inner_shape: Sequence[int],
    inner_batch_size: int,
    raw_meshes: List[TorchMesh],
    tf_out: AttrDict,
):
    _ = tf_out

    # Lazy import because pytorch3d is installed lazily.
    from shap_e.rendering.pytorch3d_util import (
        blender_uniform_lights,
        convert_cameras_torch,
        convert_meshes,
        render_images,
    )

    n_channels = len(texture_channels)
    device = camera.origin.device
    device_type = device.type

    with torch.autocast(device_type, enabled=False):
        meshes = convert_meshes(raw_meshes)

        lights = blender_uniform_lights(
            batch_size,
            device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
        )

        # Separate camera intrinsics for each view, so that we can
        # create a new camera for each batch of views.
        cam_shape = [batch_size, inner_batch_size, -1]
        position = camera.origin.reshape(cam_shape)
        x = camera.x.reshape(cam_shape)
        y = camera.y.reshape(cam_shape)
        z = camera.z.reshape(cam_shape)

        results = []
        for i in range(inner_batch_size):
            sub_cams = convert_cameras_torch(
                position[:, i], x[:, i], y[:, i], z[:, i], fov=camera.x_fov
            )
            imgs = render_images(
                camera.width,
                meshes,
                sub_cams,
                lights,
                use_checkpoint=options.checkpoint_render,
                **options.get("render_options", {}),
            )
            results.append(imgs)
        views = torch.stack(results, dim=1)
        views = views.view(batch_size, *inner_shape, camera.height, camera.width, n_channels + 1)

        out = AttrDict(
            channels=views[..., :-1],  # [batch_size, *inner_shape, height, width, n_channels]
            transmittance=1 - views[..., -1:],  # [batch_size, *inner_shape, height, width, 1]
            meshes=meshes,
        )

    return out


def _render_with_raycast(
    options: AttrDict,
    texture_channels: Sequence[str],
    ambient_color: Union[float, Tuple[float]],
    diffuse_color: Union[float, Tuple[float]],
    specular_color: Union[float, Tuple[float]],
    camera: DifferentiableCamera,
    batch_size: int,
    inner_shape: Sequence[int],
    inner_batch_size: int,
    raw_meshes: List[TorchMesh],
    tf_out: AttrDict,
):
    assert np.mean(np.array(specular_color)) == 0

    from shap_e.rendering.raycast.render import render_diffuse_mesh
    from shap_e.rendering.raycast.types import TriMesh as TorchTriMesh

    device = camera.origin.device
    device_type = device.type

    cam_shape = [batch_size, inner_batch_size, -1]
    origin = camera.origin.reshape(cam_shape)
    x = camera.x.reshape(cam_shape)
    y = camera.y.reshape(cam_shape)
    z = camera.z.reshape(cam_shape)

    with torch.autocast(device_type, enabled=False):
        all_meshes = []
        for i, mesh in enumerate(raw_meshes):
            all_meshes.append(
                TorchTriMesh(
                    faces=mesh.faces.long(),
                    vertices=mesh.verts.float(),
                    vertex_colors=tf_out.channels[i, : len(mesh.verts)].float(),
                )
            )
        all_images = []
        for i, mesh in enumerate(all_meshes):
            for j in range(inner_batch_size):
                all_images.append(
                    render_diffuse_mesh(
                        camera=ProjectiveCamera(
                            origin=origin[i, j].detach().cpu().numpy(),
                            x=x[i, j].detach().cpu().numpy(),
                            y=y[i, j].detach().cpu().numpy(),
                            z=z[i, j].detach().cpu().numpy(),
                            width=camera.width,
                            height=camera.height,
                            x_fov=camera.x_fov,
                            y_fov=camera.y_fov,
                        ),
                        mesh=mesh,
                        diffuse=float(np.array(diffuse_color).mean()),
                        ambient=float(np.array(ambient_color).mean()),
                        ray_batch_size=16,  # low memory usage
                        checkpoint=options.checkpoint_render,
                    )
                )

        n_channels = len(texture_channels)
        views = torch.stack(all_images).view(
            batch_size, *inner_shape, camera.height, camera.width, n_channels + 1
        )
        return AttrDict(
            channels=views[..., :-1],  # [batch_size, *inner_shape, height, width, n_channels]
            transmittance=1 - views[..., -1:],  # [batch_size, *inner_shape, height, width, 1]
            meshes=all_meshes,
        )


def _convert_srgb_to_linear(u: torch.Tensor) -> torch.Tensor:
    return torch.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _convert_linear_to_srgb(u: torch.Tensor) -> torch.Tensor:
    return torch.where(u <= 0.0031308, 12.92 * u, 1.055 * (u ** (1 / 2.4)) - 0.055)


def cross_entropy_sdf_loss(fields: torch.Tensor):
    logits = F.logsigmoid(fields)
    signs = (fields > 0).float()

    losses = []
    for dim in range(1, 4):
        n = logits.shape[dim]
        for (t_start, t_end, p_start, p_end) in [(0, -1, 1, n), (1, n, 0, -1)]:
            targets = slice_fields(signs, dim, t_start, t_end)
            preds = slice_fields(logits, dim, p_start, p_end)
            losses.append(
                F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
                .flatten(1)
                .mean()
            )
    return torch.stack(losses, dim=-1).sum()


def slice_fields(fields: torch.Tensor, dim: int, start: int, end: int):
    if dim == 1:
        return fields[:, start:end]
    elif dim == 2:
        return fields[:, :, start:end]
    elif dim == 3:
        return fields[:, :, :, start:end]
    else:
        raise ValueError(f"cannot slice dimension {dim}")


def volume_query_points(
    volume: Volume,
    grid_size: int,
):
    assert isinstance(volume, BoundingBoxVolume)
    indices = torch.arange(grid_size**3, device=volume.bbox_min.device)
    zs = indices % grid_size
    ys = torch.div(indices, grid_size, rounding_mode="trunc") % grid_size
    xs = torch.div(indices, grid_size**2, rounding_mode="trunc") % grid_size
    combined = torch.stack([xs, ys, zs], dim=1)
    return (combined.float() / (grid_size - 1)) * (
        volume.bbox_max - volume.bbox_min
    ) + volume.bbox_min
