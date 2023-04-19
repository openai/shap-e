import copy
import inspect
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.structures import Meshes

from shap_e.models.nn.checkpoint import checkpoint

from .blender.constants import BASIC_AMBIENT_COLOR, BASIC_DIFFUSE_COLOR, UNIFORM_LIGHT_DIRECTION
from .torch_mesh import TorchMesh
from .view_data import ProjectiveCamera

# Using a lower value like 1e-4 seems to result in weird issues
# for our high-poly meshes.
DEFAULT_RENDER_SIGMA = 1e-5

DEFAULT_RENDER_GAMMA = 1e-4


def render_images(
    image_size: int,
    meshes: Meshes,
    cameras: Any,
    lights: Any,
    sigma: float = DEFAULT_RENDER_SIGMA,
    gamma: float = DEFAULT_RENDER_GAMMA,
    max_faces_per_bin=100000,
    faces_per_pixel=50,
    bin_size=None,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    if use_checkpoint:
        # Decompose all of our arguments into a bunch of tensor lists
        # so that autograd can keep track of what the op depends on.
        verts_list = meshes.verts_list()
        faces_list = meshes.faces_list()
        assert isinstance(meshes.textures, TexturesVertex)
        assert isinstance(lights, BidirectionalLights)
        textures = meshes.textures.verts_features_padded()
        light_vecs, light_fn = _deconstruct_tensor_props(lights)
        camera_vecs, camera_fn = _deconstruct_tensor_props(cameras)

        def ckpt_fn(
            *args: torch.Tensor,
            num_verts=len(verts_list),
            num_light_vecs=len(light_vecs),
            num_camera_vecs=len(camera_vecs),
            light_fn=light_fn,
            camera_fn=camera_fn,
            faces_list=faces_list
        ):
            args = list(args)
            verts_list = args[:num_verts]
            del args[:num_verts]
            light_vecs = args[:num_light_vecs]
            del args[:num_light_vecs]
            camera_vecs = args[:num_camera_vecs]
            del args[:num_camera_vecs]
            textures = args.pop(0)

            meshes = Meshes(verts=verts_list, faces=faces_list, textures=TexturesVertex(textures))
            lights = light_fn(light_vecs)
            cameras = camera_fn(camera_vecs)
            return render_images(
                image_size=image_size,
                meshes=meshes,
                cameras=cameras,
                lights=lights,
                sigma=sigma,
                gamma=gamma,
                max_faces_per_bin=max_faces_per_bin,
                faces_per_pixel=faces_per_pixel,
                bin_size=bin_size,
                use_checkpoint=False,
            )

        result = checkpoint(ckpt_fn, (*verts_list, *light_vecs, *camera_vecs, textures), (), True)
    else:
        raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=max_faces_per_bin,
            bin_size=bin_size,
            perspective_correct=False,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_soft),
            shader=SoftPhongShader(
                device=meshes.device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(sigma=sigma, gamma=gamma, background_color=(0, 0, 0)),
            ),
        )
        result = renderer(meshes)

    return result


def _deconstruct_tensor_props(
    props: TensorProperties,
) -> Tuple[List[torch.Tensor], Callable[[List[torch.Tensor]], TensorProperties]]:
    vecs = []
    names = []
    other_props = {}
    for k in dir(props):
        if k.startswith("__"):
            continue
        v = getattr(props, k)
        if inspect.ismethod(v):
            continue
        if torch.is_tensor(v):
            vecs.append(v)
            names.append(k)
        else:
            other_props[k] = v

    def recreate_fn(vecs_arg):
        other = type(props)(device=props.device)
        for k, v in other_props.items():
            setattr(other, k, copy.deepcopy(v))
        for name, vec in zip(names, vecs_arg):
            setattr(other, name, vec)
        return other

    return vecs, recreate_fn



def convert_meshes(raw_meshes: Sequence[TorchMesh], default_brightness=0.8) -> Meshes:
    meshes = Meshes(
        verts=[mesh.verts for mesh in raw_meshes], faces=[mesh.faces for mesh in raw_meshes]
    )
    rgbs = []
    for mesh in raw_meshes:
        if mesh.vertex_channels and all(k in mesh.vertex_channels for k in "RGB"):
            rgbs.append(torch.stack([mesh.vertex_channels[k] for k in "RGB"], axis=-1))
        else:
            rgbs.append(
                torch.ones(
                    len(mesh.verts) * default_brightness,
                    3,
                    device=mesh.verts.device,
                    dtype=mesh.verts.dtype,
                )
            )
    meshes.textures = TexturesVertex(verts_features=rgbs)
    return meshes


def convert_cameras(
    cameras: Sequence[ProjectiveCamera], device: torch.device
) -> FoVPerspectiveCameras:
    Rs = []
    Ts = []
    for camera in cameras:
        assert (
            camera.width == camera.height and camera.x_fov == camera.y_fov
        ), "viewports must be square"
        assert camera.x_fov == cameras[0].x_fov, "all cameras must have same field-of-view"
        R = np.stack([-camera.x, -camera.y, camera.z], axis=0).T
        T = -R.T @ camera.origin
        Rs.append(R)
        Ts.append(T)
    return FoVPerspectiveCameras(
        R=np.stack(Rs, axis=0),
        T=np.stack(Ts, axis=0),
        fov=cameras[0].x_fov,
        degrees=False,
        device=device,
    )


def convert_cameras_torch(
    origins: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, zs: torch.Tensor, fov: float
) -> FoVPerspectiveCameras:
    Rs = []
    Ts = []
    for origin, x, y, z in zip(origins, xs, ys, zs):
        R = torch.stack([-x, -y, z], axis=0).T
        T = -R.T @ origin
        Rs.append(R)
        Ts.append(T)
    return FoVPerspectiveCameras(
        R=torch.stack(Rs, dim=0),
        T=torch.stack(Ts, dim=0),
        fov=fov,
        degrees=False,
        device=origins.device,
    )


def blender_uniform_lights(
    batch_size: int,
    device: torch.device,
    ambient_color: Union[float, Tuple[float]] = BASIC_AMBIENT_COLOR,
    diffuse_color: Union[float, Tuple[float]] = BASIC_DIFFUSE_COLOR,
    specular_color: Union[float, Tuple[float]] = 0.0,
) -> "BidirectionalLights":
    """
    Create a light that attempts to match the light used by the Blender
    renderer when run with `--light_mode basic`.
    """
    if isinstance(ambient_color, float):
        ambient_color = (ambient_color,) * 3
    if isinstance(diffuse_color, float):
        diffuse_color = (diffuse_color,) * 3
    if isinstance(specular_color, float):
        specular_color = (specular_color,) * 3
    return BidirectionalLights(
        ambient_color=(ambient_color,) * batch_size,
        diffuse_color=(diffuse_color,) * batch_size,
        specular_color=(specular_color,) * batch_size,
        direction=(UNIFORM_LIGHT_DIRECTION,) * batch_size,
        device=device,
    )


class BidirectionalLights(DirectionalLights):
    """
    Adapted from here, but effectively shines the light in both positive and negative directions:
    https://github.com/facebookresearch/pytorch3d/blob/efea540bbcab56fccde6f4bc729d640a403dac56/pytorch3d/renderer/lighting.py#L159
    """

    def diffuse(self, normals, points=None) -> torch.Tensor:
        return torch.maximum(
            super().diffuse(normals, points=points), super().diffuse(-normals, points=points)
        )

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.maximum(
            super().specular(normals, points, camera_position, shininess),
            super().specular(-normals, points, camera_position, shininess),
        )
