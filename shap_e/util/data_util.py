import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional, Union

import blobfile as bf
import numpy as np
import torch
from PIL import Image

from shap_e.rendering.blender.render import render_mesh, render_model
from shap_e.rendering.blender.view_data import BlenderViewData
from shap_e.rendering.mesh import TriMesh
from shap_e.rendering.point_cloud import PointCloud
from shap_e.rendering.view_data import ViewData
from shap_e.util.collections import AttrDict
from shap_e.util.image_util import center_crop, get_alpha, remove_alpha, resize


def load_or_create_multimodal_batch(
    device: torch.device,
    *,
    mesh_path: Optional[str] = None,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    point_count: int = 2**14,
    random_sample_count: int = 2**19,
    pc_num_views: int = 40,
    mv_light_mode: Optional[str] = None,
    mv_num_views: int = 20,
    mv_image_size: int = 512,
    mv_alpha_removal: str = "black",
    verbose: bool = False,
) -> AttrDict:
    if verbose:
        print("creating point cloud...")
    pc = load_or_create_pc(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        random_sample_count=random_sample_count,
        point_count=point_count,
        num_views=pc_num_views,
        verbose=verbose,
    )
    raw_pc = np.concatenate([pc.coords, pc.select_channels(["R", "G", "B"])], axis=-1)
    encode_me = torch.from_numpy(raw_pc).float().to(device)
    batch = AttrDict(points=encode_me.t()[None])
    if mv_light_mode:
        if verbose:
            print("creating multiview...")
        with load_or_create_multiview(
            mesh_path=mesh_path,
            model_path=model_path,
            cache_dir=cache_dir,
            num_views=mv_num_views,
            extract_material=False,
            light_mode=mv_light_mode,
            verbose=verbose,
        ) as mv:
            cameras, views, view_alphas, depths = [], [], [], []
            for view_idx in range(mv.num_views):
                camera, view = mv.load_view(
                    view_idx,
                    ["R", "G", "B", "A"] if "A" in mv.channel_names else ["R", "G", "B"],
                )
                depth = None
                if "D" in mv.channel_names:
                    _, depth = mv.load_view(view_idx, ["D"])
                    depth = process_depth(depth, mv_image_size)
                view, alpha = process_image(
                    np.round(view * 255.0).astype(np.uint8), mv_alpha_removal, mv_image_size
                )
                camera = camera.center_crop().resize_image(mv_image_size, mv_image_size)
                cameras.append(camera)
                views.append(view)
                view_alphas.append(alpha)
                depths.append(depth)
            batch.depths = [depths]
            batch.views = [views]
            batch.view_alphas = [view_alphas]
            batch.cameras = [cameras]
    return normalize_input_batch(batch, pc_scale=2.0, color_scale=1.0 / 255.0)


def load_or_create_pc(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    random_sample_count: int,
    point_count: int,
    num_views: int,
    verbose: bool = False,
) -> PointCloud:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if cache_dir is not None:
        cache_path = bf.join(
            cache_dir,
            f"pc_{bf.basename(path)}_mat_{num_views}_{random_sample_count}_{point_count}.npz",
        )
        if bf.exists(cache_path):
            return PointCloud.load(cache_path)
    else:
        cache_path = None

    with load_or_create_multiview(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        num_views=num_views,
        verbose=verbose,
    ) as mv:
        if verbose:
            print("extracting point cloud from multiview...")
        pc = mv_to_pc(
            multiview=mv, random_sample_count=random_sample_count, point_count=point_count
        )
        if cache_path is not None:
            pc.save(cache_path)
        return pc


@contextmanager
def load_or_create_multiview(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    num_views: int = 20,
    extract_material: bool = True,
    light_mode: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[BlenderViewData]:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if extract_material:
        assert light_mode is None, "light_mode is ignored when extract_material=True"
    else:
        assert light_mode is not None, "must specify light_mode when extract_material=False"

    if cache_dir is not None:
        if extract_material:
            cache_path = bf.join(cache_dir, f"mv_{bf.basename(path)}_mat_{num_views}.zip")
        else:
            cache_path = bf.join(cache_dir, f"mv_{bf.basename(path)}_{light_mode}_{num_views}.zip")
        if bf.exists(cache_path):
            with bf.BlobFile(cache_path, "rb") as f:
                yield BlenderViewData(f)
                return
    else:
        cache_path = None

    common_kwargs = dict(
        fast_mode=True,
        extract_material=extract_material,
        camera_pose="random",
        light_mode=light_mode or "uniform",
        verbose=verbose,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = bf.join(tmp_dir, "out.zip")
        if mesh_path is not None:
            mesh = TriMesh.load(mesh_path)
            render_mesh(
                mesh=mesh,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        elif model_path is not None:
            render_model(
                model_path,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        if cache_path is not None:
            bf.copy(tmp_path, cache_path)
        with bf.BlobFile(tmp_path, "rb") as f:
            yield BlenderViewData(f)


def mv_to_pc(multiview: ViewData, random_sample_count: int, point_count: int) -> PointCloud:
    pc = PointCloud.from_rgbd(multiview)

    # Handle empty samples.
    if len(pc.coords) == 0:
        pc = PointCloud(
            coords=np.zeros([1, 3]),
            channels=dict(zip("RGB", np.zeros([3, 1]))),
        )
    while len(pc.coords) < point_count:
        pc = pc.combine(pc)
        # Prevent duplicate points; some models may not like it.
        pc.coords += np.random.normal(size=pc.coords.shape) * 1e-4

    pc = pc.random_sample(random_sample_count)
    pc = pc.farthest_point_sample(point_count, average_neighbors=True)

    return pc


def normalize_input_batch(batch: AttrDict, *, pc_scale: float, color_scale: float) -> AttrDict:
    res = batch.copy()
    scale_vec = torch.tensor([*([pc_scale] * 3), *([color_scale] * 3)], device=batch.points.device)
    res.points = res.points * scale_vec[:, None]

    if "cameras" in res:
        res.cameras = [[cam.scale_scene(pc_scale) for cam in cams] for cams in res.cameras]

    if "depths" in res:
        res.depths = [[depth * pc_scale for depth in depths] for depths in res.depths]

    return res


def process_depth(depth_img: np.ndarray, image_size: int) -> np.ndarray:
    depth_img = center_crop(depth_img)
    depth_img = resize(depth_img, width=image_size, height=image_size)
    return np.squeeze(depth_img)


def process_image(
    img_or_img_arr: Union[Image.Image, np.ndarray], alpha_removal: str, image_size: int
):
    if isinstance(img_or_img_arr, np.ndarray):
        img = Image.fromarray(img_or_img_arr)
        img_arr = img_or_img_arr
    else:
        img = img_or_img_arr
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            # Grayscale
            rgb = Image.new("RGB", img.size)
            rgb.paste(img)
            img = rgb
            img_arr = np.array(img)

    img = center_crop(img)
    alpha = get_alpha(img)
    img = remove_alpha(img, mode=alpha_removal)
    alpha = alpha.resize((image_size,) * 2, resample=Image.BILINEAR)
    img = img.resize((image_size,) * 2, resample=Image.BILINEAR)
    return img, alpha
