from typing import Optional, Sequence

import torch

from shap_e.rendering.blender.constants import (
    BASIC_AMBIENT_COLOR,
    BASIC_DIFFUSE_COLOR,
    UNIFORM_LIGHT_DIRECTION,
)
from shap_e.rendering.view_data import ProjectiveCamera

from .cast import cast_camera
from .types import RayCollisions, TriMesh


def render_diffuse_mesh(
    camera: ProjectiveCamera,
    mesh: TriMesh,
    light_direction: Sequence[float] = tuple(UNIFORM_LIGHT_DIRECTION),
    diffuse: float = BASIC_DIFFUSE_COLOR,
    ambient: float = BASIC_AMBIENT_COLOR,
    ray_batch_size: Optional[int] = None,
    checkpoint: Optional[bool] = None,
) -> torch.Tensor:
    """
    Return an [H x W x 4] RGBA tensor of the rendered image.
    The pixels are floating points, with alpha in the range [0, 1] and the
    other colors matching the scale used by the mesh's vertex colors.
    """
    light_direction = torch.tensor(
        light_direction, device=mesh.vertices.device, dtype=mesh.vertices.dtype
    )

    all_collisions = RayCollisions.collect(
        cast_camera(
            camera=camera,
            mesh=mesh,
            ray_batch_size=ray_batch_size,
            checkpoint=checkpoint,
        )
    )
    num_rays = len(all_collisions.normals)
    if mesh.vertex_colors is None:
        vertex_colors = torch.tensor([[0.8, 0.8, 0.8]]).to(mesh.vertices).repeat(num_rays, 1)
    else:
        vertex_colors = mesh.vertex_colors

    light_coeffs = ambient + (
        diffuse * torch.sum(all_collisions.normals * light_direction, dim=-1).abs()
    )
    vertex_colors = mesh.vertex_colors[mesh.faces[all_collisions.tri_indices]]
    bary_products = torch.sum(vertex_colors * all_collisions.barycentric[..., None], axis=-2)
    out_colors = bary_products * light_coeffs[..., None]
    res = torch.where(all_collisions.collides[:, None], out_colors, torch.zeros_like(out_colors))
    return torch.cat([res, all_collisions.collides[:, None].float()], dim=-1).view(
        camera.height, camera.width, 4
    )
