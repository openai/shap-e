from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from shap_e.rendering.view_data import ProjectiveCamera

from ._utils import cross_product
from .types import RayCollisions, Rays, TriMesh


def cast_camera(
    camera: ProjectiveCamera,
    mesh: TriMesh,
    ray_batch_size: Optional[int] = None,
    checkpoint: Optional[bool] = None,
) -> Iterator[RayCollisions]:
    pixel_indices = np.arange(camera.width * camera.height)
    image_coords = np.stack([pixel_indices % camera.width, pixel_indices // camera.width], axis=1)
    rays = camera.camera_rays(image_coords)
    batch_size = ray_batch_size or len(rays)
    checkpoint = checkpoint if checkpoint is not None else batch_size < len(rays)
    for i in range(0, len(rays), batch_size):
        sub_rays = rays[i : i + batch_size]
        origins = torch.from_numpy(sub_rays[:, 0]).to(mesh.vertices)
        directions = torch.from_numpy(sub_rays[:, 1]).to(mesh.vertices)
        yield cast_rays(Rays(origins=origins, directions=directions), mesh, checkpoint=checkpoint)


def cast_rays(rays: Rays, mesh: TriMesh, checkpoint: bool = False) -> RayCollisions:
    """
    Cast a batch of rays onto a mesh.
    """
    if checkpoint:
        collides, ray_dists, tri_indices, barycentric, normals = RayCollisionFunction.apply(
            rays.origins, rays.directions, mesh.faces, mesh.vertices
        )
        return RayCollisions(
            collides=collides,
            ray_dists=ray_dists,
            tri_indices=tri_indices,
            barycentric=barycentric,
            normals=normals,
        )

    # https://github.com/unixpickle/vae-textures/blob/2968549ddd4a3487f9437d4db00793324453cd59/vae_textures/render.py#L98
    normals = mesh.normals()  # [N x 3]
    directions = rays.directions  # [M x 3]
    collides = (directions @ normals.T).abs() > 1e-8  # [N x M]

    tris = mesh.vertices[mesh.faces]  # [N x 3 x 3]
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]

    cross1 = cross_product(directions[:, None], v2[None])  # [N x M x 3]
    det = torch.sum(cross1 * v1[None], dim=-1)  # [N x M]
    collides = torch.logical_and(collides, det.abs() > 1e-8)

    invDet = 1 / det  # [N x M]
    o = rays.origins[:, None] - tris[None, :, 0]  # [N x M x 3]
    bary1 = invDet * torch.sum(o * cross1, dim=-1)  # [N x M]
    collides = torch.logical_and(collides, torch.logical_and(bary1 >= 0, bary1 <= 1))

    cross2 = cross_product(o, v1[None])  # [N x M x 3]
    bary2 = invDet * torch.sum(directions[:, None] * cross2, dim=-1)  # [N x M]
    collides = torch.logical_and(collides, torch.logical_and(bary2 >= 0, bary2 <= 1))

    bary0 = 1 - (bary1 + bary2)

    # Make sure this is in the positive part of the ray.
    scale = invDet * torch.sum(v2 * cross2, dim=-1)
    collides = torch.logical_and(collides, scale > 0)

    # Select the nearest collision
    ray_dists, tri_indices = torch.min(
        torch.where(collides, scale, torch.tensor(torch.inf).to(scale)), dim=-1
    )  # [N]
    nearest_bary = torch.stack(
        [
            bary0[range(len(tri_indices)), tri_indices],
            bary1[range(len(tri_indices)), tri_indices],
            bary2[range(len(tri_indices)), tri_indices],
        ],
        dim=-1,
    )

    return RayCollisions(
        collides=torch.any(collides, dim=-1),
        ray_dists=ray_dists,
        tri_indices=tri_indices,
        barycentric=nearest_bary,
        normals=normals[tri_indices],
    )


class RayCollisionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, origins, directions, faces, vertices
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(origins, directions, faces, vertices)
        with torch.no_grad():
            res = cast_rays(
                Rays(origins=origins, directions=directions),
                TriMesh(faces=faces, vertices=vertices),
                checkpoint=False,
            )
        return (res.collides, res.ray_dists, res.tri_indices, res.barycentric, res.normals)

    @staticmethod
    def backward(
        ctx, _collides_grad, ray_dists_grad, _tri_indices_grad, barycentric_grad, normals_grad
    ):
        origins, directions, faces, vertices = ctx.input_tensors

        origins = origins.detach().requires_grad_(True)
        directions = directions.detach().requires_grad_(True)
        vertices = vertices.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = cast_rays(
                Rays(origins=origins, directions=directions),
                TriMesh(faces=faces, vertices=vertices),
                checkpoint=False,
            )

        origins_grad, directions_grad, vertices_grad = torch.autograd.grad(
            (outputs.ray_dists, outputs.barycentric, outputs.normals),
            (origins, directions, vertices),
            (ray_dists_grad, barycentric_grad, normals_grad),
        )
        return (origins_grad, directions_grad, None, vertices_grad)
