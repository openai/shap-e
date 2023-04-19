from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

import shap_e.rendering.mesh

from ._utils import cross_product, normalize


@dataclass
class Rays:
    """
    A ray in ray casting.
    """

    origins: torch.Tensor  # [N x 3] float tensor
    directions: torch.Tensor  # [N x 3] float tensor

    def normalized_directions(self) -> torch.Tensor:
        return normalize(self.directions)


@dataclass
class RayCollisions:
    """
    The result of casting N rays onto a mesh.
    """

    collides: torch.Tensor  # [N] boolean tensor
    ray_dists: torch.Tensor  # [N] float tensor
    tri_indices: torch.Tensor  # [N] long tensor
    barycentric: torch.Tensor  # [N x 3] float tensor
    normals: torch.Tensor  # [N x 3] float tensor

    @classmethod
    def collect(cls, it: Iterable["RayCollisions"]) -> "RayCollisions":
        res = None
        for x in it:
            if res is None:
                res = x
            else:
                res = cls(
                    collides=torch.cat([res.collides, x.collides]),
                    ray_dists=torch.cat([res.ray_dists, x.ray_dists]),
                    tri_indices=torch.cat([res.tri_indices, x.tri_indices]),
                    barycentric=torch.cat([res.barycentric, x.barycentric]),
                    normals=torch.cat([res.normals, x.normals]),
                )
        if res is None:
            raise ValueError("cannot collect an empty iterable of RayCollisions")
        return res


@dataclass
class TriMesh:
    faces: torch.Tensor  # [N x 3] long tensor
    vertices: torch.Tensor  # [N x 3] float tensor

    vertex_colors: Optional[torch.Tensor] = None

    def normals(self) -> torch.Tensor:
        """
        Returns an [N x 3] batch of normal vectors per triangle assuming the
        right-hand rule.
        """
        tris = self.vertices[self.faces]
        v1 = tris[:, 1] - tris[:, 0]
        v2 = tris[:, 2] - tris[:, 0]
        return normalize(cross_product(v1, v2))

    @classmethod
    def from_numpy(cls, x: shap_e.rendering.mesh.TriMesh) -> "TriMesh":
        vertex_colors = None
        if all(ch in x.vertex_channels for ch in "RGB"):
            vertex_colors = torch.from_numpy(
                np.stack([x.vertex_channels[ch] for ch in "RGB"], axis=-1)
            )
        return cls(
            faces=torch.from_numpy(x.faces),
            vertices=torch.from_numpy(x.verts),
            vertex_colors=vertex_colors,
        )

    def to(self, *args, **kwargs) -> "TriMesh":
        return TriMesh(
            faces=self.faces.to(*args, **kwargs),
            vertices=self.vertices.to(*args, **kwargs),
            vertex_colors=None
            if self.vertex_colors is None
            else self.vertex_colors.to(*args, **kwargs),
        )
