from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .mesh import TriMesh


@dataclass
class TorchMesh:
    """
    A 3D triangle mesh with optional data at the vertices and faces.
    """

    # [N x 3] array of vertex coordinates.
    verts: torch.Tensor

    # [M x 3] array of triangles, pointing to indices in verts.
    faces: torch.Tensor

    # Extra data per vertex and face.
    vertex_channels: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)
    face_channels: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)

    def tri_mesh(self) -> TriMesh:
        """
        Create a CPU version of the mesh.
        """
        return TriMesh(
            verts=self.verts.detach().cpu().numpy(),
            faces=self.faces.cpu().numpy(),
            vertex_channels=(
                {k: v.detach().cpu().numpy() for k, v in self.vertex_channels.items()}
                if self.vertex_channels is not None
                else None
            ),
            face_channels=(
                {k: v.detach().cpu().numpy() for k, v in self.face_channels.items()}
                if self.face_channels is not None
                else None
            ),
        )
