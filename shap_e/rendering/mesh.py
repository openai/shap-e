from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Optional, Union

import blobfile as bf
import numpy as np

from .ply_util import write_ply


@dataclass
class TriMesh:
    """
    A 3D triangle mesh with optional data at the vertices and faces.
    """

    # [N x 3] array of vertex coordinates.
    verts: np.ndarray

    # [M x 3] array of triangles, pointing to indices in verts.
    faces: np.ndarray

    # [P x 3] array of normal vectors per face.
    normals: Optional[np.ndarray] = None

    # Extra data per vertex and face.
    vertex_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)
    face_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "TriMesh":
        """
        Load the mesh from a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            verts = obj["verts"]
            faces = obj["faces"]
            normals = obj["normals"] if "normals" in keys else None
            vertex_channels = {}
            face_channels = {}
            for key in keys:
                if key.startswith("v_"):
                    vertex_channels[key[2:]] = obj[key]
                elif key.startswith("f_"):
                    face_channels[key[2:]] = obj[key]
            return cls(
                verts=verts,
                faces=faces,
                normals=normals,
                vertex_channels=vertex_channels,
                face_channels=face_channels,
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the mesh to a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "wb") as writer:
                self.save(writer)
        else:
            obj_dict = dict(verts=self.verts, faces=self.faces)
            if self.normals is not None:
                obj_dict["normals"] = self.normals
            for k, v in self.vertex_channels.items():
                obj_dict[f"v_{k}"] = v
            for k, v in self.face_channels.items():
                obj_dict[f"f_{k}"] = v
            np.savez(f, **obj_dict)

    def has_vertex_colors(self) -> bool:
        return self.vertex_channels is not None and all(x in self.vertex_channels for x in "RGB")

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.verts,
            rgb=(
                np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
                if self.has_vertex_colors()
                else None
            ),
            faces=self.faces,
        )

    def write_obj(self, raw_f: BinaryIO):
        if self.has_vertex_colors():
            vertex_colors = np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
            vertices = [
                "{} {} {} {} {} {}".format(*coord, *color)
                for coord, color in zip(self.verts.tolist(), vertex_colors.tolist())
            ]
        else:
            vertices = ["{} {} {}".format(*coord) for coord in self.verts.tolist()]

        faces = [
            "f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1))
            for tri in self.faces.tolist()
        ]

        combined_data = ["v " + vertex for vertex in vertices] + faces

        raw_f.writelines("\n".join(combined_data))
