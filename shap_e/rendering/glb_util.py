import numpy as np
import pygltflib

from typing import BinaryIO, Optional

from shap_e.util.io import buffered_writer


def write_glb(
    raw_f: BinaryIO,
    coords: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
):
    coords = np.asarray(coords, dtype=np.float32)
    coords_binary_blob = coords.tobytes()

    # setting gltf bufferviews and accessors and mesh primitives
    primitive = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=0),
        mode=0,
    )
    bufferviews = [
        pygltflib.BufferView(
            buffer=0,
            byteLength=len(coords_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
        ),
    ]
    accessors = [
        pygltflib.Accessor(
            bufferView=0,
            componentType=pygltflib.FLOAT,
            count=len(coords),
            type=pygltflib.VEC3,
            max=coords.max(axis=0).tolist(),
            min=coords.min(axis=0).tolist(),
        ),
    ]

    # adding faces and rgb if exists
    indx = 1
    faces_binary_blob = b''
    if faces is not None:
        faces = np.asarray(faces, dtype=np.uint32)
        faces_binary_blob = faces.flatten().tobytes()
        primitive.indices = indx
        primitive.mode = 4
        accessors.append(
            pygltflib.Accessor(
                bufferView=indx,
                componentType=pygltflib.UNSIGNED_INT,
                count=faces.size,
                type=pygltflib.SCALAR,
                max=[int(faces.max())],
                min=[int(faces.min())],
            )
        )
        bufferviews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(coords_binary_blob),
                byteLength=len(faces_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            )
        )
        indx += 1

    rgb_binary_blob = b''
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.float32)
        rgb_binary_blob = rgb.tobytes()
        primitive.attributes.COLOR_0 = indx
        accessors.append(
            pygltflib.Accessor(
                bufferView=indx,
                componentType=pygltflib.FLOAT,
                count=len(rgb),
                type=pygltflib.VEC3,
                max=rgb.max(axis=0).tolist(),
                min=rgb.min(axis=0).tolist(),
            )
        )
        bufferviews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=(
                    len(faces_binary_blob)
                    + len(coords_binary_blob)
                ),
                byteLength=len(rgb_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            )
        )

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[primitive]
            )
        ],
        accessors=accessors,
        bufferViews=bufferviews,
        buffers=[
            pygltflib.Buffer(
                byteLength=(
                    len(faces_binary_blob)
                    + len(coords_binary_blob)
                    + len(rgb_binary_blob)
                )
            )
        ],
    )
    gltf.set_binary_blob(
        coords_binary_blob
        + faces_binary_blob
        + rgb_binary_blob
    )

    with buffered_writer(raw_f) as fio:
        fio.write(b"".join(gltf.save_to_bytes()))
