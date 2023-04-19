from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import torch

from ._mc_table import MC_TABLE
from .torch_mesh import TorchMesh


def marching_cubes(
    field: torch.Tensor,
    min_point: torch.Tensor,
    size: torch.Tensor,
) -> TorchMesh:
    """
    For a signed distance field, produce a mesh using marching cubes.

    :param field: a 3D tensor of field values, where negative values correspond
                  to the outside of the shape. The dimensions correspond to the
                  x, y, and z directions, respectively.
    :param min_point: a tensor of shape [3] containing the point corresponding
                      to (0, 0, 0) in the field.
    :param size: a tensor of shape [3] containing the per-axis distance from the
                 (0, 0, 0) field corner and the (-1, -1, -1) field corner.
    """
    assert len(field.shape) == 3, "input must be a 3D scalar field"
    dev = field.device

    grid_size = field.shape
    grid_size_tensor = torch.tensor(grid_size).to(size)
    lut = _lookup_table(dev)

    # Create bitmasks between 0 and 255 (inclusive) indicating the state
    # of the eight corners of each cube.
    bitmasks = (field > 0).to(torch.uint8)
    bitmasks = bitmasks[:-1, :, :] | (bitmasks[1:, :, :] << 1)
    bitmasks = bitmasks[:, :-1, :] | (bitmasks[:, 1:, :] << 2)
    bitmasks = bitmasks[:, :, :-1] | (bitmasks[:, :, 1:] << 4)

    # Compute corner coordinates across the entire grid.
    corner_coords = torch.empty(*grid_size, 3, device=dev, dtype=field.dtype)
    corner_coords[range(grid_size[0]), :, :, 0] = torch.arange(
        grid_size[0], device=dev, dtype=field.dtype
    )[:, None, None]
    corner_coords[:, range(grid_size[1]), :, 1] = torch.arange(
        grid_size[1], device=dev, dtype=field.dtype
    )[:, None]
    corner_coords[:, :, range(grid_size[2]), 2] = torch.arange(
        grid_size[2], device=dev, dtype=field.dtype
    )

    # Compute all vertices across all edges in the grid, even though we will
    # throw some out later. We have (X-1)*Y*Z + X*(Y-1)*Z + X*Y*(Z-1) vertices.
    # These are all midpoints, and don't account for interpolation (which is
    # done later based on the used edge midpoints).
    edge_midpoints = torch.cat(
        [
            ((corner_coords[:-1] + corner_coords[1:]) / 2).reshape(-1, 3),
            ((corner_coords[:, :-1] + corner_coords[:, 1:]) / 2).reshape(-1, 3),
            ((corner_coords[:, :, :-1] + corner_coords[:, :, 1:]) / 2).reshape(-1, 3),
        ],
        dim=0,
    )

    # Create a flat array of [X, Y, Z] indices for each cube.
    cube_indices = torch.zeros(
        grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1, 3, device=dev, dtype=torch.long
    )
    cube_indices[range(grid_size[0] - 1), :, :, 0] = torch.arange(grid_size[0] - 1, device=dev)[
        :, None, None
    ]
    cube_indices[:, range(grid_size[1] - 1), :, 1] = torch.arange(grid_size[1] - 1, device=dev)[
        :, None
    ]
    cube_indices[:, :, range(grid_size[2] - 1), 2] = torch.arange(grid_size[2] - 1, device=dev)
    flat_cube_indices = cube_indices.reshape(-1, 3)

    # Create a flat array mapping each cube to 12 global edge indices.
    edge_indices = _create_flat_edge_indices(flat_cube_indices, grid_size)

    # Apply the LUT to figure out the triangles.
    flat_bitmasks = bitmasks.reshape(
        -1
    ).long()  # must cast to long for indexing to believe this not a mask
    local_tris = lut.cases[flat_bitmasks]
    local_masks = lut.masks[flat_bitmasks]
    # Compute the global edge indices for the triangles.
    global_tris = torch.gather(
        edge_indices, 1, local_tris.reshape(local_tris.shape[0], -1)
    ).reshape(local_tris.shape)
    # Select the used triangles for each cube.
    selected_tris = global_tris.reshape(-1, 3)[local_masks.reshape(-1)]

    # Now we have a bunch of indices into the full list of possible vertices,
    # but we want to reduce this list to only the used vertices.
    used_vertex_indices = torch.unique(selected_tris.view(-1))
    used_edge_midpoints = edge_midpoints[used_vertex_indices]
    old_index_to_new_index = torch.zeros(len(edge_midpoints), device=dev, dtype=torch.long)
    old_index_to_new_index[used_vertex_indices] = torch.arange(
        len(used_vertex_indices), device=dev, dtype=torch.long
    )

    # Rewrite the triangles to use the new indices
    selected_tris = torch.gather(old_index_to_new_index, 0, selected_tris.view(-1)).reshape(
        selected_tris.shape
    )

    # Compute the actual interpolated coordinates corresponding to edge midpoints.
    v1 = torch.floor(used_edge_midpoints).to(torch.long)
    v2 = torch.ceil(used_edge_midpoints).to(torch.long)
    s1 = field[v1[:, 0], v1[:, 1], v1[:, 2]]
    s2 = field[v2[:, 0], v2[:, 1], v2[:, 2]]
    p1 = (v1.float() / (grid_size_tensor - 1)) * size + min_point
    p2 = (v2.float() / (grid_size_tensor - 1)) * size + min_point
    # The signs of s1 and s2 should be different. We want to find
    # t such that t*s2 + (1-t)*s1 = 0.
    t = (s1 / (s1 - s2))[:, None]
    verts = t * p2 + (1 - t) * p1

    return TorchMesh(verts=verts, faces=selected_tris)


def _create_flat_edge_indices(
    flat_cube_indices: torch.Tensor, grid_size: Tuple[int, int, int]
) -> torch.Tensor:
    num_xs = (grid_size[0] - 1) * grid_size[1] * grid_size[2]
    y_offset = num_xs
    num_ys = grid_size[0] * (grid_size[1] - 1) * grid_size[2]
    z_offset = num_xs + num_ys
    return torch.stack(
        [
            # Edges spanning x-axis.
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2],
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]
            + flat_cube_indices[:, 2],
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            # Edges spanning y-axis.
            (
                y_offset
                + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
            ),
            (
                y_offset
                + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
            ),
            (
                y_offset
                + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
                + 1
            ),
            (
                y_offset
                + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
                + 1
            ),
            # Edges spanning z-axis.
            (
                z_offset
                + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
            (
                z_offset
                + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
            (
                z_offset
                + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)
                + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
            (
                z_offset
                + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)
                + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
        ],
        dim=-1,
    )


@dataclass
class McLookupTable:
    # Coordinates in triangles are represented as edge indices from 0-12
    # Here is an MC cell with both corner and edge indices marked.
    #        6 + ---------- 3 ----------+ 7
    #         /|                       /|
    #        6 |                      7 |
    #       /  |                     /  |
    #    4 +--------- 2 ------------+ 5 |
    #      |   10                   |   |
    #      |   |                    |   11
    #      |   |                    |   |
    #      8   | 2                  9   | 3
    #      |   +--------- 1 --------|---+
    #      |  /                     |  /
    #      | 4                      | 5
    #      |/                       |/
    #      +---------- 0 -----------+
    #     0                           1
    cases: torch.Tensor  # [256 x 5 x 3] long tensor
    masks: torch.Tensor  # [256 x 5] bool tensor


@lru_cache(maxsize=9)  # if there's more than 8 GPUs and a CPU, don't bother caching
def _lookup_table(device: torch.device) -> McLookupTable:
    cases = torch.zeros(256, 5, 3, device=device, dtype=torch.long)
    masks = torch.zeros(256, 5, device=device, dtype=torch.bool)

    edge_to_index = {
        (0, 1): 0,
        (2, 3): 1,
        (4, 5): 2,
        (6, 7): 3,
        (0, 2): 4,
        (1, 3): 5,
        (4, 6): 6,
        (5, 7): 7,
        (0, 4): 8,
        (1, 5): 9,
        (2, 6): 10,
        (3, 7): 11,
    }

    for i, case in enumerate(MC_TABLE):
        for j, tri in enumerate(case):
            for k, (c1, c2) in enumerate(zip(tri[::2], tri[1::2])):
                cases[i, j, k] = edge_to_index[(c1, c2) if c1 < c2 else (c2, c1)]
            masks[i, j] = True
    return McLookupTable(cases=cases, masks=masks)
