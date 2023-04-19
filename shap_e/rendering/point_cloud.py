import random
from collections import defaultdict
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Union

import blobfile as bf
import numpy as np

from shap_e.rendering.view_data import ViewData

from .ply_util import write_ply

COLORS = frozenset(["R", "G", "B", "A"])


def preprocess(data, channel):
    if channel in COLORS:
        return np.round(data * 255.0)
    return data


@dataclass
class PointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray]

    @classmethod
    def from_rgbd(cls, vd: ViewData, num_views: Optional[int] = None) -> "PointCloud":
        """
        Construct a point cloud from the given view data.

        The data must have a depth channel. All other channels will be stored
        in the `channels` attribute of the result.

        Pixels in the rendered views are not converted into points in the cloud
        if they have infinite depth or less than 1.0 alpha.
        """
        channel_names = vd.channel_names
        if "D" not in channel_names:
            raise ValueError(f"view data must have depth channel")
        depth_index = channel_names.index("D")

        all_coords = []
        all_channels = defaultdict(list)

        if num_views is None:
            num_views = vd.num_views
        for i in range(num_views):
            camera, channel_values = vd.load_view(i, channel_names)
            flat_values = channel_values.reshape([-1, len(channel_names)])

            # Create an array of integer (x, y) image coordinates for Camera methods.
            image_coords = camera.image_coords()

            # Select subset of pixels that have meaningful depth/color.
            image_mask = np.isfinite(flat_values[:, depth_index])
            if "A" in channel_names:
                image_mask = image_mask & (flat_values[:, channel_names.index("A")] >= 1 - 1e-5)
            image_coords = image_coords[image_mask]
            flat_values = flat_values[image_mask]

            # Use the depth and camera information to compute the coordinates
            # corresponding to every visible pixel.
            camera_rays = camera.camera_rays(image_coords)
            camera_origins = camera_rays[:, 0]
            camera_directions = camera_rays[:, 1]
            depth_dirs = camera.depth_directions(image_coords)
            ray_scales = flat_values[:, depth_index] / np.sum(
                camera_directions * depth_dirs, axis=-1
            )
            coords = camera_origins + camera_directions * ray_scales[:, None]

            all_coords.append(coords)
            for j, name in enumerate(channel_names):
                if name != "D":
                    all_channels[name].append(flat_values[:, j])

        if len(all_coords) == 0:
            return cls(coords=np.zeros([0, 3], dtype=np.float32), channels={})

        return cls(
            coords=np.concatenate(all_coords, axis=0),
            channels={k: np.concatenate(v, axis=0) for k, v in all_channels.items()},
        )

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.coords,
            rgb=(
                np.stack([self.channels[x] for x in "RGB"], axis=1)
                if all(x in self.channels for x in "RGB")
                else None
            ),
        )

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx

            # Without this line, we may duplicate an index more than once if
            # there are duplicate points, due to rounding errors.
            cur_dists[idx] = -1

            cur_dists = np.minimum(cur_dists, compute_dists(idx))

        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )
