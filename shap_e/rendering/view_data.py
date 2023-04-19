from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Camera(ABC):
    """
    An object describing how a camera corresponds to pixels in an image.
    """

    @abstractmethod
    def image_coords(self) -> np.ndarray:
        """
        :return: ([self.height, self.width, 2]).reshape(self.height * self.width, 2) image coordinates
        """

    @abstractmethod
    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, get the direction that
        corresponds to "depth" in an RGBD rendering.

        This may raise an exception if there is no "D" channel in the
        corresponding ViewData.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 3] array of normalized depth directions.
        """
        _ = coords
        raise NotImplementedError

    @abstractmethod
    def center_crop(self) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with a center crop to a square of the smaller dimension.
        """

    @abstractmethod
    def resize_image(self, width: int, height: int) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with resized image dimensions.
        """

    @abstractmethod
    def scale_scene(self, factor: float) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with the scene rescaled by the given factor.
        """


@dataclass
class ProjectiveCamera(Camera):
    """
    A Camera implementation for a standard pinhole camera.

    The camera rays shoot away from the origin in the z direction, with the x
    and y directions corresponding to the positive horizontal and vertical axes
    in image space.
    """

    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    width: int
    height: int
    x_fov: float
    y_fov: float

    def image_coords(self) -> np.ndarray:
        ind = np.arange(self.width * self.height)
        coords = np.stack([ind % self.width, ind // self.width], axis=1).astype(np.float32)
        return coords

    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        fracs = (coords / (np.array([self.width, self.height], dtype=np.float32) - 1)) * 2 - 1
        fracs = fracs * np.tan(np.array([self.x_fov, self.y_fov]) / 2)
        directions = self.z + self.x * fracs[:, :1] + self.y * fracs[:, 1:]
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        return np.stack([np.broadcast_to(self.origin, directions.shape), directions], axis=1)

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        return np.tile((self.z / np.linalg.norm(self.z))[None], [len(coords), 1])

    def resize_image(self, width: int, height: int) -> "ProjectiveCamera":
        """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
        assert width * self.height == height * self.width, "The aspect ratio should not change."
        return ProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=width,
            height=height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )

    def center_crop(self) -> "ProjectiveCamera":
        """
        Creates a new camera for the center-cropped view
        """
        size = min(self.width, self.height)
        fov = min(self.x_fov, self.y_fov)
        return ProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=size,
            height=size,
            x_fov=fov,
            y_fov=fov,
        )

    def scale_scene(self, factor: float) -> "ProjectiveCamera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with the camera frame rescaled by the given factor.
        """
        return ProjectiveCamera(
            origin=self.origin * factor,
            x=self.x,
            y=self.y,
            z=self.z,
            width=self.width,
            height=self.height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )


class ViewData(ABC):
    """
    A collection of rendered camera views of a scene or object.

    This is a generalization of a NeRF dataset, since NeRF datasets only encode
    RGB or RGBA data, whereas this dataset supports arbitrary channels.
    """

    @property
    @abstractmethod
    def num_views(self) -> int:
        """
        The number of rendered views.
        """

    @property
    @abstractmethod
    def channel_names(self) -> List[str]:
        """
        Get all of the supported channels available for the views.

        This can be arbitrary, but there are some standard names:
        "R", "G", "B", "A" (alpha), and "D" (depth).
        """

    @abstractmethod
    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        """
        Load the given channels from the view at the given index.

        :return: a tuple (camera_view, data), where data is a float array of
                 shape [height x width x num_channels].
        """


class MemoryViewData(ViewData):
    """
    A ViewData that is implemented in memory.
    """

    def __init__(self, channels: Dict[str, np.ndarray], cameras: List[Camera]):
        assert all(v.shape[0] == len(cameras) for v in channels.values())
        self.channels = channels
        self.cameras = cameras

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels.keys())

    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        outputs = [self.channels[channel][index] for channel in channels]
        return self.cameras[index], np.stack(outputs, axis=-1)
