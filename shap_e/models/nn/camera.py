from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from shap_e.rendering.view_data import ProjectiveCamera


@dataclass
class DifferentiableCamera(ABC):
    """
    An object describing how a camera corresponds to pixels in an image.
    """

    @abstractmethod
    def camera_rays(self, coords: torch.Tensor) -> torch.Tensor:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x ... x 2] integer array of 2D image coordinates.
        :return: an [N x ... x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """

    @abstractmethod
    def resize_image(self, width: int, height: int) -> "DifferentiableCamera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with resized image dimensions.
        """


@dataclass
class DifferentiableProjectiveCamera(DifferentiableCamera):
    """
    Implements a batch, differentiable, standard pinhole camera
    """

    origin: torch.Tensor  # [batch_size x 3]
    x: torch.Tensor  # [batch_size x 3]
    y: torch.Tensor  # [batch_size x 3]
    z: torch.Tensor  # [batch_size x 3]
    width: int
    height: int
    x_fov: float
    y_fov: float

    def __post_init__(self):
        assert self.x.shape[0] == self.y.shape[0] == self.z.shape[0] == self.origin.shape[0]
        assert self.x.shape[1] == self.y.shape[1] == self.z.shape[1] == self.origin.shape[1] == 3
        assert (
            len(self.x.shape)
            == len(self.y.shape)
            == len(self.z.shape)
            == len(self.origin.shape)
            == 2
        )

    def resolution(self):
        return torch.from_numpy(np.array([self.width, self.height], dtype=np.float32))

    def fov(self):
        return torch.from_numpy(np.array([self.x_fov, self.y_fov], dtype=np.float32))

    def image_coords(self) -> torch.Tensor:
        """
        :return: coords of shape (width * height, 2)
        """
        pixel_indices = torch.arange(self.height * self.width)
        coords = torch.stack(
            [
                pixel_indices % self.width,
                torch.div(pixel_indices, self.width, rounding_mode="trunc"),
            ],
            axis=1,
        )
        return coords

    def camera_rays(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size, *shape, n_coords = coords.shape
        assert n_coords == 2
        assert batch_size == self.origin.shape[0]
        flat = coords.view(batch_size, -1, 2)

        res = self.resolution().to(flat.device)
        fov = self.fov().to(flat.device)

        fracs = (flat.float() / (res - 1)) * 2 - 1
        fracs = fracs * torch.tan(fov / 2)

        fracs = fracs.view(batch_size, -1, 2)
        directions = (
            self.z.view(batch_size, 1, 3)
            + self.x.view(batch_size, 1, 3) * fracs[:, :, :1]
            + self.y.view(batch_size, 1, 3) * fracs[:, :, 1:]
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)
        rays = torch.stack(
            [
                torch.broadcast_to(
                    self.origin.view(batch_size, 1, 3), [batch_size, directions.shape[1], 3]
                ),
                directions,
            ],
            dim=2,
        )
        return rays.view(batch_size, *shape, 2, 3)

    def resize_image(self, width: int, height: int) -> "DifferentiableProjectiveCamera":
        """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
        assert width * self.height == height * self.width, "The aspect ratio should not change."
        return DifferentiableProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=width,
            height=height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )


@dataclass
class DifferentiableCameraBatch(ABC):
    """
    Annotate a differentiable camera with a multi-dimensional batch shape.
    """

    shape: Tuple[int]
    flat_camera: DifferentiableCamera


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True)


def project_out(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Removes the vec2 component from vec1
    """
    vec2 = normalize(vec2)
    proj = (vec1 * vec2).sum(dim=-1, keepdim=True)
    return vec1 - proj * vec2


def camera_orientation(toward: torch.Tensor, up: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    :param toward: [batch_size x 3] unit vector from camera position to the object
    :param up: Optional [batch_size x 3] specifying the physical up direction in the world frame.
    :return: [batch_size x 3 x 3]
    """

    if up is None:
        up = torch.zeros_like(toward)
        up[:, 2] = 1

    assert len(toward.shape) == 2
    assert toward.shape[1] == 3

    assert len(up.shape) == 2
    assert up.shape[1] == 3

    z = toward / toward.norm(dim=-1, keepdim=True)
    y = -normalize(project_out(up, toward))
    x = torch.cross(y, z, dim=1)
    return torch.stack([x, y, z], dim=1)


def projective_camera_frame(
    origin: torch.Tensor,
    toward: torch.Tensor,
    camera_params: Union[ProjectiveCamera, DifferentiableProjectiveCamera],
) -> DifferentiableProjectiveCamera:
    """
    Given the origin and the direction of a view, return a differentiable
    projective camera with the given parameters.

    TODO: We need to support the rotation of the camera frame about the
    `toward` vector to fully implement 6 degrees of freedom.
    """
    rot = camera_orientation(toward)
    camera = DifferentiableProjectiveCamera(
        origin=origin,
        x=rot[:, 0],
        y=rot[:, 1],
        z=rot[:, 2],
        width=camera_params.width,
        height=camera_params.height,
        x_fov=camera_params.x_fov,
        y_fov=camera_params.y_fov,
    )
    return camera


@torch.no_grad()
def get_image_coords(width, height) -> torch.Tensor:
    pixel_indices = torch.arange(height * width)
    # torch throws warnings for pixel_indices // width
    pixel_indices_div = torch.div(pixel_indices, width, rounding_mode="trunc")
    coords = torch.stack([pixel_indices % width, pixel_indices_div], dim=1)
    return coords
