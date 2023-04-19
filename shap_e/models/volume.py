from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from shap_e.models.nn.meta import MetaModule
from shap_e.models.nn.utils import ArrayType, safe_divide, to_torch


@dataclass
class VolumeRange:
    t0: torch.Tensor
    t1: torch.Tensor
    intersected: torch.Tensor

    def __post_init__(self):
        assert self.t0.shape == self.t1.shape == self.intersected.shape

    def next_t0(self):
        """
        Given convex volume1 and volume2, where volume1 is contained in
        volume2, this function returns the t0 at which rays leave volume1 and
        intersect with volume2 \\ volume1.
        """
        return self.t1 * self.intersected.float()

    def extend(self, another: "VolumeRange") -> "VolumeRange":
        """
        The ranges at which rays intersect with either one, or both, or none of
        the self and another are merged together.
        """
        return VolumeRange(
            t0=torch.where(self.intersected, self.t0, another.t0),
            t1=torch.where(another.intersected, another.t1, self.t1),
            intersected=torch.logical_or(self.intersected, another.intersected),
        )

    def partition(self, ts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Partitions t0 and t1 into n_samples intervals.

        :param ts: [batch_size, *shape, n_samples, 1]
        :return: a tuple of (
            lower: [batch_size, *shape, n_samples, 1]
            upper: [batch_size, *shape, n_samples, 1]
            delta: [batch_size, *shape, n_samples, 1]
        ) where

            ts \\in [lower, upper]
            deltas = upper - lower
        """
        mids = (ts[..., 1:, :] + ts[..., :-1, :]) * 0.5
        lower = torch.cat([self.t0[..., None, :], mids], dim=-2)
        upper = torch.cat([mids, self.t1[..., None, :]], dim=-2)
        delta = upper - lower
        assert lower.shape == upper.shape == delta.shape == ts.shape
        return lower, upper, delta


class Volume(ABC):
    """
    An abstraction of rendering volume.
    """

    @abstractmethod
    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
        epsilon: float = 1e-6,
    ) -> VolumeRange:
        """
        :param origin: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
        :param params: Optional meta parameters in case Volume is parametric
        :param epsilon: to stabilize calculations

        :return: A tuple of (t0, t1, intersected) where each has a shape
            [batch_size, *shape, 1]. If a ray intersects with the volume, `o + td` is
            in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed
            to be on the boundary of the volume.
        """


class BoundingBoxVolume(MetaModule, Volume):
    """
    Axis-aligned bounding box defined by the two opposite corners.
    """

    def __init__(
        self,
        *,
        bbox_min: ArrayType,
        bbox_max: ArrayType,
        min_dist: float = 0.0,
        min_t_range: float = 1e-3,
        device: torch.device = torch.device("cuda"),
    ):
        """
        :param bbox_min: the left/bottommost corner of the bounding box
        :param bbox_max: the other corner of the bounding box
        :param min_dist: all rays should start at least this distance away from the origin.
        """
        super().__init__()

        self.bbox_min = to_torch(bbox_min).to(device)
        self.bbox_max = to_torch(bbox_max).to(device)
        self.min_dist = min_dist
        self.min_t_range = min_t_range
        self.bbox = torch.stack([self.bbox_min, self.bbox_max])
        assert self.bbox.shape == (2, 3)
        assert self.min_dist >= 0.0
        assert self.min_t_range > 0.0
        self.device = device

    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
        epsilon=1e-6,
    ) -> VolumeRange:
        """
        :param origin: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
        :param params: Optional meta parameters in case Volume is parametric
        :param epsilon: to stabilize calculations

        :return: A tuple of (t0, t1, intersected) where each has a shape
            [batch_size, *shape, 1]. If a ray intersects with the volume, `o + td` is
            in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed
            to be on the boundary of the volume.
        """

        batch_size, *shape, _ = origin.shape
        ones = [1] * len(shape)
        bbox = self.bbox.view(1, *ones, 2, 3)
        ts = safe_divide(bbox - origin[..., None, :], direction[..., None, :], epsilon=epsilon)

        # Cases to think about:
        #
        #   1. t1 <= t0: the ray does not pass through the AABB.
        #   2. t0 < t1 <= 0: the ray intersects but the BB is behind the origin.
        #   3. t0 <= 0 <= t1: the ray starts from inside the BB
        #   4. 0 <= t0 < t1: the ray is not inside and intersects with the BB twice.
        #
        # 1 and 4 are clearly handled from t0 < t1 below.
        # Making t0 at least min_dist (>= 0) takes care of 2 and 3.
        t0 = ts.min(dim=-2).values.max(dim=-1, keepdim=True).values.clamp(self.min_dist)
        t1 = ts.max(dim=-2).values.min(dim=-1, keepdim=True).values
        assert t0.shape == t1.shape == (batch_size, *shape, 1)
        if t0_lower is not None:
            assert t0.shape == t0_lower.shape
            t0 = torch.maximum(t0, t0_lower)

        intersected = t0 + self.min_t_range < t1
        t0 = torch.where(intersected, t0, torch.zeros_like(t0))
        t1 = torch.where(intersected, t1, torch.ones_like(t1))

        return VolumeRange(t0=t0, t1=t1, intersected=intersected)


class UnboundedVolume(MetaModule, Volume):
    """
    Originally used in NeRF. Unbounded volume but with a limited visibility
    when rendering (e.g. objects that are farther away than the max_dist from
    the ray origin are not considered)
    """

    def __init__(
        self,
        *,
        max_dist: float,
        min_dist: float = 0.0,
        min_t_range: float = 1e-3,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.min_t_range = min_t_range
        assert self.min_dist >= 0.0
        assert self.min_t_range > 0.0
        self.device = device

    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
    ) -> VolumeRange:
        """
        :param origin: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
        :param params: Optional meta parameters in case Volume is parametric
        :param epsilon: to stabilize calculations

        :return: A tuple of (t0, t1, intersected) where each has a shape
            [batch_size, *shape, 1]. If a ray intersects with the volume, `o + td` is
            in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed
            to be on the boundary of the volume.
        """

        batch_size, *shape, _ = origin.shape
        t0 = torch.zeros(batch_size, *shape, 1, dtype=origin.dtype, device=origin.device)
        if t0_lower is not None:
            t0 = torch.maximum(t0, t0_lower)
        t1 = t0 + self.max_dist
        t0 = t0.clamp(self.min_dist)
        return VolumeRange(t0=t0, t1=t1, intersected=t0 + self.min_t_range < t1)


class SphericalVolume(MetaModule, Volume):
    """
    Used in NeRF++ but will not be used probably unless we want to reproduce
    their results.
    """

    def __init__(
        self,
        *,
        radius: float,
        center: ArrayType = (0.0, 0.0, 0.0),
        min_dist: float = 0.0,
        min_t_range: float = 1e-3,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.radius = radius
        self.center = to_torch(center).to(device)
        self.min_dist = min_dist
        self.min_t_range = min_t_range
        assert self.min_dist >= 0.0
        assert self.min_t_range > 0.0
        self.device = device

    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
        epsilon=1e-6,
    ) -> VolumeRange:
        raise NotImplementedError
