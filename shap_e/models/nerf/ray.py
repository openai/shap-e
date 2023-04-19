from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from shap_e.models.nn.utils import sample_pmf
from shap_e.models.volume import Volume, VolumeRange
from shap_e.util.collections import AttrDict

from .model import NeRFModel, Query


def render_rays(
    rays: torch.Tensor,
    parts: List["RayVolumeIntegral"],
    void_model: NeRFModel,
    shared: bool = False,
    prev_raw_outputs: Optional[List[AttrDict]] = None,
    render_with_direction: bool = True,
    importance_sampling_options: Optional[Dict[str, Any]] = None,
) -> Tuple["RayVolumeIntegralResults", List["RaySampler"], List[AttrDict]]:
    """
    Perform volumetric rendering over a partition of possible t's in the union
    of rendering volumes (written below with some abuse of notations)

        C(r) := sum(
            transmittance(t[i]) *
            integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t[i], t[i + 1]],
            )
            for i in range(len(parts))
        ) + transmittance(t[-1]) * void_model(t[-1]).channels

    where

    1) transmittance(s) := exp(-integrate(density, [t[0], s])) calculates the
       probability of light passing through the volume specified by [t[0], s].
       (transmittance of 1 means light can pass freely)
    2) density and channels are obtained by evaluating the appropriate
       part.model at time t.
    3) [t[i], t[i + 1]] is defined as the range of t where the ray intersects
       (parts[i].volume \\ union(part.volume for part in parts[:i])) at the surface
       of the shell (if bounded). If the ray does not intersect, the integral over
       this segment is evaluated as 0 and transmittance(t[i + 1]) :=
       transmittance(t[i]).
    4) The last term is integration to infinity (e.g. [t[-1], math.inf]) that
       is evaluated by the void_model (i.e. we consider this space to be empty).

    :param rays: [batch_size x ... x 2 x 3] origin and direction.
    :param parts: disjoint volume integrals.
    :param void_model: use this model to integrate over the empty space
    :param shared: All RayVolumeIntegrals are calculated with the same model.
    :param prev_raw_outputs: Raw outputs from the previous rendering step

    :return: A tuple of
        - AttrDict containing the rendered `channels`, `distances`, and the `aux_losses`
        - A list of importance samplers for additional fine-grained rendering
        - A list of raw output for each interval
    """
    if importance_sampling_options is None:
        importance_sampling_options = {}

    origin, direc = rays[..., 0, :], rays[..., 1, :]

    if prev_raw_outputs is None:
        prev_raw_outputs = [None] * len(parts)

    samplers = []
    raw_outputs = []
    t0 = None
    results = None

    for part_i, prev_raw_i in zip(parts, prev_raw_outputs):

        # Integrate over [t[i], t[i + 1]]
        results_i = part_i.render_rays(
            origin,
            direc,
            t0=t0,
            prev_raw=prev_raw_i,
            shared=shared,
            render_with_direction=render_with_direction,
        )

        # Create an importance sampler for (optional) fine rendering
        samplers.append(
            ImportanceRaySampler(
                results_i.volume_range, results_i.raw, **importance_sampling_options
            )
        )
        raw_outputs.append(results_i.raw)

        # Pass t[i + 1] as the start of integration for the next interval.
        t0 = results_i.volume_range.next_t0()

        # Combine the results from [t[0], t[i]] and [t[i], t[i+1]]
        results = results_i if results is None else results.combine(results_i)

    # While integrating out [t[-1], math.inf] is the correct thing to do, this
    # erases a lot of useful information. Also, void_model is meant to predict
    # the channels at t=math.inf.

    # # Add the void background over [t[-1], math.inf] to complete integration.
    # results = results.combine(
    #     RayVolumeIntegralResults(
    #         output=AttrDict(
    #             channels=void_model(origin, direc),
    #             distances=torch.zeros_like(t0),
    #             aux_losses=AttrDict(),
    #         ),
    #         volume_range=VolumeRange(
    #             t0=t0,
    #             t1=torch.full_like(t0, math.inf),
    #             intersected=torch.full_like(results.volume_range.intersected, True),
    #         ),
    #         # Void space extends to infinity. It is assumed that no light
    #         # passes beyond the void.
    #         transmittance=torch.zeros_like(results_i.transmittance),
    #     )
    # )

    results.output.channels = results.output.channels + results.transmittance * void_model(
        Query(origin, direc)
    )

    return results, samplers, raw_outputs


@dataclass
class RayVolumeIntegralResults:
    """
    Stores the relevant state and results of

        integrate(
            lambda t: density(t) * channels(t) * transmittance(t),
            [t0, t1],
        )
    """

    # Rendered output and auxiliary losses
    # output.channels has shape [batch_size, *inner_shape, n_channels]
    output: AttrDict

    """
    Optional values
    """

    # Raw values contain the sampled `ts`, `density`, `channels`, etc.
    raw: Optional[AttrDict] = None

    # Integration
    volume_range: Optional[VolumeRange] = None

    # If a ray intersects, the transmittance from t0 to t1 (e.g. the
    # probability that the ray passes through this volume).
    # has shape [batch_size, *inner_shape, 1]
    transmittance: Optional[torch.Tensor] = None

    def combine(self, cur: "RayVolumeIntegralResults") -> "RayVolumeIntegralResults":
        """
        Combines the integration results of `self` over [t0, t1] and
        `cur` over [t1, t2] to produce a new set of results over [t0, t2] by
        using a similar equation to (4) in NeRF++:

            integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t0, t2]
            )

          = integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t0, t1]
            ) + transmittance(t1) * integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t1, t2]
            )
        """
        assert torch.allclose(self.volume_range.next_t0(), cur.volume_range.t0)

        def _combine_fn(
            prev_val: Optional[torch.Tensor],
            cur_val: Optional[torch.Tensor],
            *,
            prev_transmittance: torch.Tensor,
        ):
            assert prev_val is not None
            if cur_val is None:
                # cur_output.aux_losses are empty for the void_model.
                return prev_val
            return prev_val + prev_transmittance * cur_val

        output = self.output.combine(
            cur.output, combine_fn=partial(_combine_fn, prev_transmittance=self.transmittance)
        )

        combined = RayVolumeIntegralResults(
            output=output,
            volume_range=self.volume_range.extend(cur.volume_range),
            transmittance=self.transmittance * cur.transmittance,
        )
        return combined


@dataclass
class RayVolumeIntegral:
    model: NeRFModel
    volume: Volume
    sampler: "RaySampler"
    n_samples: int

    def render_rays(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0: Optional[torch.Tensor] = None,
        prev_raw: Optional[AttrDict] = None,
        shared: bool = False,
        render_with_direction: bool = True,
    ) -> "RayVolumeIntegralResults":
        """
        Perform volumetric rendering over the given volume.

        :param position: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0: Optional [batch_size, *shape, 1]
        :param prev_raw: the raw outputs when using multiple levels with this model.
        :param shared: means the same model is used for all RayVolumeIntegral's
        :param render_with_direction: use the incoming ray direction when querying the model.

        :return: RayVolumeIntegralResults
        """
        # 1. Intersect the rays with the current volume and sample ts to
        # integrate along.
        vrange = self.volume.intersect(origin, direction, t0_lower=t0)
        ts = self.sampler.sample(vrange.t0, vrange.t1, self.n_samples)

        if prev_raw is not None and not shared:
            # Append the previous ts now before fprop because previous
            # rendering used a different model and we can't reuse the output.
            ts = torch.sort(torch.cat([ts, prev_raw.ts], dim=-2), dim=-2).values

        # Shape sanity checks
        batch_size, *_shape, _t0_dim = vrange.t0.shape
        _, *ts_shape, _ts_dim = ts.shape

        # 2. Get the points along the ray and query the model
        directions = torch.broadcast_to(direction.unsqueeze(-2), [batch_size, *ts_shape, 3])
        positions = origin.unsqueeze(-2) + ts * directions

        optional_directions = directions if render_with_direction else None
        mids = (ts[..., 1:, :] + ts[..., :-1, :]) / 2
        raw = self.model(
            Query(
                position=positions,
                direction=optional_directions,
                t_min=torch.cat([vrange.t0[..., None, :], mids], dim=-2),
                t_max=torch.cat([mids, vrange.t1[..., None, :]], dim=-2),
            )
        )
        raw.ts = ts

        if prev_raw is not None and shared:
            # We can append the additional queries to previous raw outputs
            # before integration
            copy = prev_raw.copy()
            result = torch.sort(torch.cat([raw.pop("ts"), copy.pop("ts")], dim=-2), dim=-2)
            merge_results = partial(self._merge_results, dim=-2, indices=result.indices)
            raw = raw.combine(copy, merge_results)
            raw.ts = result.values

        # 3. Integrate the raw results
        output, transmittance = self.integrate_samples(vrange, raw)

        # 4. Clean up results that do not intersect with the volume.
        transmittance = torch.where(
            vrange.intersected, transmittance, torch.ones_like(transmittance)
        )

        def _mask_fn(_key: str, tensor: torch.Tensor):
            return torch.where(vrange.intersected, tensor, torch.zeros_like(tensor))

        def _is_tensor(_key: str, value: Any):
            return isinstance(value, torch.Tensor)

        output = output.map(map_fn=_mask_fn, should_map=_is_tensor)

        return RayVolumeIntegralResults(
            output=output,
            raw=raw,
            volume_range=vrange,
            transmittance=transmittance,
        )

    def integrate_samples(
        self,
        volume_range: VolumeRange,
        raw: AttrDict,
    ) -> Tuple[AttrDict, torch.Tensor]:
        """
        Integrate the raw.channels along with other aux_losses and values to
        produce the final output dictionary containing rendered `channels`,
        estimated `distances` and `aux_losses`.

        :param volume_range: Specifies the integral range [t0, t1]
        :param raw: Contains a dict of function evaluations at ts. Should have

            density: torch.Tensor [batch_size, *shape, n_samples, 1]
            channels: torch.Tensor [batch_size, *shape, n_samples, n_channels]
            aux_losses: {key: torch.Tensor [batch_size, *shape, n_samples, 1] for each key}
            no_weight_grad_aux_losses: an optional set of losses for which the weights
                                       should be detached before integration.

            after the call, integrate_samples populates some intermediate calculations
            for later use like

            weights: torch.Tensor [batch_size, *shape, n_samples, 1] (density *
                transmittance)[i] weight for each rgb output at [..., i, :].
        :returns: a tuple of (
            a dictionary of rendered outputs and aux_losses,
            transmittance of this volume,
        )
        """

        # 1. Calculate the weights
        _, _, dt = volume_range.partition(raw.ts)
        ddensity = raw.density * dt

        mass = torch.cumsum(ddensity, dim=-2)
        transmittance = torch.exp(-mass[..., -1, :])

        alphas = 1.0 - torch.exp(-ddensity)
        Ts = torch.exp(torch.cat([torch.zeros_like(mass[..., :1, :]), -mass[..., :-1, :]], dim=-2))
        # This is the probability of light hitting and reflecting off of
        # something at depth [..., i, :].
        weights = alphas * Ts

        # 2. Integrate all results
        def _integrate(key: str, samples: torch.Tensor, weights: torch.Tensor):
            if key == "density":
                # Omit integrating the density, because we don't need it
                return None
            return torch.sum(samples * weights, dim=-2)

        def _is_tensor(_key: str, value: Any):
            return isinstance(value, torch.Tensor)

        if raw.no_weight_grad_aux_losses:
            extra_aux_losses = raw.no_weight_grad_aux_losses.map(
                partial(_integrate, weights=weights.detach()), should_map=_is_tensor
            )
        else:
            extra_aux_losses = {}
        output = raw.map(partial(_integrate, weights=weights), should_map=_is_tensor)
        if "no_weight_grad_aux_losses" in output:
            del output["no_weight_grad_aux_losses"]
        output.aux_losses.update(extra_aux_losses)

        # Integrating the ts yields the distance away from the origin; rename the variable.
        output.distances = output.ts
        del output["ts"]
        del output["density"]

        assert output.distances.shape == (*output.channels.shape[:-1], 1)
        assert output.channels.shape[:-1] == raw.channels.shape[:-2]
        assert output.channels.shape[-1] == raw.channels.shape[-1]

        # 3. Reduce loss
        def _reduce_loss(_key: str, loss: torch.Tensor):
            return loss.view(loss.shape[0], -1).sum(dim=-1)

        # 4. Store other useful calculations
        raw.weights = weights

        output.aux_losses = output.aux_losses.map(_reduce_loss)

        return output, transmittance

    def _merge_results(
        self, a: Optional[torch.Tensor], b: torch.Tensor, dim: int, indices: torch.Tensor
    ):
        """
        :param a: [..., n_a, ...]. The other dictionary containing the b's may
            contain extra tensors from earlier calculations, so a can be None.
        :param b: [..., n_b, ...]
        :param dim: dimension to merge
        :param indices: how the merged results should be sorted at the end
        :return: a concatted and sorted tensor of size [..., n_a + n_b, ...]
        """
        if a is None:
            return None

        merged = torch.cat([a, b], dim=dim)
        return torch.gather(merged, dim=dim, index=torch.broadcast_to(indices, merged.shape))


class RaySampler(ABC):
    @abstractmethod
    def sample(self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        :param t0: start time has shape [batch_size, *shape, 1]
        :param t1: finish time has shape [batch_size, *shape, 1]
        :param n_samples: number of ts to sample
        :return: sampled ts of shape [batch_size, *shape, n_samples, 1]
        """


class StratifiedRaySampler(RaySampler):
    """
    Instead of fixed intervals, a sample is drawn uniformly at random from each
    interval.
    """

    def __init__(self, depth_mode: str = "linear"):
        """
        :param depth_mode: linear samples ts linearly in depth. harmonic ensures
            closer points are sampled more densely.
        """
        self.depth_mode = depth_mode
        assert self.depth_mode in ("linear", "geometric", "harmonic")

    def sample(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        n_samples: int,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        """
        :param t0: start time has shape [batch_size, *shape, 1]
        :param t1: finish time has shape [batch_size, *shape, 1]
        :param n_samples: number of ts to sample
        :return: sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        ones = [1] * (len(t0.shape) - 1)
        ts = torch.linspace(0, 1, n_samples).view(*ones, n_samples).to(t0.dtype).to(t0.device)

        if self.depth_mode == "linear":
            ts = t0 * (1.0 - ts) + t1 * ts
        elif self.depth_mode == "geometric":
            ts = (t0.clamp(epsilon).log() * (1.0 - ts) + t1.clamp(epsilon).log() * ts).exp()
        elif self.depth_mode == "harmonic":
            # The original NeRF recommends this interpolation scheme for
            # spherical scenes, but there could be some weird edge cases when
            # the observer crosses from the inner to outer volume.
            ts = 1.0 / (1.0 / t0.clamp(epsilon) * (1.0 - ts) + 1.0 / t1.clamp(epsilon) * ts)

        mids = 0.5 * (ts[..., 1:] + ts[..., :-1])
        upper = torch.cat([mids, t1], dim=-1)
        lower = torch.cat([t0, mids], dim=-1)
        t_rand = torch.rand_like(ts)

        ts = lower + (upper - lower) * t_rand
        return ts.unsqueeze(-1)


class ImportanceRaySampler(RaySampler):
    """
    Given the initial estimate of densities, this samples more from
    regions/bins expected to have objects.
    """

    def __init__(
        self, volume_range: VolumeRange, raw: AttrDict, blur_pool: bool = False, alpha: float = 1e-5
    ):
        """
        :param volume_range: the range in which a ray intersects the given volume.
        :param raw: dictionary of raw outputs from the NeRF models of shape
            [batch_size, *shape, n_coarse_samples, 1]. Should at least contain

            :param ts: earlier samples from the coarse rendering step
            :param weights: discretized version of density * transmittance
        :param blur_pool: if true, use 2-tap max + 2-tap blur filter from mip-NeRF.
        :param alpha: small value to add to weights.
        """
        self.volume_range = volume_range
        self.ts = raw.ts.clone().detach()
        self.weights = raw.weights.clone().detach()
        self.blur_pool = blur_pool
        self.alpha = alpha

    @torch.no_grad()
    def sample(self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        :param t0: start time has shape [batch_size, *shape, 1]
        :param t1: finish time has shape [batch_size, *shape, 1]
        :param n_samples: number of ts to sample
        :return: sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        lower, upper, _ = self.volume_range.partition(self.ts)

        batch_size, *shape, n_coarse_samples, _ = self.ts.shape

        weights = self.weights
        if self.blur_pool:
            padded = torch.cat([weights[..., :1, :], weights, weights[..., -1:, :]], dim=-2)
            maxes = torch.maximum(padded[..., :-1, :], padded[..., 1:, :])
            weights = 0.5 * (maxes[..., :-1, :] + maxes[..., 1:, :])
        weights = weights + self.alpha
        pmf = weights / weights.sum(dim=-2, keepdim=True)
        inds = sample_pmf(pmf, n_samples)
        assert inds.shape == (batch_size, *shape, n_samples, 1)
        assert (inds >= 0).all() and (inds < n_coarse_samples).all()

        t_rand = torch.rand(inds.shape, device=inds.device)
        lower_ = torch.gather(lower, -2, inds)
        upper_ = torch.gather(upper, -2, inds)

        ts = lower_ + (upper_ - lower_) * t_rand
        ts = torch.sort(ts, dim=-2).values
        return ts
