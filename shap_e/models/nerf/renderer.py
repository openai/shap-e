from functools import partial
from typing import Any, Dict, Optional

import torch

from shap_e.models.nn.meta import subdict
from shap_e.models.renderer import RayRenderer
from shap_e.models.volume import Volume
from shap_e.util.collections import AttrDict

from .model import NeRFModel
from .ray import RayVolumeIntegral, StratifiedRaySampler, render_rays


class TwoStepNeRFRenderer(RayRenderer):
    """
    Coarse and fine-grained rendering as proposed by NeRF. This class
    additionally supports background rendering like NeRF++.
    """

    def __init__(
        self,
        n_coarse_samples: int,
        n_fine_samples: int,
        void_model: NeRFModel,
        fine_model: NeRFModel,
        volume: Volume,
        coarse_model: Optional[NeRFModel] = None,
        coarse_background_model: Optional[NeRFModel] = None,
        fine_background_model: Optional[NeRFModel] = None,
        outer_volume: Optional[Volume] = None,
        foreground_stratified_depth_sampling_mode: str = "linear",
        background_stratified_depth_sampling_mode: str = "linear",
        importance_sampling_options: Optional[Dict[str, Any]] = None,
        channel_scale: float = 255,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        """
        :param outer_volume: is where distant objects are encoded.
        """
        super().__init__(**kwargs)

        if coarse_model is None:
            assert (
                fine_background_model is None or coarse_background_model is None
            ), "models should be shared for both fg and bg"

        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.void_model = void_model
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.volume = volume
        self.coarse_background_model = coarse_background_model
        self.fine_background_model = fine_background_model
        self.outer_volume = outer_volume
        self.foreground_stratified_depth_sampling_mode = foreground_stratified_depth_sampling_mode
        self.background_stratified_depth_sampling_mode = background_stratified_depth_sampling_mode
        self.importance_sampling_options = AttrDict(importance_sampling_options or {})
        self.channel_scale = channel_scale
        self.device = device
        self.to(device)

        if self.coarse_background_model is not None:
            assert self.fine_background_model is not None
            assert self.outer_volume is not None

    def render_rays(
        self,
        batch: Dict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        params = self.update(params)

        batch = AttrDict(batch)
        if options is None:
            options = AttrDict()
        options.setdefault("render_background", True)
        options.setdefault("render_with_direction", True)
        options.setdefault("n_coarse_samples", self.n_coarse_samples)
        options.setdefault("n_fine_samples", self.n_fine_samples)
        options.setdefault(
            "foreground_stratified_depth_sampling_mode",
            self.foreground_stratified_depth_sampling_mode,
        )
        options.setdefault(
            "background_stratified_depth_sampling_mode",
            self.background_stratified_depth_sampling_mode,
        )

        shared = self.coarse_model is None

        # First, render rays using the coarse models with stratified ray samples.
        coarse_model, coarse_key = (
            (self.fine_model, "fine_model") if shared else (self.coarse_model, "coarse_model")
        )
        coarse_model = partial(
            coarse_model,
            params=subdict(params, coarse_key),
            options=options,
        )
        parts = [
            RayVolumeIntegral(
                model=coarse_model,
                volume=self.volume,
                sampler=StratifiedRaySampler(
                    depth_mode=options.foreground_stratified_depth_sampling_mode,
                ),
                n_samples=options.n_coarse_samples,
            ),
        ]
        if options.render_background and self.outer_volume is not None:
            coarse_background_model, coarse_background_key = (
                (self.fine_background_model, "fine_background_model")
                if shared
                else (self.coarse_background_model, "coarse_background_model")
            )
            coarse_background_model = partial(
                coarse_background_model,
                params=subdict(params, coarse_background_key),
                options=options,
            )
            parts.append(
                RayVolumeIntegral(
                    model=coarse_background_model,
                    volume=self.outer_volume,
                    sampler=StratifiedRaySampler(
                        depth_mode=options.background_stratified_depth_sampling_mode,
                    ),
                    n_samples=options.n_coarse_samples,
                )
            )
        coarse_results, samplers, coarse_raw_outputs = render_rays(
            batch.rays,
            parts,
            partial(self.void_model, options=options),
            shared=shared,
            render_with_direction=options.render_with_direction,
            importance_sampling_options=AttrDict(self.importance_sampling_options),
        )

        # Then, render rays using the fine models with importance-weighted ray samples.
        fine_model = partial(
            self.fine_model,
            params=subdict(params, "fine_model"),
            options=options,
        )
        parts = [
            RayVolumeIntegral(
                model=fine_model,
                volume=self.volume,
                sampler=samplers[0],
                n_samples=options.n_fine_samples,
            ),
        ]
        if options.render_background and self.outer_volume is not None:
            fine_background_model = partial(
                self.fine_background_model,
                params=subdict(params, "fine_background_model"),
                options=options,
            )
            parts.append(
                RayVolumeIntegral(
                    model=fine_background_model,
                    volume=self.outer_volume,
                    sampler=samplers[1],
                    n_samples=options.n_fine_samples,
                )
            )
        fine_results, *_ = render_rays(
            batch.rays,
            parts,
            partial(self.void_model, options=options),
            shared=shared,
            prev_raw_outputs=coarse_raw_outputs,
            render_with_direction=options.render_with_direction,
        )

        # Combine results
        aux_losses = fine_results.output.aux_losses.copy()
        for key, val in coarse_results.output.aux_losses.items():
            aux_losses[key + "_coarse"] = val

        return AttrDict(
            channels=fine_results.output.channels * self.channel_scale,
            channels_coarse=coarse_results.output.channels * self.channel_scale,
            distances=fine_results.output.distances,
            transmittance=fine_results.transmittance,
            transmittance_coarse=coarse_results.transmittance,
            t0=fine_results.volume_range.t0,
            t1=fine_results.volume_range.t1,
            intersected=fine_results.volume_range.intersected,
            aux_losses=aux_losses,
        )


class OneStepNeRFRenderer(RayRenderer):
    """
    Renders rays using stratified sampling only unlike vanilla NeRF.
    The same setup as NeRF++.
    """

    def __init__(
        self,
        n_samples: int,
        void_model: NeRFModel,
        foreground_model: NeRFModel,
        volume: Volume,
        background_model: Optional[NeRFModel] = None,
        outer_volume: Optional[Volume] = None,
        foreground_stratified_depth_sampling_mode: str = "linear",
        background_stratified_depth_sampling_mode: str = "linear",
        channel_scale: float = 255,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.void_model = void_model
        self.foreground_model = foreground_model
        self.volume = volume
        self.background_model = background_model
        self.outer_volume = outer_volume
        self.foreground_stratified_depth_sampling_mode = foreground_stratified_depth_sampling_mode
        self.background_stratified_depth_sampling_mode = background_stratified_depth_sampling_mode
        self.channel_scale = channel_scale
        self.device = device
        self.to(device)

    def render_rays(
        self,
        batch: Dict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        params = self.update(params)

        batch = AttrDict(batch)
        if options is None:
            options = AttrDict()
        options.setdefault("render_background", True)
        options.setdefault("render_with_direction", True)
        options.setdefault("n_samples", self.n_samples)
        options.setdefault(
            "foreground_stratified_depth_sampling_mode",
            self.foreground_stratified_depth_sampling_mode,
        )
        options.setdefault(
            "background_stratified_depth_sampling_mode",
            self.background_stratified_depth_sampling_mode,
        )

        foreground_model = partial(
            self.foreground_model,
            params=subdict(params, "foreground_model"),
            options=options,
        )
        parts = [
            RayVolumeIntegral(
                model=foreground_model,
                volume=self.volume,
                sampler=StratifiedRaySampler(
                    depth_mode=options.foreground_stratified_depth_sampling_mode
                ),
                n_samples=options.n_samples,
            ),
        ]
        if options.render_background and self.outer_volume is not None:
            background_model = partial(
                self.background_model,
                params=subdict(params, "background_model"),
                options=options,
            )
            parts.append(
                RayVolumeIntegral(
                    model=background_model,
                    volume=self.outer_volume,
                    sampler=StratifiedRaySampler(
                        depth_mode=options.background_stratified_depth_sampling_mode
                    ),
                    n_samples=options.n_samples,
                )
            )
        results, *_ = render_rays(
            batch.rays,
            parts,
            self.void_model,
            render_with_direction=options.render_with_direction,
        )

        return AttrDict(
            channels=results.output.channels * self.channel_scale,
            distances=results.output.distances,
            transmittance=results.transmittance,
            t0=results.volume_range.t0,
            t1=results.volume_range.t1,
            intersected=results.volume_range.intersected,
            aux_losses=results.output.aux_losses,
        )
