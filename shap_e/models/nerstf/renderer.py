from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from shap_e.models.nerf.model import NeRFModel
from shap_e.models.nerf.ray import RayVolumeIntegral, StratifiedRaySampler, render_rays
from shap_e.models.nn.meta import subdict
from shap_e.models.nn.utils import to_torch
from shap_e.models.query import Query
from shap_e.models.renderer import RayRenderer, render_views_from_rays
from shap_e.models.stf.base import Model
from shap_e.models.stf.renderer import STFRendererBase, render_views_from_stf
from shap_e.models.volume import BoundingBoxVolume, Volume
from shap_e.rendering.blender.constants import BASIC_AMBIENT_COLOR, BASIC_DIFFUSE_COLOR
from shap_e.util.collections import AttrDict


class NeRSTFRenderer(RayRenderer, STFRendererBase):
    def __init__(
        self,
        sdf: Optional[Model],
        tf: Optional[Model],
        nerstf: Optional[Model],
        void: NeRFModel,
        volume: Volume,
        grid_size: int,
        n_coarse_samples: int,
        n_fine_samples: int,
        importance_sampling_options: Optional[Dict[str, Any]] = None,
        separate_shared_samples: bool = False,
        texture_channels: Sequence[str] = ("R", "G", "B"),
        channel_scale: Sequence[float] = (255.0, 255.0, 255.0),
        ambient_color: Union[float, Tuple[float]] = BASIC_AMBIENT_COLOR,
        diffuse_color: Union[float, Tuple[float]] = BASIC_DIFFUSE_COLOR,
        specular_color: Union[float, Tuple[float]] = 0.0,
        output_srgb: bool = True,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(volume, BoundingBoxVolume), "cannot sample points in unknown volume"
        assert (nerstf is not None) ^ (sdf is not None and tf is not None)
        self.sdf = sdf
        self.tf = tf
        self.nerstf = nerstf
        self.void = void
        self.volume = volume
        self.grid_size = grid_size
        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.importance_sampling_options = AttrDict(importance_sampling_options or {})
        self.separate_shared_samples = separate_shared_samples
        self.texture_channels = texture_channels
        self.channel_scale = to_torch(channel_scale).to(device)
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.output_srgb = output_srgb
        self.device = device
        self.to(device)

    def _query(
        self,
        query: Query,
        params: AttrDict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> AttrDict:
        no_dir_query = query.copy()
        no_dir_query.direction = None

        if options.get("rendering_mode", "stf") == "stf":
            assert query.direction is None

        if self.nerstf is not None:
            sdf = tf = self.nerstf(
                query,
                params=subdict(params, "nerstf"),
                options=options,
            )
        else:
            sdf = self.sdf(no_dir_query, params=subdict(params, "sdf"), options=options)
            tf = self.tf(query, params=subdict(params, "tf"), options=options)

        return AttrDict(
            density=sdf.density,
            signed_distance=sdf.signed_distance,
            channels=tf.channels,
            aux_losses=dict(),
        )

    def render_rays(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[AttrDict] = None,
    ) -> AttrDict:
        """
        :param batch: has

            - rays: [batch_size x ... x 2 x 3] specify the origin and direction of each ray.
        :param options: Optional[Dict]
        """
        params = self.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        # Necessary to tell the TF to use specific NeRF channels.
        options.rendering_mode = "nerf"

        model = partial(self._query, params=params, options=options)

        # First, render rays with coarse, stratified samples.
        options.nerf_level = "coarse"
        parts = [
            RayVolumeIntegral(
                model=model,
                volume=self.volume,
                sampler=StratifiedRaySampler(),
                n_samples=self.n_coarse_samples,
            ),
        ]
        coarse_results, samplers, coarse_raw_outputs = render_rays(
            batch.rays,
            parts,
            self.void,
            shared=not self.separate_shared_samples,
            render_with_direction=options.render_with_direction,
            importance_sampling_options=self.importance_sampling_options,
        )

        # Then, render with additional importance-weighted ray samples.
        options.nerf_level = "fine"
        parts = [
            RayVolumeIntegral(
                model=model,
                volume=self.volume,
                sampler=samplers[0],
                n_samples=self.n_fine_samples,
            ),
        ]
        fine_results, _, raw_outputs = render_rays(
            batch.rays,
            parts,
            self.void,
            shared=not self.separate_shared_samples,
            prev_raw_outputs=coarse_raw_outputs,
            render_with_direction=options.render_with_direction,
        )
        raw = raw_outputs[0]

        aux_losses = fine_results.output.aux_losses.copy()
        if self.separate_shared_samples:
            for key, val in coarse_results.output.aux_losses.items():
                aux_losses[key + "_coarse"] = val

        channels = fine_results.output.channels
        shape = [1] * (channels.ndim - 1) + [len(self.texture_channels)]
        channels = channels * self.channel_scale.view(*shape)

        res = AttrDict(
            channels=channels,
            transmittance=fine_results.transmittance,
            raw_signed_distance=raw.signed_distance,
            raw_density=raw.density,
            distances=fine_results.output.distances,
            t0=fine_results.volume_range.t0,
            t1=fine_results.volume_range.t1,
            intersected=fine_results.volume_range.intersected,
            aux_losses=aux_losses,
        )

        if self.separate_shared_samples:
            res.update(
                dict(
                    channels_coarse=(
                        coarse_results.output.channels * self.channel_scale.view(*shape)
                    ),
                    distances_coarse=coarse_results.output.distances,
                    transmittance_coarse=coarse_results.transmittance,
                )
            )

        return res

    def render_views(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[AttrDict] = None,
    ) -> AttrDict:
        """
        Returns a backproppable rendering of a view

        :param batch: contains either ["poses", "camera"], or ["cameras"]. Can
            optionally contain any of ["height", "width", "query_batch_size"]

        :param params: Meta parameters
            contains rendering_mode in ["stf", "nerf"]
        :param options: controls checkpointing, caching, and rendering.
            Can provide a `rendering_mode` in ["stf", "nerf"]
        """
        params = self.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        if options.cache is None:
            created_cache = True
            options.cache = AttrDict()
        else:
            created_cache = False

        rendering_mode = options.get("rendering_mode", "stf")

        if rendering_mode == "nerf":

            output = render_views_from_rays(
                self.render_rays,
                batch,
                params=params,
                options=options,
                device=self.device,
            )

        elif rendering_mode == "stf":

            sdf_fn = tf_fn = nerstf_fn = None
            if self.nerstf is not None:
                nerstf_fn = partial(
                    self.nerstf.forward_batched,
                    params=subdict(params, "nerstf"),
                    options=options,
                )
            else:
                sdf_fn = partial(
                    self.sdf.forward_batched,
                    params=subdict(params, "sdf"),
                    options=options,
                )
                tf_fn = partial(
                    self.tf.forward_batched,
                    params=subdict(params, "tf"),
                    options=options,
                )
            output = render_views_from_stf(
                batch,
                options,
                sdf_fn=sdf_fn,
                tf_fn=tf_fn,
                nerstf_fn=nerstf_fn,
                volume=self.volume,
                grid_size=self.grid_size,
                channel_scale=self.channel_scale,
                texture_channels=self.texture_channels,
                ambient_color=self.ambient_color,
                diffuse_color=self.diffuse_color,
                specular_color=self.specular_color,
                output_srgb=self.output_srgb,
                device=self.device,
            )

        else:

            raise NotImplementedError

        if created_cache:
            del options["cache"]

        return output

    def get_signed_distance(
        self,
        query: Query,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        if self.sdf is not None:
            return self.sdf(query, params=subdict(params, "sdf"), options=options).signed_distance
        assert self.nerstf is not None
        return self.nerstf(query, params=subdict(params, "nerstf"), options=options).signed_distance

    def get_texture(
        self,
        query: Query,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        if self.tf is not None:
            return self.tf(query, params=subdict(params, "tf"), options=options).channels
        assert self.nerstf is not None
        return self.nerstf(query, params=subdict(params, "nerstf"), options=options).channels
