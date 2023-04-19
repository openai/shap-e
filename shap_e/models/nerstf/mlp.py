from typing import Any, Dict, Optional, Tuple

import torch

from shap_e.models.nn.ops import get_act
from shap_e.models.query import Query
from shap_e.models.stf.mlp import MLPModel
from shap_e.util.collections import AttrDict


class MLPDensitySDFModel(MLPModel):
    def __init__(
        self,
        initial_bias: float = -0.1,
        sdf_activation="tanh",
        density_activation="exp",
        **kwargs,
    ):
        super().__init__(
            n_output=2,
            output_activation="identity",
            **kwargs,
        )
        self.mlp[-1].bias[0].data.fill_(initial_bias)
        self.sdf_activation = get_act(sdf_activation)
        self.density_activation = get_act(density_activation)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        # query.direction is None typically for SDF models and training
        h, _h_directionless = self._mlp(
            query.position, query.direction, params=params, options=options
        )
        h_sdf, h_density = h.split(1, dim=-1)
        return AttrDict(
            density=self.density_activation(h_density),
            signed_distance=self.sdf_activation(h_sdf),
        )


class MLPNeRSTFModel(MLPModel):
    def __init__(
        self,
        sdf_activation="tanh",
        density_activation="exp",
        channel_activation="sigmoid",
        direction_dependent_shape: bool = True,  # To be able to load old models. Set this to be False in future models.
        separate_nerf_channels: bool = False,
        separate_coarse_channels: bool = False,
        initial_density_bias: float = 0.0,
        initial_sdf_bias: float = -0.1,
        **kwargs,
    ):
        h_map, h_directionless_map = indices_for_output_mode(
            direction_dependent_shape=direction_dependent_shape,
            separate_nerf_channels=separate_nerf_channels,
            separate_coarse_channels=separate_coarse_channels,
        )
        n_output = index_mapping_max(h_map)
        super().__init__(
            n_output=n_output,
            output_activation="identity",
            **kwargs,
        )
        self.direction_dependent_shape = direction_dependent_shape
        self.separate_nerf_channels = separate_nerf_channels
        self.separate_coarse_channels = separate_coarse_channels
        self.sdf_activation = get_act(sdf_activation)
        self.density_activation = get_act(density_activation)
        self.channel_activation = get_act(channel_activation)
        self.h_map = h_map
        self.h_directionless_map = h_directionless_map
        self.mlp[-1].bias.data.zero_()
        layer = -1 if self.direction_dependent_shape else self.insert_direction_at
        self.mlp[layer].bias[0].data.fill_(initial_sdf_bias)
        self.mlp[layer].bias[1].data.fill_(initial_density_bias)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        options = AttrDict() if options is None else AttrDict(options)
        h, h_directionless = self._mlp(
            query.position, query.direction, params=params, options=options
        )
        activations = map_indices_to_keys(self.h_map, h)
        activations.update(map_indices_to_keys(self.h_directionless_map, h_directionless))

        if options.nerf_level == "coarse":
            h_density = activations.density_coarse
        else:
            h_density = activations.density_fine

        if options.get("rendering_mode", "stf") == "nerf":
            if options.nerf_level == "coarse":
                h_channels = activations.nerf_coarse
            else:
                h_channels = activations.nerf_fine
        else:
            h_channels = activations.stf
        return AttrDict(
            density=self.density_activation(h_density),
            signed_distance=self.sdf_activation(activations.sdf),
            channels=self.channel_activation(h_channels),
        )


IndexMapping = AttrDict[str, Tuple[int, int]]


def indices_for_output_mode(
    direction_dependent_shape: bool,
    separate_nerf_channels: bool,
    separate_coarse_channels: bool,
) -> Tuple[IndexMapping, IndexMapping]:
    """
    Get output mappings for (h, h_directionless).
    """
    h_map = AttrDict()
    h_directionless_map = AttrDict()
    if direction_dependent_shape:
        h_map.sdf = (0, 1)
        if separate_coarse_channels:
            assert separate_nerf_channels
            h_map.density_coarse = (1, 2)
            h_map.density_fine = (2, 3)
            h_map.stf = (3, 6)
            h_map.nerf_coarse = (6, 9)
            h_map.nerf_fine = (9, 12)
        else:
            h_map.density_coarse = (1, 2)
            h_map.density_fine = (1, 2)
            if separate_nerf_channels:
                h_map.stf = (2, 5)
                h_map.nerf_coarse = (5, 8)
                h_map.nerf_fine = (5, 8)
            else:
                h_map.stf = (2, 5)
                h_map.nerf_coarse = (2, 5)
                h_map.nerf_fine = (2, 5)
    else:
        h_directionless_map.sdf = (0, 1)
        h_directionless_map.density_coarse = (1, 2)
        if separate_coarse_channels:
            h_directionless_map.density_fine = (2, 3)
        else:
            h_directionless_map.density_fine = h_directionless_map.density_coarse
        h_map.stf = (0, 3)
        if separate_coarse_channels:
            assert separate_nerf_channels
            h_map.nerf_coarse = (3, 6)
            h_map.nerf_fine = (6, 9)
        else:
            if separate_nerf_channels:
                h_map.nerf_coarse = (3, 6)
            else:
                h_map.nerf_coarse = (0, 3)
            h_map.nerf_fine = h_map.nerf_coarse
    return h_map, h_directionless_map


def map_indices_to_keys(mapping: IndexMapping, data: torch.Tensor) -> AttrDict[str, torch.Tensor]:
    return AttrDict({k: data[..., start:end] for k, (start, end) in mapping.items()})


def index_mapping_max(mapping: IndexMapping) -> int:
    return max(end for _, (_, end) in mapping.items())
