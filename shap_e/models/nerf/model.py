from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from shap_e.models.nn.checkpoint import checkpoint
from shap_e.models.nn.encoding import encode_position, spherical_harmonics_basis
from shap_e.models.nn.meta import MetaModule, subdict
from shap_e.models.nn.ops import MLP, MetaMLP, get_act, mlp_init, zero_init
from shap_e.models.nn.utils import ArrayType
from shap_e.models.query import Query
from shap_e.util.collections import AttrDict


class NeRFModel(ABC):
    """
    Parametric scene representation whose outputs are integrated by NeRFRenderer
    """

    @abstractmethod
    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        """
        :param query: the points in the field to query.
        :param params: Meta parameters
        :param options: Optional hyperparameters
        :return: An AttrDict containing at least
            - density: [batch_size x ... x 1]
            - channels: [batch_size x ... x n_channels]
            - aux_losses: [batch_size x ... x 1]
        """


class VoidNeRFModel(MetaModule, NeRFModel):
    """
    Implements the default empty space model where all queries are rendered as
    background.
    """

    def __init__(
        self,
        background: ArrayType,
        trainable: bool = False,
        channel_scale: float = 255.0,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        background = nn.Parameter(
            torch.from_numpy(np.array(background)).to(dtype=torch.float32, device=device)
            / channel_scale
        )
        if trainable:
            self.register_parameter("background", background)
        else:
            self.register_buffer("background", background)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        _ = params
        default_bg = self.background[None]
        background = options.get("background", default_bg) if options is not None else default_bg

        shape = query.position.shape[:-1]
        ones = [1] * (len(shape) - 1)
        n_channels = background.shape[-1]
        background = torch.broadcast_to(
            background.view(background.shape[0], *ones, n_channels), [*shape, n_channels]
        )
        return background


class MLPNeRFModel(MetaModule, NeRFModel):
    def __init__(
        self,
        # Positional encoding parameters
        n_levels: int = 10,
        # MLP parameters
        d_hidden: int = 256,
        n_density_layers: int = 4,
        n_channel_layers: int = 1,
        n_channels: int = 3,
        sh_degree: int = 4,
        activation: str = "relu",
        density_activation: str = "exp",
        init: Optional[str] = None,
        init_scale: float = 1.0,
        output_activation: str = "sigmoid",
        meta_parameters: bool = False,
        trainable_meta: bool = False,
        zero_out: bool = True,
        register_freqs: bool = True,
        posenc_version: str = "v1",
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        # Positional encoding
        if register_freqs:
            # not used anymore
            self.register_buffer(
                "freqs",
                2.0 ** torch.arange(n_levels, device=device, dtype=torch.float).view(1, n_levels),
            )

        self.posenc_version = posenc_version
        dummy = torch.eye(1, 3)
        d_input = encode_position(posenc_version, position=dummy).shape[-1]

        self.n_levels = n_levels

        self.sh_degree = sh_degree
        d_sh_coeffs = sh_degree**2

        self.meta_parameters = meta_parameters

        mlp_cls = (
            partial(
                MetaMLP,
                meta_scale=False,
                meta_shift=False,
                meta_proj=True,
                meta_bias=True,
                trainable_meta=trainable_meta,
            )
            if meta_parameters
            else MLP
        )

        self.density_mlp = mlp_cls(
            d_input=d_input,
            d_hidden=[d_hidden] * (n_density_layers - 1),
            d_output=d_hidden,
            act_name=activation,
            init_scale=init_scale,
        )

        self.channel_mlp = mlp_cls(
            d_input=d_hidden + d_sh_coeffs,
            d_hidden=[d_hidden] * n_channel_layers,
            d_output=n_channels,
            act_name=activation,
            init_scale=init_scale,
        )

        self.act = get_act(output_activation)
        self.density_act = get_act(density_activation)

        mlp_init(
            list(self.density_mlp.affines) + list(self.channel_mlp.affines),
            init=init,
            init_scale=init_scale,
        )

        if zero_out:
            zero_init(self.channel_mlp.affines[-1])

        self.to(device)

    def encode_position(self, query: Query):
        h = encode_position(self.posenc_version, position=query.position)
        return h

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        params = self.update(params)

        options = AttrDict() if options is None else AttrDict(options)

        query = query.copy()

        h_position = self.encode_position(query)

        if self.meta_parameters:
            density_params = subdict(params, "density_mlp")
            density_mlp = partial(
                self.density_mlp, params=density_params, options=options, log_prefix="density_"
            )
            density_mlp_parameters = list(density_params.values())
        else:
            density_mlp = partial(self.density_mlp, options=options, log_prefix="density_")
            density_mlp_parameters = self.density_mlp.parameters()
        h_density = checkpoint(
            density_mlp,
            (h_position,),
            density_mlp_parameters,
            options.checkpoint_nerf_mlp,
        )
        h_direction = maybe_get_spherical_harmonics_basis(
            sh_degree=self.sh_degree,
            coords_shape=query.position.shape,
            coords=query.direction,
            device=query.position.device,
        )

        if self.meta_parameters:
            channel_params = subdict(params, "channel_mlp")
            channel_mlp = partial(
                self.channel_mlp, params=channel_params, options=options, log_prefix="channel_"
            )
            channel_mlp_parameters = list(channel_params.values())
        else:
            channel_mlp = partial(self.channel_mlp, options=options, log_prefix="channel_")
            channel_mlp_parameters = self.channel_mlp.parameters()
        h_channel = checkpoint(
            channel_mlp,
            (torch.cat([h_density, h_direction], dim=-1),),
            channel_mlp_parameters,
            options.checkpoint_nerf_mlp,
        )

        density_logit = h_density[..., :1]

        res = AttrDict(
            density_logit=density_logit,
            density=self.density_act(density_logit),
            channels=self.act(h_channel),
            aux_losses=AttrDict(),
            no_weight_grad_aux_losses=AttrDict(),
        )
        if options.return_h_density:
            res.h_density = h_density

        return res


def maybe_get_spherical_harmonics_basis(
    sh_degree: int,
    coords_shape: Tuple[int],
    coords: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    :param sh_degree: Spherical harmonics degree
    :param coords_shape: [*shape, 3]
    :param coords: optional coordinate tensor of coords_shape
    """
    if coords is None:
        return torch.zeros(*coords_shape[:-1], sh_degree**2).to(device)

    return spherical_harmonics_basis(coords, sh_degree)
