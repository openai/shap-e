from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from shap_e.models.nn.checkpoint import checkpoint
from shap_e.models.nn.encoding import encode_position, maybe_encode_direction
from shap_e.models.nn.meta import MetaModule, subdict
from shap_e.models.nn.ops import MetaLinear, get_act, mlp_init
from shap_e.models.query import Query
from shap_e.util.collections import AttrDict

from .base import Model


class MLPModel(MetaModule, Model):
    def __init__(
        self,
        n_output: int,
        output_activation: str,
        # Positional encoding parameters
        posenc_version: str = "v1",
        # Direction related channel prediction
        insert_direction_at: Optional[int] = None,
        # MLP parameters
        d_hidden: int = 256,
        n_hidden_layers: int = 4,
        activation: str = "relu",
        init: Optional[str] = None,
        init_scale: float = 1.0,
        meta_parameters: bool = False,
        trainable_meta: bool = False,
        meta_proj: bool = True,
        meta_bias: bool = True,
        meta_start: int = 0,
        meta_stop: Optional[int] = None,
        n_meta_layers: Optional[int] = None,
        register_freqs: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        if register_freqs:
            self.register_buffer("freqs", 2.0 ** torch.arange(10, device=device).view(1, 10))

        # Positional encoding
        self.posenc_version = posenc_version
        dummy = torch.eye(1, 3)
        d_posenc_pos = encode_position(posenc_version, position=dummy).shape[-1]
        d_posenc_dir = maybe_encode_direction(posenc_version, position=dummy).shape[-1]

        # Instantiate the MLP
        mlp_widths = [d_hidden] * n_hidden_layers
        input_widths = [d_posenc_pos, *mlp_widths]
        output_widths = mlp_widths + [n_output]

        self.meta_parameters = meta_parameters

        # When this model is used jointly to express NeRF, it may have to
        # process directions as well in which case we simply concatenate
        # the direction representation at the specified layer.
        self.insert_direction_at = insert_direction_at
        if insert_direction_at is not None:
            input_widths[self.insert_direction_at] += d_posenc_dir

        linear_cls = lambda meta: (
            partial(
                MetaLinear,
                meta_scale=False,
                meta_shift=False,
                meta_proj=meta_proj,
                meta_bias=meta_bias,
                trainable_meta=trainable_meta,
            )
            if meta
            else nn.Linear
        )

        if meta_stop is None:
            if n_meta_layers is not None:
                assert n_meta_layers > 0
                meta_stop = meta_start + n_meta_layers - 1
            else:
                meta_stop = n_hidden_layers

        if meta_parameters:
            metas = [meta_start <= layer <= meta_stop for layer in range(n_hidden_layers + 1)]
        else:
            metas = [False] * (n_hidden_layers + 1)

        self.mlp = nn.ModuleList(
            [
                linear_cls(meta)(d_in, d_out, device=device)
                for meta, d_in, d_out in zip(metas, input_widths, output_widths)
            ]
        )

        mlp_init(self.mlp, init=init, init_scale=init_scale)

        self.activation = get_act(activation)
        self.output_activation = get_act(output_activation)

        self.device = device
        self.to(device)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        """
        :param position: [batch_size x ... x 3]
        :param params: Meta parameters
        :param options: Optional hyperparameters
        """

        # query.direction is None typically for SDF models and training
        h_final, _h_directionless = self._mlp(
            query.position, query.direction, params=params, options=options
        )
        return self.output_activation(h_final)

    def _run_mlp(
        self, position: torch.Tensor, direction: torch.Tensor, params: AttrDict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: the final and directionless activations at the given query
        """
        h_preact = h = encode_position(self.posenc_version, position=position)
        h_directionless = None
        for i, layer in enumerate(self.mlp):
            if i == self.insert_direction_at:
                h_directionless = h_preact
                h_direction = maybe_encode_direction(
                    self.posenc_version, position=position, direction=direction
                )
                h = torch.cat([h, h_direction], dim=-1)
            if isinstance(layer, MetaLinear):
                h = layer(h, params=subdict(params, f"mlp.{i}"))
            else:
                h = layer(h)
            h_preact = h
            if i < len(self.mlp) - 1:
                h = self.activation(h)
        h_final = h
        if h_directionless is None:
            h_directionless = h_preact
        return h_final, h_directionless

    def _mlp(
        self,
        position: torch.Tensor,
        direction: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param position: [batch_size x ... x 3]
        :param params: Meta parameters
        :param options: Optional hyperparameters
        :return: the final and directionless activations at the given query
        """
        params = self.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        mlp = partial(self._run_mlp, direction=direction, params=params)
        parameters = []
        for i, layer in enumerate(self.mlp):
            if isinstance(layer, MetaLinear):
                parameters.extend(list(subdict(params, f"mlp.{i}").values()))
            else:
                parameters.extend(layer.parameters())

        h_final, h_directionless = checkpoint(
            mlp, (position,), parameters, options.checkpoint_stf_model
        )

        return h_final, h_directionless


class MLPSDFModel(MLPModel):
    def __init__(self, initial_bias: float = -0.1, **kwargs):
        super().__init__(n_output=1, output_activation="identity", **kwargs)
        self.mlp[-1].bias.data.fill_(initial_bias)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        signed_distance = super().forward(query=query, params=params, options=options)
        return AttrDict(signed_distance=signed_distance)


class MLPTextureFieldModel(MLPModel):
    def __init__(
        self,
        n_channels: int = 3,
        **kwargs,
    ):
        super().__init__(n_output=n_channels, output_activation="sigmoid", **kwargs)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        channels = super().forward(query=query, params=params, options=options)
        return AttrDict(channels=channels)
