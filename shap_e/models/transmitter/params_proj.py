import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch.nn as nn
from torch import torch

from shap_e.util.collections import AttrDict


def flatten_param_shapes(param_shapes: Dict[str, Tuple[int]]):
    flat_shapes = OrderedDict(
        (name, (int(np.prod(shape)) // shape[-1], shape[-1]))
        for name, shape in param_shapes.items()
    )
    return flat_shapes


class ParamsProj(nn.Module, ABC):
    def __init__(self, *, device: torch.device, param_shapes: Dict[str, Tuple[int]], d_latent: int):
        super().__init__()
        self.device = device
        self.param_shapes = param_shapes
        self.d_latent = d_latent

    @abstractmethod
    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass


class LinearParamsProj(ParamsProj):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        d_latent: int,
        init_scale: Optional[float] = None,
    ):
        super().__init__(device=device, param_shapes=param_shapes, d_latent=d_latent)
        self.param_shapes = param_shapes
        self.projections = nn.ModuleDict({})
        for k, v in param_shapes.items():
            self.projections[_sanitize_name(k)] = nn.Linear(
                d_latent, int(np.prod(v)), device=device
            )
            if init_scale is not None:
                scale = init_scale / math.sqrt(d_latent)
                mod = self.projections[_sanitize_name(k)]
                nn.init.normal_(mod.weight, std=scale)
                nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        out = AttrDict()
        for k in self.param_shapes.keys():
            proj = self.projections[_sanitize_name(k)]
            out[k] = proj(x).reshape([len(x), *self.param_shapes[k]])
        return out


class MLPParamsProj(ParamsProj):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        d_latent: int,
        hidden_size: Optional[int] = None,
    ):
        super().__init__(device=device, param_shapes=param_shapes, d_latent=d_latent)
        if hidden_size is None:
            hidden_size = d_latent
        self.param_shapes = param_shapes
        self.projections = nn.ModuleDict({})
        for k, v in param_shapes.items():
            self.projections[_sanitize_name(k)] = nn.Sequential(
                nn.Linear(d_latent, hidden_size, device=device),
                nn.GELU(),
                nn.Linear(hidden_size, int(np.prod(v)), device=device),
            )

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        out = AttrDict()
        for k in self.param_shapes.keys():
            proj = self.projections[_sanitize_name(k)]
            out[k] = proj(x).reshape([len(x), *self.param_shapes[k]])
        return out


class ChannelsProj(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        vectors: int,
        channels: int,
        d_latent: int,
        init_scale: float = 1.0,
        learned_scale: Optional[float] = None,
        use_ln: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(d_latent, vectors * channels, device=device)
        self.use_ln = use_ln
        self.learned_scale = learned_scale
        if use_ln:
            self.norm = nn.LayerNorm(normalized_shape=(channels,), device=device)
            if learned_scale is not None:
                self.norm.weight.data.fill_(learned_scale)
            scale = init_scale / math.sqrt(d_latent)
        elif learned_scale is not None:
            gain = torch.ones((channels,), device=device) * learned_scale
            self.register_parameter("gain", nn.Parameter(gain))
            scale = init_scale / math.sqrt(d_latent)
        else:
            scale = init_scale / math.sqrt(d_latent * channels)
        nn.init.normal_(self.proj.weight, std=scale)
        nn.init.zeros_(self.proj.bias)
        self.d_latent = d_latent
        self.vectors = vectors
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bvd = x
        w_vcd = self.proj.weight.view(self.vectors, self.channels, self.d_latent)
        b_vc = self.proj.bias.view(1, self.vectors, self.channels)
        h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        if self.use_ln:
            h = self.norm(h)
        elif self.learned_scale is not None:
            h = h * self.gain.view(1, 1, -1)
        h = h + b_vc
        return h


class ChannelsParamsProj(ParamsProj):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        d_latent: int,
        init_scale: float = 1.0,
        learned_scale: Optional[float] = None,
        use_ln: bool = False,
    ):
        super().__init__(device=device, param_shapes=param_shapes, d_latent=d_latent)
        self.param_shapes = param_shapes
        self.projections = nn.ModuleDict({})
        self.flat_shapes = flatten_param_shapes(param_shapes)
        self.learned_scale = learned_scale
        self.use_ln = use_ln
        for k, (vectors, channels) in self.flat_shapes.items():
            self.projections[_sanitize_name(k)] = ChannelsProj(
                device=device,
                vectors=vectors,
                channels=channels,
                d_latent=d_latent,
                init_scale=init_scale,
                learned_scale=learned_scale,
                use_ln=use_ln,
            )

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        out = AttrDict()
        start = 0
        for k, shape in self.param_shapes.items():
            vectors, _ = self.flat_shapes[k]
            end = start + vectors
            x_bvd = x[:, start:end]
            out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape(len(x), *shape)
            start = end
        return out


def params_proj_from_config(
    config: Dict[str, Any], device: torch.device, param_shapes: Dict[str, Tuple[int]], d_latent: int
):
    name = config.pop("name")
    if name == "linear":
        return LinearParamsProj(
            **config, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
    elif name == "mlp":
        return MLPParamsProj(**config, device=device, param_shapes=param_shapes, d_latent=d_latent)
    elif name == "channels":
        return ChannelsParamsProj(
            **config, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
    else:
        raise ValueError(f"unknown params proj: {name}")


def _sanitize_name(x: str) -> str:
    return x.replace(".", "__")
