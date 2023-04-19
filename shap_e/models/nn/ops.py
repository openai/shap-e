import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shap_e.util.collections import AttrDict

from .meta import MetaModule, subdict
from .pointnet2_utils import sample_and_group, sample_and_group_all


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def torch_gelu(x):
    return torch.nn.functional.gelu(x)


def geglu(x):
    v, gates = x.chunk(2, dim=-1)
    return v * gelu(gates)


class SirenSin:
    def __init__(self, w0=30.0):
        self.w0 = w0

    def __call__(self, x):
        return torch.sin(self.w0 * x)


def get_act(name):
    return {
        "relu": torch.nn.functional.relu,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "swish": swish,
        "tanh": torch.tanh,
        "gelu": gelu,
        "quick_gelu": quick_gelu,
        "torch_gelu": torch_gelu,
        "gelu2": quick_gelu,
        "geglu": geglu,
        "sigmoid": torch.sigmoid,
        "sin": torch.sin,
        "sin30": SirenSin(w0=30.0),
        "softplus": F.softplus,
        "exp": torch.exp,
        "identity": lambda x: x,
    }[name]


def zero_init(affine):
    nn.init.constant_(affine.weight, 0.0)
    if affine.bias is not None:
        nn.init.constant_(affine.bias, 0.0)


def siren_init_first_layer(affine, init_scale: float = 1.0):
    n_input = affine.weight.shape[1]
    u = init_scale / n_input
    nn.init.uniform_(affine.weight, -u, u)
    if affine.bias is not None:
        nn.init.constant_(affine.bias, 0.0)


def siren_init(affine, coeff=1.0, init_scale: float = 1.0):
    n_input = affine.weight.shape[1]
    u = init_scale * np.sqrt(6.0 / n_input) / coeff
    nn.init.uniform_(affine.weight, -u, u)
    if affine.bias is not None:
        nn.init.constant_(affine.bias, 0.0)


def siren_init_30(affine, init_scale: float = 1.0):
    siren_init(affine, coeff=30.0, init_scale=init_scale)


def std_init(affine, init_scale: float = 1.0):
    n_in = affine.weight.shape[1]
    stddev = init_scale / math.sqrt(n_in)
    nn.init.normal_(affine.weight, std=stddev)
    if affine.bias is not None:
        nn.init.constant_(affine.bias, 0.0)


def mlp_init(affines, init: Optional[str] = None, init_scale: float = 1.0):
    if init == "siren30":
        for idx, affine in enumerate(affines):
            init = siren_init_first_layer if idx == 0 else siren_init_30
            init(affine, init_scale=init_scale)
    elif init == "siren":
        for idx, affine in enumerate(affines):
            init = siren_init_first_layer if idx == 0 else siren_init
            init(affine, init_scale=init_scale)
    elif init is None:
        for affine in affines:
            std_init(affine, init_scale=init_scale)
    else:
        raise NotImplementedError(init)


class MetaLinear(MetaModule):
    def __init__(
        self,
        n_in,
        n_out,
        bias: bool = True,
        meta_scale: bool = True,
        meta_shift: bool = True,
        meta_proj: bool = False,
        meta_bias: bool = False,
        trainable_meta: bool = False,
        **kwargs,
    ):
        super().__init__()
        # n_in, n_out, bias=bias)
        register_meta_fn = (
            self.register_meta_parameter if trainable_meta else self.register_meta_buffer
        )
        if meta_scale:
            register_meta_fn("scale", nn.Parameter(torch.ones(n_out, **kwargs)))
        if meta_shift:
            register_meta_fn("shift", nn.Parameter(torch.zeros(n_out, **kwargs)))

        register_proj_fn = self.register_parameter if not meta_proj else register_meta_fn
        register_proj_fn("weight", nn.Parameter(torch.empty((n_out, n_in), **kwargs)))

        if not bias:
            self.register_parameter("bias", None)
        else:
            register_bias_fn = self.register_parameter if not meta_bias else register_meta_fn
            register_bias_fn("bias", nn.Parameter(torch.empty(n_out, **kwargs)))

        self.reset_parameters()

    def reset_parameters(self) -> None:

        # from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _bcast(self, op, left, right):
        if right.ndim == 2:
            # Has dimension [batch x d_output]
            right = right.unsqueeze(1)
        return op(left, right)

    def forward(self, x, params=None):
        params = self.update(params)

        batch_size, *shape, d_in = x.shape
        x = x.view(batch_size, -1, d_in)

        if params.weight.ndim == 2:
            h = torch.einsum("bni,oi->bno", x, params.weight)
        elif params.weight.ndim == 3:
            h = torch.einsum("bni,boi->bno", x, params.weight)

        if params.bias is not None:
            h = self._bcast(torch.add, h, params.bias)

        if params.scale is not None:
            h = self._bcast(torch.mul, h, params.scale)

        if params.shift is not None:
            h = self._bcast(torch.add, h, params.shift)

        h = h.view(batch_size, *shape, -1)
        return h


def Conv(n_dim, d_in, d_out, kernel, stride=1, padding=0, dilation=1, **kwargs):
    cls = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }[n_dim]
    return cls(d_in, d_out, kernel, stride=stride, padding=padding, dilation=dilation, **kwargs)


def flatten(x):
    batch_size, *shape, n_channels = x.shape
    n_ctx = np.prod(shape)
    return x.view(batch_size, n_ctx, n_channels), AttrDict(
        shape=shape, n_ctx=n_ctx, n_channels=n_channels
    )


def unflatten(x, info):
    batch_size = x.shape[0]
    return x.view(batch_size, *info.shape, info.n_channels)


def torchify(x):
    extent = list(range(1, x.ndim - 1))
    return x.permute([0, x.ndim - 1, *extent])


def untorchify(x):
    extent = list(range(2, x.ndim))
    return x.permute([0, *extent, 1])


class MLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: List[int],
        d_output: int,
        act_name: str = "quick_gelu",
        bias: bool = True,
        init: Optional[str] = None,
        init_scale: float = 1.0,
        zero_out: bool = False,
    ):
        """
        Required: d_input, d_hidden, d_output
        Optional: act_name, bias
        """
        super().__init__()

        ds = [d_input] + d_hidden + [d_output]
        affines = [nn.Linear(d_in, d_out, bias=bias) for d_in, d_out in zip(ds[:-1], ds[1:])]
        self.d = ds
        self.affines = nn.ModuleList(affines)
        self.act = get_act(act_name)

        mlp_init(self.affines, init=init, init_scale=init_scale)
        if zero_out:
            zero_init(affines[-1])

    def forward(self, h, options: Optional[AttrDict] = None, log_prefix: str = ""):
        options = AttrDict() if options is None else AttrDict(options)
        *hid, out = self.affines
        for i, f in enumerate(hid):
            h = self.act(f(h))
        h = out(h)
        return h


class MetaMLP(MetaModule):
    def __init__(
        self,
        d_input: int,
        d_hidden: List[int],
        d_output: int,
        act_name: str = "quick_gelu",
        bias: bool = True,
        meta_scale: bool = True,
        meta_shift: bool = True,
        meta_proj: bool = False,
        meta_bias: bool = False,
        trainable_meta: bool = False,
        init: Optional[str] = None,
        init_scale: float = 1.0,
        zero_out: bool = False,
    ):
        super().__init__()
        ds = [d_input] + d_hidden + [d_output]
        affines = [
            MetaLinear(
                d_in,
                d_out,
                bias=bias,
                meta_scale=meta_scale,
                meta_shift=meta_shift,
                meta_proj=meta_proj,
                meta_bias=meta_bias,
                trainable_meta=trainable_meta,
            )
            for d_in, d_out in zip(ds[:-1], ds[1:])
        ]
        self.d = ds
        self.affines = nn.ModuleList(affines)
        self.act = get_act(act_name)

        mlp_init(affines, init=init, init_scale=init_scale)
        if zero_out:
            zero_init(affines[-1])

    def forward(self, h, params=None, options: Optional[AttrDict] = None, log_prefix: str = ""):
        options = AttrDict() if options is None else AttrDict(options)
        params = self.update(params)
        *hid, out = self.affines
        for i, layer in enumerate(hid):
            h = self.act(layer(h, params=subdict(params, f"{log_prefix}affines.{i}")))
        last = len(self.affines) - 1
        h = out(h, params=subdict(params, f"{log_prefix}affines.{last}"))
        return h


class LayerNorm(nn.LayerNorm):
    def __init__(
        self, norm_shape: Union[int, Tuple[int]], eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__(norm_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.width = np.prod(norm_shape)
        self.max_numel = 65535 * self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(
                input.float(), self.normalized_shape, self.weight, self.bias, self.eps
            ).type_as(input)
        else:
            return super(LayerNorm, self).forward(input.float()).type_as(input)


class PointSetEmbedding(nn.Module):
    def __init__(
        self,
        *,
        radius: float,
        n_point: int,
        n_sample: int,
        d_input: int,
        d_hidden: List[int],
        patch_size: int = 1,
        stride: int = 1,
        activation: str = "swish",
        group_all: bool = False,
        padding_mode: str = "zeros",
        fps_method: str = "fps",
        **kwargs,
    ):
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample
        self.mlp_convs = nn.ModuleList()
        self.act = get_act(activation)
        self.patch_size = patch_size
        self.stride = stride
        last_channel = d_input + 3
        for out_channel in d_hidden:
            self.mlp_convs.append(
                nn.Conv2d(
                    last_channel,
                    out_channel,
                    kernel_size=(patch_size, 1),
                    stride=(stride, 1),
                    padding=(patch_size // 2, 0),
                    padding_mode=padding_mode,
                    **kwargs,
                )
            )
            last_channel = out_channel
        self.group_all = group_all
        self.fps_method = fps_method

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: sample points feature data, [B, d_hidden[-1], n_point]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.n_point,
                self.radius,
                self.n_sample,
                xyz,
                points,
                deterministic=not self.training,
                fps_method=self.fps_method,
            )
        # new_xyz: sampled points position data, [B, n_point, C]
        # new_points: sampled points data, [B, n_point, n_sample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, n_sample, n_point]
        for i, conv in enumerate(self.mlp_convs):
            new_points = self.act(self.apply_conv(new_points, conv))

        new_points = new_points.mean(dim=2)
        return new_points

    def apply_conv(self, points: torch.Tensor, conv: nn.Module):
        batch, channels, n_samples, _ = points.shape
        # Shuffle the representations
        if self.patch_size > 1:
            # TODO shuffle deterministically when not self.training
            _, indices = torch.rand(batch, channels, n_samples, 1, device=points.device).sort(dim=2)
            points = torch.gather(points, 2, torch.broadcast_to(indices, points.shape))
        return conv(points)
