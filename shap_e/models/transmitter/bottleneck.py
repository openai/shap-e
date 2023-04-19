from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch.nn as nn
from torch import torch

from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.collections import AttrDict


class LatentBottleneck(nn.Module, ABC):
    def __init__(self, *, device: torch.device, d_latent: int):
        super().__init__()
        self.device = device
        self.d_latent = d_latent

    @abstractmethod
    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass


class LatentWarp(nn.Module, ABC):
    def __init__(self, *, device: torch.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def warp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass

    @abstractmethod
    def unwarp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass


class IdentityLatentWarp(LatentWarp):
    def warp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return x

    def unwarp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return x


class Tan2LatentWarp(LatentWarp):
    def __init__(self, *, coeff1: float = 1.0, device: torch.device):
        super().__init__(device=device)
        self.coeff1 = coeff1
        self.scale = np.tan(np.tan(1.0) * coeff1)

    def warp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return ((x.float().tan() * self.coeff1).tan() / self.scale).to(x.dtype)

    def unwarp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return ((x.float() * self.scale).arctan() / self.coeff1).arctan().to(x.dtype)


class IdentityLatentBottleneck(LatentBottleneck):
    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return x


class ClampNoiseBottleneck(LatentBottleneck):
    def __init__(self, *, device: torch.device, d_latent: int, noise_scale: float):
        super().__init__(device=device, d_latent=d_latent)
        self.noise_scale = noise_scale

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        x = x.tanh()
        if not self.training:
            return x
        return x + torch.randn_like(x) * self.noise_scale


class ClampDiffusionNoiseBottleneck(LatentBottleneck):
    def __init__(
        self,
        *,
        device: torch.device,
        d_latent: int,
        diffusion: Dict[str, Any],
        diffusion_prob: float = 1.0,
    ):
        super().__init__(device=device, d_latent=d_latent)
        self.diffusion = diffusion_from_config(diffusion)
        self.diffusion_prob = diffusion_prob

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        x = x.tanh()
        if not self.training:
            return x
        t = torch.randint(low=0, high=self.diffusion.num_timesteps, size=(len(x),), device=x.device)
        t = torch.where(
            torch.rand(len(x), device=x.device) < self.diffusion_prob, t, torch.zeros_like(t)
        )
        return self.diffusion.q_sample(x, t)


def latent_bottleneck_from_config(config: Dict[str, Any], device: torch.device, d_latent: int):
    name = config.pop("name")
    if name == "clamp_noise":
        return ClampNoiseBottleneck(**config, device=device, d_latent=d_latent)
    elif name == "identity":
        return IdentityLatentBottleneck(**config, device=device, d_latent=d_latent)
    elif name == "clamp_diffusion_noise":
        return ClampDiffusionNoiseBottleneck(**config, device=device, d_latent=d_latent)
    else:
        raise ValueError(f"unknown latent bottleneck: {name}")


def latent_warp_from_config(config: Dict[str, Any], device: torch.device):
    name = config.pop("name")
    if name == "identity":
        return IdentityLatentWarp(**config, device=device)
    elif name == "tan2":
        return Tan2LatentWarp(**config, device=device)
    else:
        raise ValueError(f"unknown latent warping function: {name}")
