from typing import Any, Dict

import torch
import torch.nn as nn


class SplitVectorDiffusion(nn.Module):
    def __init__(self, *, device: torch.device, wrapped: nn.Module, n_ctx: int, d_latent: int):
        super().__init__()
        self.device = device
        self.n_ctx = n_ctx
        self.d_latent = d_latent
        self.wrapped = wrapped

        if hasattr(self.wrapped, "cached_model_kwargs"):
            self.cached_model_kwargs = self.wrapped.cached_model_kwargs

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        h = x.reshape(x.shape[0], self.n_ctx, -1).permute(0, 2, 1)
        pre_channels = h.shape[1]
        h = self.wrapped(h, t, **kwargs)
        assert (
            h.shape[1] == pre_channels * 2
        ), "expected twice as many outputs for variance prediction"
        eps, var = torch.chunk(h, 2, dim=1)
        return torch.cat(
            [
                eps.permute(0, 2, 1).flatten(1),
                var.permute(0, 2, 1).flatten(1),
            ],
            dim=1,
        )
