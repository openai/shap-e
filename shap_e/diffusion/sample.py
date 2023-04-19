from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .gaussian_diffusion import GaussianDiffusion
from .k_diffusion import karras_sample

DEFAULT_KARRAS_STEPS = 64
DEFAULT_KARRAS_SIGMA_MIN = 1e-3
DEFAULT_KARRAS_SIGMA_MAX = 160
DEFAULT_KARRAS_S_CHURN = 0.0


def uncond_guide_model(
    model: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
        half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    return model_fn


def sample_latents(
    *,
    batch_size: int,
    model: nn.Module,
    diffusion: GaussianDiffusion,
    model_kwargs: Dict[str, Any],
    guidance_scale: float,
    clip_denoised: bool,
    use_fp16: bool,
    use_karras: bool,
    karras_steps: int,
    sigma_min: float,
    sigma_max: float,
    s_churn: float,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> torch.Tensor:
    sample_shape = (batch_size, model.d_latent)

    if device is None:
        device = next(model.parameters()).device

    if hasattr(model, "cached_model_kwargs"):
        model_kwargs = model.cached_model_kwargs(batch_size, model_kwargs)
    if guidance_scale != 1.0 and guidance_scale != 0.0:
        for k, v in model_kwargs.copy().items():
            model_kwargs[k] = torch.cat([v, torch.zeros_like(v)], dim=0)

    sample_shape = (batch_size, model.d_latent)
    with torch.autocast(device_type=device.type, enabled=use_fp16):
        if use_karras:
            samples = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=sample_shape,
                steps=karras_steps,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                s_churn=s_churn,
                guidance_scale=guidance_scale,
                progress=progress,
            )
        else:
            internal_batch_size = batch_size
            if guidance_scale != 1.0:
                model = uncond_guide_model(model, guidance_scale)
                internal_batch_size *= 2
            samples = diffusion.p_sample_loop(
                model,
                shape=(internal_batch_size, *sample_shape[1:]),
                model_kwargs=model_kwargs,
                device=device,
                clip_denoised=clip_denoised,
                progress=progress,
            )

    return samples
