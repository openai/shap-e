from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from torch import torch

from shap_e.models.renderer import Renderer
from shap_e.util.collections import AttrDict

from .bottleneck import latent_bottleneck_from_config, latent_warp_from_config
from .params_proj import flatten_param_shapes, params_proj_from_config


class Encoder(nn.Module, ABC):
    def __init__(self, *, device: torch.device, param_shapes: Dict[str, Tuple[int]]):
        """
        Instantiate the encoder with information about the renderer's input
        parameters. This information can be used to create output layers to
        generate the necessary latents.
        """
        super().__init__()
        self.param_shapes = param_shapes
        self.device = device

    @abstractmethod
    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        """
        Encode a batch of data into a batch of latent information.
        """


class VectorEncoder(Encoder):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        d_latent: int,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(device=device, param_shapes=param_shapes)
        if latent_bottleneck is None:
            latent_bottleneck = dict(name="identity")
        if latent_warp is None:
            latent_warp = dict(name="identity")
        self.d_latent = d_latent
        self.params_proj = params_proj_from_config(
            params_proj, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
        self.latent_bottleneck = latent_bottleneck_from_config(
            latent_bottleneck, device=device, d_latent=d_latent
        )
        self.latent_warp = latent_warp_from_config(latent_warp, device=device)

    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        h = self.encode_to_bottleneck(batch, options=options)
        return self.bottleneck_to_params(h, options=options)

    def encode_to_bottleneck(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        return self.latent_warp.warp(
            self.latent_bottleneck(self.encode_to_vector(batch, options=options), options=options),
            options=options,
        )

    @abstractmethod
    def encode_to_vector(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        """
        Encode the batch into a single latent vector.
        """

    def bottleneck_to_params(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> AttrDict:
        _ = options
        return self.params_proj(self.latent_warp.unwarp(vector, options=options), options=options)


class ChannelsEncoder(VectorEncoder):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        d_latent: int,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            d_latent=d_latent,
            latent_bottleneck=latent_bottleneck,
            latent_warp=latent_warp,
        )
        self.flat_shapes = flatten_param_shapes(param_shapes)
        self.latent_ctx = sum(flat[0] for flat in self.flat_shapes.values())

    @abstractmethod
    def encode_to_channels(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        """
        Encode the batch into a per-data-point set of latents.
        :return: [batch_size, latent_ctx, latent_width]
        """

    def encode_to_vector(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        return self.encode_to_channels(batch, options=options).flatten(1)

    def bottleneck_to_channels(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        _ = options
        return vector.view(vector.shape[0], self.latent_ctx, -1)

    def bottleneck_to_params(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> AttrDict:
        _ = options
        return self.params_proj(
            self.bottleneck_to_channels(self.latent_warp.unwarp(vector)), options=options
        )


class Transmitter(nn.Module):
    def __init__(self, encoder: Encoder, renderer: Renderer):
        super().__init__()
        self.encoder = encoder
        self.renderer = renderer

    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        """
        Transmit the batch through the encoder and then the renderer.
        """
        params = self.encoder(batch, options=options)
        return self.renderer(batch, params=params, options=options)


class VectorDecoder(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        d_latent: int,
        latent_warp: Optional[Dict[str, Any]] = None,
        renderer: Renderer,
    ):
        super().__init__()
        self.device = device
        self.param_shapes = param_shapes

        if latent_warp is None:
            latent_warp = dict(name="identity")
        self.d_latent = d_latent
        self.params_proj = params_proj_from_config(
            params_proj, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
        self.latent_warp = latent_warp_from_config(latent_warp, device=device)
        self.renderer = renderer

    def bottleneck_to_params(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> AttrDict:
        _ = options
        return self.params_proj(self.latent_warp.unwarp(vector, options=options), options=options)


class ChannelsDecoder(VectorDecoder):
    def __init__(
        self,
        *,
        latent_ctx: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_ctx = latent_ctx

    def bottleneck_to_channels(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        _ = options
        return vector.view(vector.shape[0], self.latent_ctx, -1)

    def bottleneck_to_params(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> AttrDict:
        _ = options
        return self.params_proj(
            self.bottleneck_to_channels(self.latent_warp.unwarp(vector)), options=options
        )
