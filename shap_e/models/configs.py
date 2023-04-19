from typing import Any, Dict, Union

import blobfile as bf
import torch
import torch.nn as nn
import yaml

from shap_e.models.generation.latent_diffusion import SplitVectorDiffusion
from shap_e.models.generation.perceiver import PointDiffusionPerceiver
from shap_e.models.generation.pooled_mlp import PooledMLP
from shap_e.models.generation.transformer import (
    CLIPImageGridPointDiffusionTransformer,
    CLIPImageGridUpsamplePointDiffusionTransformer,
    CLIPImagePointDiffusionTransformer,
    PointDiffusionTransformer,
    UpsamplePointDiffusionTransformer,
)
from shap_e.models.nerf.model import MLPNeRFModel, VoidNeRFModel
from shap_e.models.nerf.renderer import OneStepNeRFRenderer, TwoStepNeRFRenderer
from shap_e.models.nerstf.mlp import MLPDensitySDFModel, MLPNeRSTFModel
from shap_e.models.nerstf.renderer import NeRSTFRenderer
from shap_e.models.nn.meta import batch_meta_state_dict
from shap_e.models.stf.mlp import MLPSDFModel, MLPTextureFieldModel
from shap_e.models.stf.renderer import STFRenderer
from shap_e.models.transmitter.base import ChannelsDecoder, Transmitter, VectorDecoder
from shap_e.models.transmitter.channels_encoder import (
    PointCloudPerceiverChannelsEncoder,
    PointCloudTransformerChannelsEncoder,
)
from shap_e.models.transmitter.multiview_encoder import MultiviewTransformerEncoder
from shap_e.models.transmitter.pc_encoder import (
    PointCloudPerceiverEncoder,
    PointCloudTransformerEncoder,
)
from shap_e.models.volume import BoundingBoxVolume, SphericalVolume, UnboundedVolume


def model_from_config(config: Union[str, Dict[str, Any]], device: torch.device) -> nn.Module:
    if isinstance(config, str):
        with bf.BlobFile(config, "rb") as f:
            obj = yaml.load(f, Loader=yaml.SafeLoader)
        return model_from_config(obj, device=device)

    config = config.copy()
    name = config.pop("name")

    if name == "PointCloudTransformerEncoder":
        return PointCloudTransformerEncoder(device=device, dtype=torch.float32, **config)
    elif name == "PointCloudPerceiverEncoder":
        return PointCloudPerceiverEncoder(device=device, dtype=torch.float32, **config)
    elif name == "PointCloudTransformerChannelsEncoder":
        return PointCloudTransformerChannelsEncoder(device=device, dtype=torch.float32, **config)
    elif name == "PointCloudPerceiverChannelsEncoder":
        return PointCloudPerceiverChannelsEncoder(device=device, dtype=torch.float32, **config)
    elif name == "MultiviewTransformerEncoder":
        return MultiviewTransformerEncoder(device=device, dtype=torch.float32, **config)
    elif name == "Transmitter":
        renderer = model_from_config(config.pop("renderer"), device=device)
        param_shapes = {
            k: v.shape[1:] for k, v in batch_meta_state_dict(renderer, batch_size=1).items()
        }
        encoder_config = config.pop("encoder").copy()
        encoder_config["param_shapes"] = param_shapes
        encoder = model_from_config(encoder_config, device=device)
        return Transmitter(encoder=encoder, renderer=renderer, **config)
    elif name == "VectorDecoder":
        renderer = model_from_config(config.pop("renderer"), device=device)
        param_shapes = {
            k: v.shape[1:] for k, v in batch_meta_state_dict(renderer, batch_size=1).items()
        }
        return VectorDecoder(param_shapes=param_shapes, renderer=renderer, device=device, **config)
    elif name == "ChannelsDecoder":
        renderer = model_from_config(config.pop("renderer"), device=device)
        param_shapes = {
            k: v.shape[1:] for k, v in batch_meta_state_dict(renderer, batch_size=1).items()
        }
        return ChannelsDecoder(
            param_shapes=param_shapes, renderer=renderer, device=device, **config
        )
    elif name == "OneStepNeRFRenderer":
        config = config.copy()
        for field in [
            # Required
            "void_model",
            "foreground_model",
            "volume",
            # Optional to use NeRF++
            "background_model",
            "outer_volume",
        ]:
            if field in config:
                config[field] = model_from_config(config.pop(field).copy(), device)
        return OneStepNeRFRenderer(device=device, **config)
    elif name == "TwoStepNeRFRenderer":
        config = config.copy()
        for field in [
            # Required
            "void_model",
            "coarse_model",
            "fine_model",
            "volume",
            # Optional to use NeRF++
            "coarse_background_model",
            "fine_background_model",
            "outer_volume",
        ]:
            if field in config:
                config[field] = model_from_config(config.pop(field).copy(), device)
        return TwoStepNeRFRenderer(device=device, **config)
    elif name == "PooledMLP":
        return PooledMLP(device, **config)
    elif name == "PointDiffusionTransformer":
        return PointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "PointDiffusionPerceiver":
        return PointDiffusionPerceiver(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImagePointDiffusionTransformer":
        return CLIPImagePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImageGridPointDiffusionTransformer":
        return CLIPImageGridPointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "UpsamplePointDiffusionTransformer":
        return UpsamplePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImageGridUpsamplePointDiffusionTransformer":
        return CLIPImageGridUpsamplePointDiffusionTransformer(
            device=device, dtype=torch.float32, **config
        )
    elif name == "SplitVectorDiffusion":
        inner_config = config.pop("inner")
        d_latent = config.pop("d_latent")
        latent_ctx = config.pop("latent_ctx", 1)
        inner_config["input_channels"] = d_latent // latent_ctx
        inner_config["n_ctx"] = latent_ctx
        inner_config["output_channels"] = d_latent // latent_ctx * 2
        inner_model = model_from_config(inner_config, device)
        return SplitVectorDiffusion(
            device=device, wrapped=inner_model, n_ctx=latent_ctx, d_latent=d_latent
        )
    elif name == "STFRenderer":
        config = config.copy()
        for field in ["sdf", "tf", "volume"]:
            config[field] = model_from_config(config.pop(field), device)
        return STFRenderer(device=device, **config)
    elif name == "NeRSTFRenderer":
        config = config.copy()
        for field in ["sdf", "tf", "nerstf", "void", "volume"]:
            if field not in config:
                continue
            config[field] = model_from_config(config.pop(field), device)
        config.setdefault("sdf", None)
        config.setdefault("tf", None)
        config.setdefault("nerstf", None)
        return NeRSTFRenderer(device=device, **config)

    model_cls = {
        "MLPSDFModel": MLPSDFModel,
        "MLPTextureFieldModel": MLPTextureFieldModel,
        "MLPNeRFModel": MLPNeRFModel,
        "MLPDensitySDFModel": MLPDensitySDFModel,
        "MLPNeRSTFModel": MLPNeRSTFModel,
        "VoidNeRFModel": VoidNeRFModel,
        "BoundingBoxVolume": BoundingBoxVolume,
        "SphericalVolume": SphericalVolume,
        "UnboundedVolume": UnboundedVolume,
    }[name]
    return model_cls(device=device, **config)
