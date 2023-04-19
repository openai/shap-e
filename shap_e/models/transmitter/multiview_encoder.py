from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from shap_e.models.generation.transformer import Transformer
from shap_e.rendering.view_data import ProjectiveCamera
from shap_e.util.collections import AttrDict

from .base import VectorEncoder


class MultiviewTransformerEncoder(VectorEncoder):
    """
    Encode cameras and views using a transformer model with extra output
    token(s) used to extract a latent vector.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        d_latent: int = 512,
        latent_ctx: int = 1,
        num_views: int = 20,
        image_size: int = 256,
        patch_size: int = 32,
        use_depth: bool = False,
        max_depth: float = 5.0,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        pos_emb_init_scale: float = 1.0,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            latent_bottleneck=latent_bottleneck,
            d_latent=d_latent,
        )
        self.num_views = num_views
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_depth = use_depth
        self.max_depth = max_depth
        self.n_ctx = num_views * (1 + (image_size // patch_size) ** 2)
        self.latent_ctx = latent_ctx
        self.width = width

        assert d_latent % latent_ctx == 0

        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=self.n_ctx + latent_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.register_parameter(
            "output_tokens",
            nn.Parameter(torch.randn(latent_ctx, width, device=device, dtype=dtype)),
        )
        self.register_parameter(
            "pos_emb",
            nn.Parameter(
                pos_emb_init_scale * torch.randn(self.n_ctx, width, device=device, dtype=dtype)
            ),
        )
        self.patch_emb = nn.Conv2d(
            in_channels=3 if not use_depth else 4,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            device=device,
            dtype=dtype,
        )
        self.camera_emb = nn.Sequential(
            nn.Linear(
                3 * 4 + 1, width, device=device, dtype=dtype
            ),  # input size is for origin+x+y+z+fov
            nn.GELU(),
            nn.Linear(width, width, device=device, dtype=dtype),
        )
        self.output_proj = nn.Linear(width, d_latent // latent_ctx, device=device, dtype=dtype)

    def encode_to_vector(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        _ = options

        all_views = self.views_to_tensor(batch.views).to(self.device)
        if self.use_depth:
            all_views = torch.cat([all_views, self.depths_to_tensor(batch.depths)], dim=2)
        all_cameras = self.cameras_to_tensor(batch.cameras).to(self.device)

        batch_size, num_views, _, _, _ = all_views.shape

        views_proj = self.patch_emb(
            all_views.reshape([batch_size * num_views, *all_views.shape[2:]])
        )
        views_proj = (
            views_proj.reshape([batch_size, num_views, self.width, -1])
            .permute(0, 1, 3, 2)
            .contiguous()
        )  # [batch_size x num_views x n_patches x width]

        cameras_proj = self.camera_emb(all_cameras).reshape([batch_size, num_views, 1, self.width])

        h = torch.cat([views_proj, cameras_proj], dim=2).reshape([batch_size, -1, self.width])
        h = h + self.pos_emb
        h = torch.cat([h, self.output_tokens[None].repeat(len(h), 1, 1)], dim=1)
        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        h = h[:, self.n_ctx :]
        h = self.output_proj(h).flatten(1)

        return h

    def views_to_tensor(self, views: Union[torch.Tensor, List[List[Image.Image]]]) -> torch.Tensor:
        """
        Returns a [batch x num_views x 3 x size x size] tensor in the range [-1, 1].
        """
        if isinstance(views, torch.Tensor):
            return views

        tensor_batch = []
        for inner_list in views:
            assert len(inner_list) == self.num_views
            inner_batch = []
            for img in inner_list:
                img = img.resize((self.image_size,) * 2).convert("RGB")
                inner_batch.append(
                    torch.from_numpy(np.array(img)).to(device=self.device, dtype=torch.float32)
                    / 127.5
                    - 1
                )
            tensor_batch.append(torch.stack(inner_batch, dim=0))
        return torch.stack(tensor_batch, dim=0).permute(0, 1, 4, 2, 3)

    def depths_to_tensor(
        self, depths: Union[torch.Tensor, List[List[Image.Image]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 1 x size x size] tensor in the range [-1, 1].
        """
        if isinstance(depths, torch.Tensor):
            return depths

        tensor_batch = []
        for inner_list in depths:
            assert len(inner_list) == self.num_views
            inner_batch = []
            for arr in inner_list:
                tensor = torch.from_numpy(arr).clamp(max=self.max_depth) / self.max_depth
                tensor = tensor * 2 - 1
                tensor = F.interpolate(
                    tensor[None, None],
                    (self.image_size,) * 2,
                    mode="nearest",
                )
                inner_batch.append(tensor.to(device=self.device, dtype=torch.float32))
            tensor_batch.append(torch.cat(inner_batch, dim=0))
        return torch.stack(tensor_batch, dim=0)

    def cameras_to_tensor(
        self, cameras: Union[torch.Tensor, List[List[ProjectiveCamera]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 3*4+1] tensor of camera information.
        """
        if isinstance(cameras, torch.Tensor):
            return cameras
        outer_batch = []
        for inner_list in cameras:
            inner_batch = []
            for camera in inner_list:
                inner_batch.append(
                    np.array(
                        [
                            *camera.x,
                            *camera.y,
                            *camera.z,
                            *camera.origin,
                            camera.x_fov,
                        ]
                    )
                )
            outer_batch.append(np.stack(inner_batch, axis=0))
        return torch.from_numpy(np.stack(outer_batch, axis=0)).float()
