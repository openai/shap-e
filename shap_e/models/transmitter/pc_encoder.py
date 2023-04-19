from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import torch

from shap_e.models.generation.perceiver import SimplePerceiver
from shap_e.models.generation.transformer import Transformer
from shap_e.models.nn.encoding import PosEmbLinear
from shap_e.rendering.view_data import ProjectiveCamera
from shap_e.util.collections import AttrDict

from .base import VectorEncoder
from .channels_encoder import DatasetIterator, sample_pcl_fps


class PointCloudTransformerEncoder(VectorEncoder):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
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
        input_channels: int = 6,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        pos_emb: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            latent_bottleneck=latent_bottleneck,
            d_latent=d_latent,
        )
        self.input_channels = input_channels
        self.n_ctx = n_ctx
        self.latent_ctx = latent_ctx

        assert d_latent % latent_ctx == 0

        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + latent_ctx,
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

        self.input_proj = PosEmbLinear(pos_emb, input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, d_latent // latent_ctx, device=device, dtype=dtype)

    def encode_to_vector(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        _ = options
        points = batch.points.permute(0, 2, 1)  # NCL -> NLC
        h = self.input_proj(points)
        h = torch.cat([h, self.output_tokens[None].repeat(len(h), 1, 1)], dim=1)
        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        h = h[:, self.n_ctx :]
        h = self.output_proj(h).flatten(1)
        return h


class PerceiverEncoder(VectorEncoder):
    """
    Encode point clouds using a perceiver model with an extra output
    token used to extract a latent vector.
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
        width: int = 512,
        layers: int = 12,
        xattn_layers: int = 1,
        heads: int = 8,
        init_scale: float = 0.25,
        # Training hparams
        inner_batch_size: int = 1,
        data_ctx: int = 1,
        min_unrolls: int,
        max_unrolls: int,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            latent_bottleneck=latent_bottleneck,
            d_latent=d_latent,
        )
        self.width = width
        self.device = device
        self.dtype = dtype
        self.latent_ctx = latent_ctx

        self.inner_batch_size = inner_batch_size
        self.data_ctx = data_ctx
        self.min_unrolls = min_unrolls
        self.max_unrolls = max_unrolls

        self.encoder = SimplePerceiver(
            device=device,
            dtype=dtype,
            n_ctx=self.data_ctx + self.latent_ctx,
            n_data=self.inner_batch_size,
            width=width,
            layers=xattn_layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.processor = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=self.data_ctx + self.latent_ctx,
            layers=layers - xattn_layers,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.register_parameter(
            "output_tokens",
            nn.Parameter(torch.randn(self.latent_ctx, width, device=device, dtype=dtype)),
        )
        self.output_proj = nn.Linear(width, d_latent // self.latent_ctx, device=device, dtype=dtype)

    @abstractmethod
    def get_h_and_iterator(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> Tuple[torch.Tensor, Iterable]:
        """
        :return: a tuple of (
            the initial output tokens of size [batch_size, data_ctx + latent_ctx, width],
            an iterator over the given data
        )
        """

    def encode_to_vector(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        h, it = self.get_h_and_iterator(batch, options=options)
        n_unrolls = self.get_n_unrolls()

        for _ in range(n_unrolls):
            data = next(it)
            h = self.encoder(h, data)
            h = self.processor(h)

        h = self.output_proj(self.ln_post(h[:, -self.latent_ctx :]))
        return h.flatten(1)

    def get_n_unrolls(self):
        if self.training:
            n_unrolls = torch.randint(
                self.min_unrolls, self.max_unrolls + 1, size=(), device=self.device
            )
            dist.broadcast(n_unrolls, 0)
            n_unrolls = n_unrolls.item()
        else:
            n_unrolls = self.max_unrolls
        return n_unrolls


class PointCloudPerceiverEncoder(PerceiverEncoder):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(
        self,
        *,
        cross_attention_dataset: str = "pcl",
        fps_method: str = "fps",
        # point cloud hyperparameters
        input_channels: int = 6,
        pos_emb: Optional[str] = None,
        # multiview hyperparameters
        image_size: int = 256,
        patch_size: int = 32,
        pose_dropout: float = 0.0,
        use_depth: bool = False,
        max_depth: float = 5.0,
        # other hyperparameters
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert cross_attention_dataset in ("pcl", "multiview")
        assert fps_method in ("fps", "first")
        self.cross_attention_dataset = cross_attention_dataset
        self.fps_method = fps_method
        self.input_channels = input_channels
        self.input_proj = PosEmbLinear(
            pos_emb, input_channels, self.width, device=self.device, dtype=self.dtype
        )
        if self.cross_attention_dataset == "multiview":
            self.image_size = image_size
            self.patch_size = patch_size
            self.pose_dropout = pose_dropout
            self.use_depth = use_depth
            self.max_depth = max_depth
            pos_ctx = (image_size // patch_size) ** 2
            self.register_parameter(
                "pos_emb",
                nn.Parameter(
                    torch.randn(
                        pos_ctx * self.inner_batch_size,
                        self.width,
                        device=self.device,
                        dtype=self.dtype,
                    )
                ),
            )
            self.patch_emb = nn.Conv2d(
                in_channels=3 if not use_depth else 4,
                out_channels=self.width,
                kernel_size=patch_size,
                stride=patch_size,
                device=self.device,
                dtype=self.dtype,
            )
            self.camera_emb = nn.Sequential(
                nn.Linear(
                    3 * 4 + 1, self.width, device=self.device, dtype=self.dtype
                ),  # input size is for origin+x+y+z+fov
                nn.GELU(),
                nn.Linear(self.width, 2 * self.width, device=self.device, dtype=self.dtype),
            )

    def get_h_and_iterator(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> Tuple[torch.Tensor, Iterable]:
        """
        :return: a tuple of (
            the initial output tokens of size [batch_size, data_ctx + latent_ctx, width],
            an iterator over the given data
        )
        """
        options = AttrDict() if options is None else options

        # Build the initial query embeddings
        points = batch.points.permute(0, 2, 1)  # NCL -> NLC
        fps_samples = self.sample_pcl_fps(points)
        batch_size = points.shape[0]
        data_tokens = self.input_proj(fps_samples)
        latent_tokens = self.output_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.data_ctx + self.latent_ctx, self.width)

        # Build the dataset embedding iterator
        dataset_fn = {
            "pcl": self.get_pcl_dataset,
            "multiview": self.get_multiview_dataset,
        }[self.cross_attention_dataset]
        it = dataset_fn(batch, options=options)

        return h, it

    def sample_pcl_fps(self, points: torch.Tensor) -> torch.Tensor:
        return sample_pcl_fps(points, data_ctx=self.data_ctx, method=self.fps_method)

    def get_pcl_dataset(
        self, batch: AttrDict, options: Optional[AttrDict[str, Any]] = None
    ) -> Iterable:
        _ = options
        dataset_emb = self.input_proj(batch.points.permute(0, 2, 1))  # NCL -> NLC
        assert dataset_emb.shape[1] >= self.inner_batch_size
        return iter(DatasetIterator(dataset_emb, batch_size=self.inner_batch_size))

    def get_multiview_dataset(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> Iterable:
        _ = options

        dataset_emb = self.encode_views(batch)
        batch_size, num_views, n_patches, width = dataset_emb.shape

        assert num_views >= self.inner_batch_size

        it = iter(DatasetIterator(dataset_emb, batch_size=self.inner_batch_size))

        def gen():
            while True:
                examples = next(it)
                assert examples.shape == (batch_size, self.inner_batch_size, n_patches, self.width)
                views = examples.reshape(batch_size, -1, width) + self.pos_emb
                yield views

        return gen()

    def encode_views(self, batch: AttrDict) -> torch.Tensor:
        """
        :return: [batch_size, num_views, n_patches, width]
        """
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

        # [batch_size, num_views, 1, 2 * width]
        camera_proj = self.camera_emb(all_cameras).reshape(
            [batch_size, num_views, 1, self.width * 2]
        )
        pose_dropout = self.pose_dropout if self.training else 0.0
        mask = torch.rand(batch_size, 1, 1, 1, device=views_proj.device) >= pose_dropout
        camera_proj = torch.where(mask, camera_proj, torch.zeros_like(camera_proj))
        scale, shift = camera_proj.chunk(2, dim=3)
        views_proj = views_proj * (scale + 1.0) + shift
        return views_proj

    def views_to_tensor(self, views: Union[torch.Tensor, List[List[Image.Image]]]) -> torch.Tensor:
        """
        Returns a [batch x num_views x 3 x size x size] tensor in the range [-1, 1].
        """
        if isinstance(views, torch.Tensor):
            return views

        tensor_batch = []
        num_views = len(views[0])
        for inner_list in views:
            assert len(inner_list) == num_views
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
        num_views = len(depths[0])
        for inner_list in depths:
            assert len(inner_list) == num_views
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
