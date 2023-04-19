from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import torch

from shap_e.models.generation.perceiver import SimplePerceiver
from shap_e.models.generation.transformer import Transformer
from shap_e.models.nn.camera import DifferentiableProjectiveCamera
from shap_e.models.nn.encoding import (
    MultiviewPointCloudEmbedding,
    MultiviewPoseEmbedding,
    PosEmbLinear,
)
from shap_e.models.nn.ops import PointSetEmbedding
from shap_e.rendering.point_cloud import PointCloud
from shap_e.rendering.view_data import ProjectiveCamera
from shap_e.util.collections import AttrDict

from .base import ChannelsEncoder


class TransformerChannelsEncoder(ChannelsEncoder, ABC):
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
        d_latent: int = 512,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        latent_scale: float = 1.0,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            d_latent=d_latent,
            latent_bottleneck=latent_bottleneck,
            latent_warp=latent_warp,
        )
        self.width = width
        self.device = device
        self.dtype = dtype

        self.n_ctx = n_ctx

        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + self.latent_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.register_parameter(
            "output_tokens",
            nn.Parameter(torch.randn(self.latent_ctx, width, device=device, dtype=dtype)),
        )
        self.output_proj = nn.Linear(width, d_latent, device=device, dtype=dtype)
        self.latent_scale = latent_scale

    @abstractmethod
    def encode_input(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        pass

    def encode_to_channels(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        h = self.encode_input(batch, options=options)
        h = torch.cat([h, self.output_tokens[None].repeat(len(h), 1, 1)], dim=1)
        h = self.ln_pre(h)
        h = self.backbone(h)
        h = h[:, -self.latent_ctx :]
        h = self.ln_post(h)
        h = self.output_proj(h)
        return h


class PerceiverChannelsEncoder(ChannelsEncoder, ABC):
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
        min_unrolls: int,
        max_unrolls: int,
        d_latent: int = 512,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
        width: int = 512,
        layers: int = 12,
        xattn_layers: int = 1,
        heads: int = 8,
        init_scale: float = 0.25,
        # Training hparams
        inner_batch_size: Union[int, List[int]] = 1,
        data_ctx: int = 1,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            d_latent=d_latent,
            latent_bottleneck=latent_bottleneck,
            latent_warp=latent_warp,
        )
        self.width = width
        self.device = device
        self.dtype = dtype

        if isinstance(inner_batch_size, int):
            inner_batch_size = [inner_batch_size]
        self.inner_batch_size = inner_batch_size
        self.data_ctx = data_ctx
        self.min_unrolls = min_unrolls
        self.max_unrolls = max_unrolls

        encoder_fn = lambda inner_batch_size: SimplePerceiver(
            device=device,
            dtype=dtype,
            n_ctx=self.data_ctx + self.latent_ctx,
            n_data=inner_batch_size,
            width=width,
            layers=xattn_layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.encoder = (
            encoder_fn(self.inner_batch_size[0])
            if len(self.inner_batch_size) == 1
            else nn.ModuleList([encoder_fn(inner_bsz) for inner_bsz in self.inner_batch_size])
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
        self.output_proj = nn.Linear(width, d_latent, device=device, dtype=dtype)

    @abstractmethod
    def get_h_and_iterator(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> Tuple[torch.Tensor, Iterable[Union[torch.Tensor, Tuple]]]:
        """
        :return: a tuple of (
            the initial output tokens of size [batch_size, data_ctx + latent_ctx, width],
            an iterator over the given data
        )
        """

    def encode_to_channels(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        h, it = self.get_h_and_iterator(batch, options=options)
        n_unrolls = self.get_n_unrolls()

        for _ in range(n_unrolls):
            data = next(it)
            if isinstance(data, tuple):
                for data_i, encoder_i in zip(data, self.encoder):
                    h = encoder_i(h, data_i)
            else:
                h = self.encoder(h, data)
            h = self.processor(h)

        h = self.output_proj(self.ln_post(h[:, -self.latent_ctx :]))
        return h

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


@dataclass
class DatasetIterator:

    embs: torch.Tensor  # [batch_size, dataset_size, *shape]
    batch_size: int

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        _outer_batch_size, dataset_size, *_shape = self.embs.shape

        while True:
            start = self.idx
            self.idx += self.batch_size
            end = self.idx
            if end <= dataset_size:
                break
            self._reset()

        return self.embs[:, start:end]

    def _reset(self):
        self._shuffle()
        self.idx = 0  # pylint: disable=attribute-defined-outside-init

    def _shuffle(self):
        outer_batch_size, dataset_size, *shape = self.embs.shape
        idx = torch.stack(
            [
                torch.randperm(dataset_size, device=self.embs.device)
                for _ in range(outer_batch_size)
            ],
            dim=0,
        )
        idx = idx.view(outer_batch_size, dataset_size, *([1] * len(shape)))
        idx = torch.broadcast_to(idx, self.embs.shape)
        self.embs = torch.gather(self.embs, 1, idx)


class PointCloudTransformerChannelsEncoder(TransformerChannelsEncoder):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(
        self,
        *,
        input_channels: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.input_proj = nn.Linear(
            input_channels, self.width, device=self.device, dtype=self.dtype
        )

    def encode_input(self, batch: AttrDict, options: Optional[AttrDict] = None) -> torch.Tensor:
        _ = options
        points = batch.points
        h = self.input_proj(points.permute(0, 2, 1))  # NCL -> NLC
        return h


class PointCloudPerceiverChannelsEncoder(PerceiverChannelsEncoder):
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
        # point conv hyperparameters
        pointconv_radius: float = 0.5,
        pointconv_samples: int = 32,
        pointconv_hidden: Optional[List[int]] = None,
        pointconv_patch_size: int = 1,
        pointconv_stride: int = 1,
        pointconv_padding_mode: str = "zeros",
        use_pointconv: bool = False,
        # other hyperparameters
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert cross_attention_dataset in (
            "pcl",
            "multiview",
            "dense_pose_multiview",
            "multiview_pcl",
            "pcl_and_multiview_pcl",
            "incorrect_multiview_pcl",
            "pcl_and_incorrect_multiview_pcl",
        )
        assert fps_method in ("fps", "first")
        self.cross_attention_dataset = cross_attention_dataset
        self.fps_method = fps_method
        self.input_channels = input_channels
        self.input_proj = PosEmbLinear(
            pos_emb,
            input_channels,
            self.width,
            device=self.device,
            dtype=self.dtype,
        )
        self.use_pointconv = use_pointconv
        if use_pointconv:
            if pointconv_hidden is None:
                pointconv_hidden = [self.width]
            self.point_conv = PointSetEmbedding(
                n_point=self.data_ctx,
                radius=pointconv_radius,
                n_sample=pointconv_samples,
                d_input=self.input_proj.weight.shape[0],
                d_hidden=pointconv_hidden,
                patch_size=pointconv_patch_size,
                stride=pointconv_stride,
                padding_mode=pointconv_padding_mode,
                fps_method=fps_method,
                device=self.device,
                dtype=self.dtype,
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
        elif self.cross_attention_dataset == "dense_pose_multiview":
            # The number of output features is halved, because a patch_size of
            # 32 ends up with a large patch_emb weight.
            self.view_pose_width = self.width // 2
            self.image_size = image_size
            self.patch_size = patch_size
            self.use_depth = use_depth
            self.max_depth = max_depth
            self.mv_pose_embed = MultiviewPoseEmbedding(
                posemb_version="nerf",
                n_channels=4 if self.use_depth else 3,
                out_features=self.view_pose_width,
                device=self.device,
                dtype=self.dtype,
            )
            pos_ctx = (image_size // patch_size) ** 2
            # Positional embedding is unnecessary because pose information is baked into each pixel
            self.patch_emb = nn.Conv2d(
                in_channels=self.view_pose_width,
                out_channels=self.width,
                kernel_size=patch_size,
                stride=patch_size,
                device=self.device,
                dtype=self.dtype,
            )

        elif (
            self.cross_attention_dataset == "multiview_pcl"
            or self.cross_attention_dataset == "incorrect_multiview_pcl"
        ):
            self.view_pose_width = self.width // 2
            self.image_size = image_size
            self.patch_size = patch_size
            self.max_depth = max_depth
            assert use_depth
            self.mv_pcl_embed = MultiviewPointCloudEmbedding(
                posemb_version="nerf",
                n_channels=3,
                out_features=self.view_pose_width,
                device=self.device,
                dtype=self.dtype,
            )
            self.patch_emb = nn.Conv2d(
                in_channels=self.view_pose_width,
                out_channels=self.width,
                kernel_size=patch_size,
                stride=patch_size,
                device=self.device,
                dtype=self.dtype,
            )

        elif (
            self.cross_attention_dataset == "pcl_and_multiview_pcl"
            or self.cross_attention_dataset == "pcl_and_incorrect_multiview_pcl"
        ):
            self.view_pose_width = self.width // 2
            self.image_size = image_size
            self.patch_size = patch_size
            self.max_depth = max_depth
            assert use_depth
            self.mv_pcl_embed = MultiviewPointCloudEmbedding(
                posemb_version="nerf",
                n_channels=3,
                out_features=self.view_pose_width,
                device=self.device,
                dtype=self.dtype,
            )
            self.patch_emb = nn.Conv2d(
                in_channels=self.view_pose_width,
                out_channels=self.width,
                kernel_size=patch_size,
                stride=patch_size,
                device=self.device,
                dtype=self.dtype,
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
        if self.use_pointconv:
            points = self.input_proj(points).permute(0, 2, 1)  # NLC -> NCL
            xyz = batch.points[:, :3]
            data_tokens = self.point_conv(xyz, points).permute(0, 2, 1)  # NCL -> NLC
        else:
            fps_samples = self.sample_pcl_fps(points)
            data_tokens = self.input_proj(fps_samples)
        batch_size = points.shape[0]
        latent_tokens = self.output_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.data_ctx + self.latent_ctx, self.width)

        # Build the dataset embedding iterator
        dataset_fn = {
            "pcl": self.get_pcl_dataset,
            "multiview": self.get_multiview_dataset,
            "dense_pose_multiview": self.get_dense_pose_multiview_dataset,
            "pcl_and_multiview_pcl": self.get_pcl_and_multiview_pcl_dataset,
            "multiview_pcl": self.get_multiview_pcl_dataset,
        }[self.cross_attention_dataset]
        it = dataset_fn(batch, options=options)

        return h, it

    def sample_pcl_fps(self, points: torch.Tensor) -> torch.Tensor:
        return sample_pcl_fps(points, data_ctx=self.data_ctx, method=self.fps_method)

    def get_pcl_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict[str, Any]] = None,
        inner_batch_size: Optional[int] = None,
    ) -> Iterable:
        _ = options
        if inner_batch_size is None:
            inner_batch_size = self.inner_batch_size[0]
        points = batch.points.permute(0, 2, 1)  # NCL -> NLC
        dataset_emb = self.input_proj(points)
        assert dataset_emb.shape[1] >= inner_batch_size
        return iter(DatasetIterator(dataset_emb, batch_size=inner_batch_size))

    def get_multiview_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict] = None,
        inner_batch_size: Optional[int] = None,
    ) -> Iterable:
        _ = options

        if inner_batch_size is None:
            inner_batch_size = self.inner_batch_size[0]

        dataset_emb = self.encode_views(batch)
        batch_size, num_views, n_patches, width = dataset_emb.shape

        assert num_views >= inner_batch_size

        it = iter(DatasetIterator(dataset_emb, batch_size=inner_batch_size))

        def gen():
            while True:
                examples = next(it)
                assert examples.shape == (batch_size, self.inner_batch_size, n_patches, self.width)
                views = examples.reshape(batch_size, -1, width) + self.pos_emb
                yield views

        return gen()

    def get_dense_pose_multiview_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict] = None,
        inner_batch_size: Optional[int] = None,
    ) -> Iterable:
        _ = options

        if inner_batch_size is None:
            inner_batch_size = self.inner_batch_size[0]

        dataset_emb = self.encode_dense_pose_views(batch)
        batch_size, num_views, n_patches, width = dataset_emb.shape

        assert num_views >= inner_batch_size

        it = iter(DatasetIterator(dataset_emb, batch_size=inner_batch_size))

        def gen():
            while True:
                examples = next(it)
                assert examples.shape == (batch_size, inner_batch_size, n_patches, self.width)
                views = examples.reshape(batch_size, -1, width)
                yield views

        return gen()

    def get_pcl_and_multiview_pcl_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict] = None,
        use_distance: bool = True,
    ) -> Iterable:
        _ = options

        pcl_it = self.get_pcl_dataset(
            batch, options=options, inner_batch_size=self.inner_batch_size[0]
        )
        multiview_pcl_emb = self.encode_multiview_pcl(batch, use_distance=use_distance)
        batch_size, num_views, n_patches, width = multiview_pcl_emb.shape

        assert num_views >= self.inner_batch_size[1]

        multiview_pcl_it = iter(
            DatasetIterator(multiview_pcl_emb, batch_size=self.inner_batch_size[1])
        )

        def gen():
            while True:
                pcl = next(pcl_it)
                multiview_pcl = next(multiview_pcl_it)
                assert multiview_pcl.shape == (
                    batch_size,
                    self.inner_batch_size[1],
                    n_patches,
                    self.width,
                )
                yield pcl, multiview_pcl.reshape(batch_size, -1, width)

        return gen()

    def get_multiview_pcl_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict] = None,
        inner_batch_size: Optional[int] = None,
        use_distance: bool = True,
    ) -> Iterable:
        _ = options

        if inner_batch_size is None:
            inner_batch_size = self.inner_batch_size[0]

        multiview_pcl_emb = self.encode_multiview_pcl(batch, use_distance=use_distance)
        batch_size, num_views, n_patches, width = multiview_pcl_emb.shape

        assert num_views >= inner_batch_size

        multiview_pcl_it = iter(DatasetIterator(multiview_pcl_emb, batch_size=inner_batch_size))

        def gen():
            while True:
                multiview_pcl = next(multiview_pcl_it)
                assert multiview_pcl.shape == (
                    batch_size,
                    inner_batch_size,
                    n_patches,
                    self.width,
                )
                yield multiview_pcl.reshape(batch_size, -1, width)

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

    def encode_dense_pose_views(self, batch: AttrDict) -> torch.Tensor:
        """
        :return: [batch_size, num_views, n_patches, width]
        """
        all_views = self.views_to_tensor(batch.views).to(self.device)
        if self.use_depth:
            depths = self.depths_to_tensor(batch.depths)
            all_views = torch.cat([all_views, depths], dim=2)

        dense_poses, _ = self.dense_pose_cameras_to_tensor(batch.cameras)
        dense_poses = dense_poses.permute(0, 1, 4, 5, 2, 3)
        position, direction = dense_poses[:, :, 0], dense_poses[:, :, 1]
        all_view_poses = self.mv_pose_embed(all_views, position, direction)

        batch_size, num_views, _, _, _ = all_view_poses.shape

        views_proj = self.patch_emb(
            all_view_poses.reshape([batch_size * num_views, *all_view_poses.shape[2:]])
        )
        views_proj = (
            views_proj.reshape([batch_size, num_views, self.width, -1])
            .permute(0, 1, 3, 2)
            .contiguous()
        )  # [batch_size x num_views x n_patches x width]

        return views_proj

    def encode_multiview_pcl(self, batch: AttrDict, use_distance: bool = True) -> torch.Tensor:
        """
        :return: [batch_size, num_views, n_patches, width]
        """
        all_views = self.views_to_tensor(batch.views).to(self.device)
        depths = self.raw_depths_to_tensor(batch.depths)
        all_view_alphas = self.view_alphas_to_tensor(batch.view_alphas).to(self.device)
        mask = all_view_alphas >= 0.999

        dense_poses, camera_z = self.dense_pose_cameras_to_tensor(batch.cameras)
        dense_poses = dense_poses.permute(0, 1, 4, 5, 2, 3)

        origin, direction = dense_poses[:, :, 0], dense_poses[:, :, 1]
        if use_distance:
            ray_depth_factor = torch.sum(direction * camera_z[..., None, None], dim=2, keepdim=True)
            depths = depths / ray_depth_factor
        position = origin + depths * direction
        all_view_poses = self.mv_pcl_embed(all_views, origin, position, mask)

        batch_size, num_views, _, _, _ = all_view_poses.shape

        views_proj = self.patch_emb(
            all_view_poses.reshape([batch_size * num_views, *all_view_poses.shape[2:]])
        )
        views_proj = (
            views_proj.reshape([batch_size, num_views, self.width, -1])
            .permute(0, 1, 3, 2)
            .contiguous()
        )  # [batch_size x num_views x n_patches x width]

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

    def view_alphas_to_tensor(
        self, view_alphas: Union[torch.Tensor, List[List[Image.Image]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 1 x size x size] tensor in the range [0, 1].
        """
        if isinstance(view_alphas, torch.Tensor):
            return view_alphas

        tensor_batch = []
        num_views = len(view_alphas[0])
        for inner_list in view_alphas:
            assert len(inner_list) == num_views
            inner_batch = []
            for img in inner_list:
                tensor = (
                    torch.from_numpy(np.array(img)).to(device=self.device, dtype=torch.float32)
                    / 255.0
                )
                tensor = F.interpolate(
                    tensor[None, None],
                    (self.image_size,) * 2,
                    mode="nearest",
                )
                inner_batch.append(tensor)
            tensor_batch.append(torch.cat(inner_batch, dim=0))
        return torch.stack(tensor_batch, dim=0)

    def raw_depths_to_tensor(
        self, depths: Union[torch.Tensor, List[List[Image.Image]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 1 x size x size] tensor
        """
        if isinstance(depths, torch.Tensor):
            return depths

        tensor_batch = []
        num_views = len(depths[0])
        for inner_list in depths:
            assert len(inner_list) == num_views
            inner_batch = []
            for arr in inner_list:
                tensor = torch.from_numpy(arr).clamp(max=self.max_depth)
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

    def dense_pose_cameras_to_tensor(
        self, cameras: Union[torch.Tensor, List[List[ProjectiveCamera]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (rays, z_directions) where
            - rays: [batch, num_views, height, width, 2, 3] tensor of camera information.
            - z_directions: [batch, num_views, 3] tensor of camera z directions.
        """
        if isinstance(cameras, torch.Tensor):
            raise NotImplementedError

        for inner_list in cameras:
            assert len(inner_list) == len(cameras[0])

        camera = cameras[0][0]
        flat_camera = DifferentiableProjectiveCamera(
            origin=torch.from_numpy(
                np.stack(
                    [cam.origin for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            x=torch.from_numpy(
                np.stack(
                    [cam.x for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            y=torch.from_numpy(
                np.stack(
                    [cam.y for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            z=torch.from_numpy(
                np.stack(
                    [cam.z for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            width=camera.width,
            height=camera.height,
            x_fov=camera.x_fov,
            y_fov=camera.y_fov,
        )
        batch_size = len(cameras) * len(cameras[0])
        coords = (
            flat_camera.image_coords()
            .to(flat_camera.origin.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        rays = flat_camera.camera_rays(coords)
        return (
            rays.view(len(cameras), len(cameras[0]), camera.height, camera.width, 2, 3).to(
                self.device
            ),
            flat_camera.z.view(len(cameras), len(cameras[0]), 3).to(self.device),
        )


def sample_pcl_fps(points: torch.Tensor, data_ctx: int, method: str = "fps") -> torch.Tensor:
    """
    Run farthest-point sampling on a batch of point clouds.

    :param points: batch of shape [N x num_points].
    :param data_ctx: subsample count.
    :param method: either 'fps' or 'first'. Using 'first' assumes that the
                   points are already sorted according to FPS sampling.
    :return: batch of shape [N x min(num_points, data_ctx)].
    """
    n_points = points.shape[1]
    if n_points == data_ctx:
        return points
    if method == "first":
        return points[:, :data_ctx]
    elif method == "fps":
        batch = points.cpu().split(1, dim=0)
        fps = [sample_fps(x, n_samples=data_ctx) for x in batch]
        return torch.cat(fps, dim=0).to(points.device)
    else:
        raise ValueError(f"unsupported farthest-point sampling method: {method}")


def sample_fps(example: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    :param example: [1, n_points, 3 + n_channels]
    :return: [1, n_samples, 3 + n_channels]
    """
    points = example.cpu().squeeze(0).numpy()
    coords, raw_channels = points[:, :3], points[:, 3:]
    n_points, n_channels = raw_channels.shape
    assert n_samples <= n_points
    channels = {str(idx): raw_channels[:, idx] for idx in range(n_channels)}
    max_points = min(32768, n_points)
    fps_pcl = (
        PointCloud(coords=coords, channels=channels)
        .random_sample(max_points)
        .farthest_point_sample(n_samples)
    )
    fps_channels = np.stack([fps_pcl.channels[str(idx)] for idx in range(n_channels)], axis=1)
    fps = np.concatenate([fps_pcl.coords, fps_channels], axis=1)
    fps = torch.from_numpy(fps).unsqueeze(0)
    assert fps.shape == (1, n_samples, 3 + n_channels)
    return fps
