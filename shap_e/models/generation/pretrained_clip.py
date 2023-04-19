from typing import Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from shap_e.models.download import default_cache_dir

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]


class ImageCLIP(nn.Module):
    """
    A wrapper around a pre-trained CLIP model that automatically handles
    batches of texts, images, and embeddings.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = torch.float32,
        ensure_used_params: bool = True,
        clip_name: str = "ViT-L/14",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        assert clip_name in ["ViT-L/14", "ViT-B/32"]

        self.device = device
        self.ensure_used_params = ensure_used_params

        # Lazy import because of torchvision.
        import clip

        self.clip_model, self.preprocess = clip.load(
            clip_name, device=device, download_root=cache_dir or default_cache_dir()
        )
        self.clip_name = clip_name

        if dtype is not None:
            self.clip_model.to(dtype)
        self._tokenize = clip.tokenize

    @property
    def feature_dim(self) -> int:
        if self.clip_name == "ViT-L/14":
            return 768
        else:
            return 512

    @property
    def grid_size(self) -> int:
        if self.clip_name == "ViT-L/14":
            return 16
        else:
            return 7

    @property
    def grid_feature_dim(self) -> int:
        if self.clip_name == "ViT-L/14":
            return 1024
        else:
            return 768

    def forward(
        self,
        batch_size: int,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Generate a batch of embeddings from a mixture of images, texts,
        precomputed embeddings, and possibly empty values.

        For each batch element, at most one of images, texts, and embeddings
        should have a non-None value. Embeddings from multiple modalities
        cannot be mixed for a single batch element. If no modality is provided,
        a zero embedding will be used for the batch element.
        """
        image_seq = [None] * batch_size if images is None else list(images)
        text_seq = [None] * batch_size if texts is None else list(texts)
        embedding_seq = [None] * batch_size if embeddings is None else list(embeddings)
        assert len(image_seq) == batch_size, "number of images should match batch size"
        assert len(text_seq) == batch_size, "number of texts should match batch size"
        assert len(embedding_seq) == batch_size, "number of embeddings should match batch size"

        if self.ensure_used_params:
            return self._static_multimodal_embed(
                images=image_seq, texts=text_seq, embeddings=embedding_seq
            )

        result = torch.zeros((batch_size, self.feature_dim), device=self.device)
        index_images = []
        index_texts = []
        for i, (image, text, emb) in enumerate(zip(image_seq, text_seq, embedding_seq)):
            assert (
                sum([int(image is not None), int(text is not None), int(emb is not None)]) < 2
            ), "only one modality may be non-None per batch element"
            if image is not None:
                index_images.append((i, image))
            elif text is not None:
                index_texts.append((i, text))
            elif emb is not None:
                result[i] = emb.to(result)

        if len(index_images):
            embs = self.embed_images((img for _, img in index_images))
            for (i, _), emb in zip(index_images, embs):
                result[i] = emb.to(result)
        if len(index_texts):
            embs = self.embed_text((text for _, text in index_texts))
            for (i, _), emb in zip(index_texts, embs):
                result[i] = emb.to(result)

        return result

    def _static_multimodal_embed(
        self,
        images: List[Optional[ImageType]] = None,
        texts: List[Optional[str]] = None,
        embeddings: List[Optional[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Like forward(), but always runs all encoders to ensure that
        the forward graph looks the same on every rank.
        """
        image_emb = self.embed_images(images)
        text_emb = self.embed_text(t if t else "" for t in texts)
        joined_embs = torch.stack(
            [
                emb.to(device=self.device, dtype=torch.float32)
                if emb is not None
                else torch.zeros(self.feature_dim, device=self.device)
                for emb in embeddings
            ],
            dim=0,
        )

        image_flag = torch.tensor([x is not None for x in images], device=self.device)[
            :, None
        ].expand_as(image_emb)
        text_flag = torch.tensor([x is not None for x in texts], device=self.device)[
            :, None
        ].expand_as(image_emb)
        emb_flag = torch.tensor([x is not None for x in embeddings], device=self.device)[
            :, None
        ].expand_as(image_emb)

        return (
            image_flag.float() * image_emb
            + text_flag.float() * text_emb
            + emb_flag.float() * joined_embs
            + self.clip_model.logit_scale * 0  # avoid unused parameters
        )

    def embed_images(self, xs: Iterable[Optional[ImageType]]) -> torch.Tensor:
        """
        :param xs: N images, stored as numpy arrays, tensors, or PIL images.
        :return: an [N x D] tensor of features.
        """
        clip_inputs = self.images_to_tensor(xs)
        results = self.clip_model.encode_image(clip_inputs).float()
        return results / torch.linalg.norm(results, dim=-1, keepdim=True)

    def embed_text(self, prompts: Iterable[str]) -> torch.Tensor:
        """
        Embed text prompts as an [N x D] tensor.
        """
        enc = self.clip_model.encode_text(
            self._tokenize(list(prompts), truncate=True).to(self.device)
        ).float()
        return enc / torch.linalg.norm(enc, dim=-1, keepdim=True)

    def embed_images_grid(self, xs: Iterable[Optional[ImageType]]) -> torch.Tensor:
        """
        Embed images into latent grids.

        :param xs: an iterable of images to embed.
        :return: a tensor of shape [N x C x L], where L = self.grid_size**2.
        """
        if self.ensure_used_params:
            extra_value = 0.0
            for p in self.parameters():
                extra_value = extra_value + p.mean() * 0.0
        else:
            extra_value = 0.0

        x = self.images_to_tensor(xs).to(self.clip_model.dtype)

        # https://github.com/openai/CLIP/blob/4d120f3ec35b30bd0f992f5d8af2d793aad98d2a/clip/model.py#L225
        vt = self.clip_model.visual
        x = vt.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                vt.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + vt.positional_embedding.to(x.dtype)
        x = vt.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = vt.transformer(x)
        x = x.permute(1, 2, 0)  # LND -> NDL

        return x[..., 1:].contiguous().float() + extra_value

    def images_to_tensor(self, xs: Iterable[Optional[ImageType]]) -> torch.Tensor:
        return torch.stack([self.preprocess(_image_to_pil(x)) for x in xs], dim=0).to(self.device)


class FrozenImageCLIP:
    def __init__(self, device: torch.device, **kwargs):
        self.model = ImageCLIP(device, dtype=None, ensure_used_params=False, **kwargs)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.model.feature_dim

    @property
    def grid_size(self) -> int:
        return self.model.grid_size

    @property
    def grid_feature_dim(self) -> int:
        return self.model.grid_feature_dim

    def __call__(
        self,
        batch_size: int,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        # We don't do a no_grad() here so that gradients could still
        # flow to the input embeddings argument.
        # This behavior is currently not used, but it could be.
        return self.model(batch_size=batch_size, images=images, texts=texts, embeddings=embeddings)

    def embed_images(self, xs: Iterable[Optional[ImageType]]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.embed_images(xs)

    def embed_text(self, prompts: Iterable[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.embed_text(prompts)

    def embed_images_grid(self, xs: Iterable[Optional[ImageType]]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.embed_images_grid(xs)


def _image_to_pil(obj: Optional[ImageType]) -> Image.Image:
    if obj is None:
        return Image.fromarray(np.zeros([64, 64, 3], dtype=np.uint8))
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj.astype(np.uint8))
    elif isinstance(obj, torch.Tensor):
        return Image.fromarray(obj.detach().cpu().numpy().astype(np.uint8))
    else:
        return obj
