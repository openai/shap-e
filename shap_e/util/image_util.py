import random
from typing import Any, List, Optional, Union

import blobfile as bf
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def center_crop(
    img: Union[Image.Image, torch.Tensor, np.ndarray]
) -> Union[Image.Image, torch.Tensor, np.ndarray]:
    """
    Center crops an image.
    """
    if isinstance(img, (np.ndarray, torch.Tensor)):
        height, width = img.shape[:2]
    else:
        width, height = img.size
    size = min(width, height)
    left, top = (width - size) // 2, (height - size) // 2
    right, bottom = left + size, top + size
    if isinstance(img, (np.ndarray, torch.Tensor)):
        img = img[top:bottom, left:right]
    else:
        img = img.crop((left, top, right, bottom))
    return img


def resize(
    img: Union[Image.Image, torch.Tensor, np.ndarray],
    *,
    height: int,
    width: int,
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> Union[Image.Image, torch.Tensor, np.ndarray]:
    """
    :param: img: image in HWC order
    :return: currently written for downsampling
    """

    orig, cls = img, type(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    dtype = img.dtype
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    ndim = img.ndim
    if img.ndim == 2:
        img = img.unsqueeze(-1)

    if min_value is None and max_value is None:
        # .clamp throws an error when both are None
        min_value = -np.inf

    img = img.permute(2, 0, 1)
    size = (height, width)
    img = (
        F.interpolate(img[None].float(), size=size, mode="area")[0]
        .clamp(min_value, max_value)
        .to(img.dtype)
        .permute(1, 2, 0)
    )

    if ndim < img.ndim:
        img = img.squeeze(-1)
    if not isinstance(orig, torch.Tensor):
        img = img.numpy()
    img = img.astype(dtype)
    if isinstance(orig, Image.Image):
        img = Image.fromarray(img)

    return img


def get_alpha(img: Image.Image) -> Image.Image:
    """
    :return: the alpha channel separated out as a grayscale image
    """
    img_arr = np.asarray(img)
    if img_arr.shape[2] == 4:
        alpha = img_arr[:, :, 3]
    else:
        alpha = np.full(img_arr.shape[:2], 255, dtype=np.uint8)
    alpha = Image.fromarray(alpha)
    return alpha


def remove_alpha(img: Image.Image, mode: str = "random") -> Image.Image:
    """
    No op if the image doesn't have an alpha channel.

    :param: mode: Defaults to "random" but has an option to use a "black" or
        "white" background

    :return: image with alpha removed
    """
    img_arr = np.asarray(img)
    if img_arr.shape[2] == 4:
        # Add bg to get rid of alpha channel
        if mode == "random":
            height, width = img_arr.shape[:2]
            bg = Image.fromarray(
                random.choice([_black_bg, _gray_bg, _checker_bg, _noise_bg])(height, width)
            )
            bg.paste(img, mask=img)
            img = bg
        elif mode == "black" or mode == "white":
            img_arr = img_arr.astype(float)
            rgb, alpha = img_arr[:, :, :3], img_arr[:, :, -1:] / 255
            background = np.zeros((1, 1, 3)) if mode == "black" else np.full((1, 1, 3), 255)
            rgb = rgb * alpha + background * (1 - alpha)
            img = Image.fromarray(np.round(rgb).astype(np.uint8))
    return img


def _black_bg(h: int, w: int) -> np.ndarray:
    return np.zeros([h, w, 3], dtype=np.uint8)


def _gray_bg(h: int, w: int) -> np.ndarray:
    return (np.zeros([h, w, 3]) + np.random.randint(low=0, high=256)).astype(np.uint8)


def _checker_bg(h: int, w: int) -> np.ndarray:
    checker_size = np.ceil(np.exp(np.random.uniform() * np.log(min(h, w))))
    c1 = np.random.randint(low=0, high=256)
    c2 = np.random.randint(low=0, high=256)

    xs = np.arange(w)[None, :, None] + np.random.randint(low=0, high=checker_size + 1)
    ys = np.arange(h)[:, None, None] + np.random.randint(low=0, high=checker_size + 1)

    fields = np.logical_xor((xs // checker_size) % 2 == 0, (ys // checker_size) % 2 == 0)
    return np.where(fields, np.array([c1] * 3), np.array([c2] * 3)).astype(np.uint8)


def _noise_bg(h: int, w: int) -> np.ndarray:
    return np.random.randint(low=0, high=256, size=[h, w, 3]).astype(np.uint8)


def load_image(image_path: str) -> Image.Image:
    with bf.BlobFile(image_path, "rb") as thefile:
        img = Image.open(thefile)
        img.load()
    return img


def make_tile(images: List[Union[np.ndarray, Image.Image]], columns=8) -> Image.Image:
    """
    to test, run
        >>> display(make_tile([(np.zeros((128, 128, 3)) + c).astype(np.uint8) for c in np.linspace(0, 255, 15)]))
    """
    images = list(map(np.array, images))
    size = images[0].shape[0]
    n = round_up(len(images), columns)
    n_blanks = n - len(images)
    images.extend([np.zeros((size, size, 3), dtype=np.uint8)] * n_blanks)
    images = (
        np.array(images)
        .reshape(n // columns, columns, size, size, 3)
        .transpose([0, 2, 1, 3, 4])
        .reshape(n // columns * size, columns * size, 3)
    )
    return Image.fromarray(images)


def round_up(n: int, b: int) -> int:
    return (n + b - 1) // b * b
