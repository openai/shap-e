"""
Adapted from: https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/download.py
"""

import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch
import yaml
from filelock import FileLock
from tqdm.auto import tqdm

MODEL_PATHS = {
    "transmitter": "https://openaipublic.azureedge.net/main/shap-e/transmitter.pt",
    "decoder": "https://openaipublic.azureedge.net/main/shap-e/vector_decoder.pt",
    "text300M": "https://openaipublic.azureedge.net/main/shap-e/text_cond.pt",
    "image300M": "https://openaipublic.azureedge.net/main/shap-e/image_cond.pt",
}

CONFIG_PATHS = {
    "transmitter": "https://openaipublic.azureedge.net/main/shap-e/transmitter_config.yaml",
    "decoder": "https://openaipublic.azureedge.net/main/shap-e/vector_decoder_config.yaml",
    "text300M": "https://openaipublic.azureedge.net/main/shap-e/text_cond_config.yaml",
    "image300M": "https://openaipublic.azureedge.net/main/shap-e/image_cond_config.yaml",
    "diffusion": "https://openaipublic.azureedge.net/main/shap-e/diffusion_config.yaml",
}


@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "shap_e_model_cache")


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.
    If cache_dir is specified, it will be used to download the files.
    Otherwise, default_cache_dir() is used.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(local_path):
        return local_path

    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


def load_config(
    config_name: str,
    progress: bool = False,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
):
    if config_name not in CONFIG_PATHS:
        raise ValueError(
            f"Unknown config name {config_name}. Known names are: {CONFIG_PATHS.keys()}."
        )
    path = fetch_file_cached(
        CONFIG_PATHS[config_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(
    checkpoint_name: str,
    device: torch.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    if checkpoint_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    return torch.load(path, map_location=device)


def load_model(
    model_name: str,
    device: torch.device,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    from .configs import model_from_config

    model = model_from_config(load_config(model_name, **kwargs), device=device)
    model.load_state_dict(load_checkpoint(model_name, device=device, **kwargs))
    model.eval()
    return model
