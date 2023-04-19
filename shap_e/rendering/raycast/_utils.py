import torch


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def cross_product(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            v1[..., 1] * v2[..., 2] - v2[..., 1] * v1[..., 2],
            -(v1[..., 0] * v2[..., 2] - v2[..., 0] * v1[..., 2]),
            v1[..., 0] * v2[..., 1] - v2[..., 0] * v1[..., 1],
        ],
        dim=-1,
    )
