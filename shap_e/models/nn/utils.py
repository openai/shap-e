from typing import Iterable, Union

import numpy as np
import torch

ArrayType = Union[np.ndarray, Iterable[int], torch.Tensor]


def to_torch(arr: ArrayType, dtype=torch.float):
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(np.array(arr)).to(dtype)


def sample_pmf(pmf: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Sample from the given discrete probability distribution with replacement.

    The i-th bin is assumed to have mass pmf[i].

    :param pmf: [batch_size, *shape, n_samples, 1] where (pmf.sum(dim=-2) == 1).all()
    :param n_samples: number of samples

    :return: indices sampled with replacement
    """

    *shape, support_size, last_dim = pmf.shape
    assert last_dim == 1

    cdf = torch.cumsum(pmf.view(-1, support_size), dim=1)
    inds = torch.searchsorted(cdf, torch.rand(cdf.shape[0], n_samples, device=cdf.device))

    return inds.view(*shape, n_samples, 1).clamp(0, support_size - 1)


def safe_divide(a, b, epsilon=1e-6):
    return a / torch.where(b < 0, b - epsilon, b + epsilon)
