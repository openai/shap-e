from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class Query:
    # Both of these are of shape [batch_size x ... x 3]
    position: torch.Tensor
    direction: Optional[torch.Tensor] = None

    t_min: Optional[torch.Tensor] = None
    t_max: Optional[torch.Tensor] = None

    def copy(self) -> "Query":
        return Query(
            position=self.position,
            direction=self.direction,
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def map_tensors(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Query":
        return Query(
            position=f(self.position),
            direction=f(self.direction) if self.direction is not None else None,
            t_min=f(self.t_min) if self.t_min is not None else None,
            t_max=f(self.t_max) if self.t_max is not None else None,
        )
