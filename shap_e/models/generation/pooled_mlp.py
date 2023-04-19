import torch
import torch.nn as nn

from .util import timestep_embedding


class PooledMLP(nn.Module):
    def __init__(
        self,
        device: torch.device,
        *,
        input_channels: int = 3,
        output_channels: int = 6,
        hidden_size: int = 256,
        resblocks: int = 4,
        pool_op: str = "max",
    ):
        super().__init__()
        self.input_embed = nn.Conv1d(input_channels, hidden_size, kernel_size=1, device=device)
        self.time_embed = nn.Linear(hidden_size, hidden_size, device=device)

        blocks = []
        for _ in range(resblocks):
            blocks.append(ResBlock(hidden_size, pool_op, device=device))
        self.sequence = nn.Sequential(*blocks)

        self.out = nn.Conv1d(hidden_size, output_channels, kernel_size=1, device=device)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.weight.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        in_embed = self.input_embed(x)
        t_embed = self.time_embed(timestep_embedding(t, in_embed.shape[1]))
        h = in_embed + t_embed[..., None]
        h = self.sequence(h)
        h = self.out(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int, pool_op: str, device: torch.device):
        super().__init__()
        assert pool_op in ["mean", "max"]
        self.pool_op = pool_op
        self.body = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm((hidden_size,), device=device),
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.SiLU(),
            nn.LayerNorm((hidden_size,), device=device),
            nn.Linear(hidden_size, hidden_size, device=device),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        N, C, T = x.shape
        out = self.body(x.permute(0, 2, 1).reshape(N * T, C)).reshape([N, T, C]).permute(0, 2, 1)
        pooled = pool(self.pool_op, x)
        gate = self.gate(pooled)
        return x + out * gate[..., None]


def pool(op_name: str, x: torch.Tensor) -> torch.Tensor:
    if op_name == "max":
        pooled, _ = torch.max(x, dim=-1)
    elif op_name == "mean":
        pooled, _ = torch.mean(x, dim=-1)
    else:
        raise ValueError(f"unknown pool op: {op_name}")
    return pooled
