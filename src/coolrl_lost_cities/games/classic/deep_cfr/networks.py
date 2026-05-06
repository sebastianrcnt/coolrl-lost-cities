from __future__ import annotations

import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.config import NetworkConfig


def _activation(name: str) -> nn.Module:
    token = name.lower()
    if token == "relu":
        return nn.ReLU()
    if token == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name!r}")


class DeepCFRMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 64,
        *,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(max(0, int(num_layers))):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(_activation(activation))
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, input_dim: int, output_dim: int, config: NetworkConfig) -> DeepCFRMLP:
        return cls(
            input_dim,
            output_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            activation=config.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
