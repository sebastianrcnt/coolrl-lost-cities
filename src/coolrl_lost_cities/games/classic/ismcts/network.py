from __future__ import annotations

import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.networks import _activation

from .config import IsMctsConfig


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_size: int,
        hidden_size: int = 512,
        *,
        num_layers: int = 3,
        activation: str = "relu",
        value_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_size = int(action_size)
        self.value_scale = float(value_scale)
        layers: list[nn.Module] = []
        last_dim = self.input_dim
        for _ in range(max(0, int(num_layers))):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(_activation(activation))
            last_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, self.action_size)
        self.value_head = nn.Linear(last_dim, 1)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        action_size: int,
        config: IsMctsConfig | object,
    ) -> AlphaZeroNet:
        network_config = config.network if hasattr(config, "network") else config
        return cls(
            input_dim,
            action_size,
            network_config.hidden_size,
            num_layers=network_config.num_layers,
            activation=network_config.activation,
        )

    def forward(
        self,
        info_state_tensor: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(info_state_tensor)
        logits = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1) * self.value_scale
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)
        return logits, value

    def policy_distribution(
        self,
        info_state_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits, _value = self.forward(info_state_tensor, legal_mask)
        probs = torch.softmax(logits, dim=-1).masked_fill(~legal_mask.bool(), 0.0)
        normalizer = probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return probs / normalizer


class AlphaZeroLogitsView(nn.Module):
    """Expose AlphaZeroNet's policy logits as a one-argument module."""

    def __init__(self, net: AlphaZeroNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _value = self.net(x, legal_mask=None)
        return logits
