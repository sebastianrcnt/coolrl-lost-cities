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


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    num_layers: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(max(0, int(num_layers))):
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(_activation(activation))
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


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
        self.net = _build_mlp(input_dim, output_dim, hidden_size, num_layers, activation)

    @classmethod
    def from_config(cls, input_dim: int, output_dim: int, config: NetworkConfig) -> nn.Module:
        if config.kind == "color_shared":
            return ColorSharedNetwork(
                input_dim,
                output_dim,
                config.hidden_size,
                num_layers=config.num_layers,
                activation=config.activation,
                color_attention_layers=config.color_attention_layers,
                color_attention_heads=config.color_attention_heads,
            )
        return cls(
            input_dim,
            output_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            activation=config.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ColorSharedNetwork(nn.Module):
    """Color-shared architecture that splits input into per-color blocks.

    Splits the input into n_colors equal parts, encodes each with shared weights,
    pools the color embeddings, and concatenates with the original input.
    """

    N_COLORS = 5

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 64,
        *,
        num_layers: int = 2,
        activation: str = "relu",
        color_attention_layers: int = 0,
        color_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_colors = self.N_COLORS
        self.color_attention_layers = color_attention_layers
        self.color_attention_heads = color_attention_heads

        color_block_size = input_dim // self.n_colors
        self.color_block_size = color_block_size

        self.color_encoder = _build_mlp(
            color_block_size,
            hidden_size,
            hidden_size,
            num_layers,
            activation,
        )

        self.color_attention = None
        if color_attention_layers > 0:
            self.color_attention = ColorAttention(
                hidden_size,
                num_layers=color_attention_layers,
                num_heads=color_attention_heads,
                activation=activation,
            )

        final_input_dim = hidden_size * 2 + input_dim % self.n_colors

        self.final_net = _build_mlp(
            final_input_dim,
            output_dim,
            hidden_size,
            num_layers,
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color_embeddings = []
        for i in range(self.n_colors):
            start = i * self.color_block_size
            end = start + self.color_block_size
            block = x[:, start:end]
            embedding = self.color_encoder(block)
            color_embeddings.append(embedding)

        color_embeddings = torch.stack(color_embeddings, dim=1)

        if self.color_attention is not None:
            color_embeddings = self.color_attention(color_embeddings)

        mean_pooled = color_embeddings.mean(dim=1)
        max_pooled = color_embeddings.max(dim=1)[0]

        remainder = x[:, self.n_colors * self.color_block_size :]
        final_features = torch.cat([mean_pooled, max_pooled, remainder], dim=1)

        logits = self.final_net(final_features)
        return logits


class ColorAttention(nn.Module):
    """Self-attention over per-color embeddings."""

    def __init__(
        self,
        dim: int,
        *,
        num_layers: int = 1,
        num_heads: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    activation=activation.lower(),
                    batch_first=True,
                    norm_first=True,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
