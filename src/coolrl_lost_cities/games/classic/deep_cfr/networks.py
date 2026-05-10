from __future__ import annotations

import warnings
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ColorLayout:
    """Per-color and common index lists for a Lost Cities encoding vector."""

    per_color_indices: tuple[tuple[int, ...], ...]
    common_indices: tuple[int, ...]

    @property
    def n_colors(self) -> int:
        return len(self.per_color_indices)

    @property
    def color_block_size(self) -> int:
        return len(self.per_color_indices[0])

    @property
    def common_size(self) -> int:
        return len(self.common_indices)


def compute_lost_cities_color_layout(
    input_dim: int,
    *,
    n_colors: int = 5,
    hand_size: int = 8,
    n_ranks: int = 9,
) -> ColorLayout | None:
    """Map encoding offsets to per-color blocks for the standard Lost Cities schema.

    Returns ``None`` when ``input_dim`` does not match a recognised combination of
    encoding flags (``derived_playability``, ``slot_aware_playability``) under the
    given schema. The standard Lost Cities tier-3 game uses the defaults
    n_colors=5, hand_size=8, n_ranks=9, which produces 171 / 219 / 249 / 297 dims.

    Per-color indices, in semantic order, gather:
      - both players' expedition state for the color (4 dims each)
      - discard top metadata for the color (4 dims)
      - public card-type histogram for the color (n_ranks + 1 dims)
      - pending-discard one-hot bit for the color
      - legal-action draw-pile bit for the color
      - derived_playability per-color block (15 dims) when enabled
    Slot-aware features are slot-major (not color-major) and stay in common.
    """
    base_dim = (
        5
        + hand_size * 3
        + 2 * n_colors * 4
        + n_colors * 4
        + n_colors * (n_ranks + 1)
        + 3
        + 1
        + (n_colors + 1)
        + (2 * hand_size + 1 + n_colors)
    )
    derived_size = n_colors * 15 + 3
    slot_size = hand_size * 6

    has_derived = False
    if input_dim == base_dim:
        pass
    elif input_dim == base_dim + derived_size:
        has_derived = True
    elif input_dim == base_dim + slot_size:
        pass
    elif input_dim == base_dim + derived_size + slot_size:
        has_derived = True
    else:
        return None

    per_color: list[list[int]] = [[] for _ in range(n_colors)]

    expedition_start = 5 + hand_size * 3
    for player in range(2):
        for color in range(n_colors):
            base_idx = expedition_start + player * n_colors * 4 + color * 4
            per_color[color].extend(range(base_idx, base_idx + 4))

    discard_start = expedition_start + 2 * n_colors * 4
    for color in range(n_colors):
        per_color[color].extend(range(discard_start + color * 4, discard_start + color * 4 + 4))

    histogram_start = discard_start + n_colors * 4
    for color in range(n_colors):
        block_start = histogram_start + color * (n_ranks + 1)
        per_color[color].extend(range(block_start, block_start + n_ranks + 1))

    pending_start = histogram_start + n_colors * (n_ranks + 1) + 3 + 1
    for color in range(n_colors):
        per_color[color].append(pending_start + color)

    legal_start = pending_start + n_colors + 1
    draw_pile_start = legal_start + 2 * hand_size + 1
    for color in range(n_colors):
        per_color[color].append(draw_pile_start + color)

    if has_derived:
        derived_start = base_dim
        for color in range(n_colors):
            block_start = derived_start + color * 15
            per_color[color].extend(range(block_start, block_start + 15))

    color_set: set[int] = set()
    for indices in per_color:
        color_set.update(indices)
    common = [i for i in range(input_dim) if i not in color_set]

    return ColorLayout(
        per_color_indices=tuple(tuple(indices) for indices in per_color),
        common_indices=tuple(common),
    )


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
    """Color-shared architecture for Lost Cities encodings.

    For a recognised Lost Cities encoding (``input_dim`` matching a known
    combination of base, derived_playability, and slot_aware_playability), the
    forward pass gathers the per-color feature indices computed by
    :func:`compute_lost_cities_color_layout`, runs a shared encoder over each
    color block, mean+max pools, and concatenates the result with the
    color-independent ("common") features before the final head.

    For any other ``input_dim`` (e.g. unit tests using ``input_dim=100``), the
    network falls back to a *chunked* layout that splits the input into
    ``input_dim // n_colors`` equal slices. The chunked layout was the only
    behaviour shipped before 2026-05-10 and does **not** correspond to actual
    per-color blocks in the encoding — adjacent slices contain unrelated
    features (phase flags, hand slots, scores, etc.). It is preserved purely
    for backward compatibility with older checkpoints and tests; new training
    runs should always use the standard Lost Cities encoding so the proper
    layout is selected automatically.
    """

    DEFAULT_N_COLORS = 5

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
        self.color_attention_layers = color_attention_layers
        self.color_attention_heads = color_attention_heads

        layout = compute_lost_cities_color_layout(input_dim)

        if layout is not None:
            self.use_chunked_fallback = False
            self.n_colors = layout.n_colors
            self.color_block_size = layout.color_block_size
            self.common_size = layout.common_size
            per_color_idx = torch.tensor(
                [list(indices) for indices in layout.per_color_indices], dtype=torch.long
            )
            common_idx = torch.tensor(list(layout.common_indices), dtype=torch.long)
            self.register_buffer("per_color_indices", per_color_idx, persistent=False)
            self.register_buffer("common_indices", common_idx, persistent=False)
        else:
            warnings.warn(
                f"ColorSharedNetwork: input_dim={input_dim} does not match the "
                "standard Lost Cities encoding schema; falling back to chunked "
                "input slicing (legacy behaviour, semantically not per-color).",
                stacklevel=2,
            )
            self.use_chunked_fallback = True
            self.n_colors = self.DEFAULT_N_COLORS
            self.color_block_size = input_dim // self.n_colors
            self.common_size = input_dim - self.n_colors * self.color_block_size

        self.color_encoder = _build_mlp(
            self.color_block_size,
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

        final_input_dim = hidden_size * 2 + self.common_size
        self.final_net = _build_mlp(
            final_input_dim,
            output_dim,
            hidden_size,
            num_layers,
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_chunked_fallback:
            color_blocks = []
            for i in range(self.n_colors):
                start = i * self.color_block_size
                end = start + self.color_block_size
                color_blocks.append(x[:, start:end])
            stacked = torch.stack(color_blocks, dim=1)
            common = x[:, self.n_colors * self.color_block_size :]
        else:
            stacked = x[:, self.per_color_indices]
            common = x[:, self.common_indices]

        batch_size = stacked.shape[0]
        flat = stacked.reshape(batch_size * self.n_colors, self.color_block_size)
        encoded_flat = self.color_encoder(flat)
        encoded = encoded_flat.reshape(batch_size, self.n_colors, self.hidden_size)

        if self.color_attention is not None:
            encoded = self.color_attention(encoded)

        mean_pooled = encoded.mean(dim=1)
        max_pooled = encoded.max(dim=1)[0]

        final_features = torch.cat([mean_pooled, max_pooled, common], dim=1)
        return self.final_net(final_features)


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
