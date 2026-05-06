from __future__ import annotations

from typing import Any


def encode_card(color: int, rank: int, n_ranks: int) -> int:
    return int(color) * (int(n_ranks) + 1) + int(rank)


def decode_card(card: int, n_ranks: int) -> tuple[int, int]:
    stride = int(n_ranks) + 1
    return int(card) // stride, int(card) % stride


def card_to_snapshot(card: int, n_ranks: int) -> dict[str, int]:
    color, rank = decode_card(card, n_ranks)
    return {"color": color, "rank": rank}


def encode_card_snapshot(data: Any, n_ranks: int) -> int:
    if isinstance(data, int):
        return data
    if isinstance(data, dict):
        return encode_card(int(data["color"]), int(data["rank"]), n_ranks)
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return encode_card(int(data[0]), int(data[1]), n_ranks)
    color = getattr(data, "color", None)
    rank = getattr(data, "rank", None)
    if color is not None and rank is not None:
        return encode_card(int(color), int(rank), n_ranks)
    raise ValueError(f"invalid card snapshot: {data!r}")
