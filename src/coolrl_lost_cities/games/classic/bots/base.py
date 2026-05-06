from __future__ import annotations

from ..game import GameState
from ..interfaces import BotInput, Snapshot

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Lost Cities bots") from exc


def legal_from_obs(obs_or_state: BotInput) -> np.ndarray:
    if isinstance(obs_or_state, GameState):
        return np.asarray(obs_or_state.legal_mask(), dtype=bool)
    if isinstance(obs_or_state, Snapshot):
        return np.asarray(obs_or_state.legal_mask, dtype=bool)
    return np.asarray(obs_or_state["legal_mask"], dtype=bool)


def first_legal(legal: list[bool] | np.ndarray) -> int:
    legal_indices = np.nonzero(np.asarray(legal, dtype=bool))[0]
    if len(legal_indices) == 0:
        raise RuntimeError("no legal action available")
    return int(legal_indices[0])
