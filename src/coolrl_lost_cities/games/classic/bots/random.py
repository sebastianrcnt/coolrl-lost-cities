from __future__ import annotations

from ..interfaces import BotInput, LostCitiesBot
from .base import legal_from_obs

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Lost Cities bots") from exc


class RandomBot(LostCitiesBot):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs_or_state: BotInput) -> int:
        legal = legal_from_obs(obs_or_state)
        legal_indices = np.nonzero(legal)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("no legal action available")
        return int(self.rng.choice(legal_indices))
