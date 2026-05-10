from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplaySample:
    info_state: np.ndarray
    legal_mask: np.ndarray
    pi_target: np.ndarray
    v_target: float
    player: int
    prior: np.ndarray | None = None
    game_index: int | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self._items: deque[ReplaySample] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, samples: Iterable[ReplaySample]) -> None:
        self._items.extend(samples)

    def sample(self, batch_size: int) -> list[ReplaySample]:
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        size = min(int(batch_size), len(self._items))
        indices = self.rng.choice(len(self._items), size=size, replace=False)
        items = list(self._items)
        return [items[int(index)] for index in indices]
