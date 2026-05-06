from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingSample:
    info_state: np.ndarray
    target: np.ndarray
    legal_mask: np.ndarray
    iteration: int
    player: int


class ReservoirMemory:
    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = capacity
        self._samples: list[TrainingSample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: TrainingSample) -> None:
        self._samples.append(sample)
        if self.capacity is not None and len(self._samples) > self.capacity:
            del self._samples[0 : len(self._samples) - self.capacity]

    def all(self) -> list[TrainingSample]:
        return list(self._samples)
