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
    is_first_open: bool = False


class ReservoirMemory:
    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = capacity
        self._samples: list[TrainingSample] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: TrainingSample, rng: np.random.Generator | None = None) -> None:
        self.seen += 1
        sample = TrainingSample(
            info_state=np.asarray(sample.info_state, dtype=np.float32).copy(),
            target=np.asarray(sample.target, dtype=np.float32).copy(),
            legal_mask=np.asarray(sample.legal_mask, dtype=bool).copy(),
            iteration=int(sample.iteration),
            player=int(sample.player),
            is_first_open=bool(sample.is_first_open),
        )
        if self.capacity is None or len(self._samples) < self.capacity:
            self._samples.append(sample)
            return
        rng = rng or np.random.default_rng()
        index = int(rng.integers(0, self.seen))
        if index < self.capacity:
            self._samples[index] = sample

    def extend(self, samples: list[TrainingSample], rng: np.random.Generator | None = None) -> None:
        self.add_many(samples, rng)

    def add_many(
        self,
        samples: list[TrainingSample],
        rng: np.random.Generator | None = None,
    ) -> None:
        for sample in samples:
            self.add(sample, rng)

    def all(self) -> list[TrainingSample]:
        return list(self._samples)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        player: int | None = None,
        first_open_only: bool = False,
    ) -> list[TrainingSample]:
        candidates = self._samples
        if player is not None:
            candidates = [sample for sample in candidates if sample.player == player]
        if first_open_only:
            candidates = [sample for sample in candidates if sample.is_first_open]
        if not candidates:
            raise ValueError("cannot sample from empty memory")
        size = min(int(batch_size), len(candidates))
        indices = rng.choice(len(candidates), size=size, replace=len(candidates) < size)
        return [candidates[int(index)] for index in indices]

    def count(self, *, player: int | None = None, first_open_only: bool = False) -> int:
        candidates = self._samples
        if player is not None:
            candidates = [sample for sample in candidates if sample.player == player]
        if first_open_only:
            candidates = [sample for sample in candidates if sample.is_first_open]
        return len(candidates)
