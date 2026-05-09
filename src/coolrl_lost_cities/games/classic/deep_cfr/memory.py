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
        self._first_open_indices: list[int] = []
        self._first_open_positions: dict[int, int] = {}
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
            self._track_index(len(self._samples) - 1, sample)
            return
        rng = rng or np.random.default_rng()
        index = int(rng.integers(0, self.seen))
        if index < self.capacity:
            self._untrack_index(index)
            self._samples[index] = sample
            self._track_index(index, sample)

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
        if first_open_only and player is None:
            if not self._first_open_indices:
                raise ValueError("cannot sample from empty memory")
            size = min(int(batch_size), len(self._first_open_indices))
            indices = rng.choice(
                len(self._first_open_indices),
                size=size,
                replace=len(self._first_open_indices) < size,
            )
            return [self._samples[self._first_open_indices[int(index)]] for index in indices]
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
        if first_open_only and player is None:
            return len(self._first_open_indices)
        candidates = self._samples
        if player is not None:
            candidates = [sample for sample in candidates if sample.player == player]
        if first_open_only:
            candidates = [sample for sample in candidates if sample.is_first_open]
        return len(candidates)

    def _track_index(self, index: int, sample: TrainingSample) -> None:
        if not sample.is_first_open:
            return
        self._first_open_positions[index] = len(self._first_open_indices)
        self._first_open_indices.append(index)

    def _untrack_index(self, index: int) -> None:
        position = self._first_open_positions.pop(index, None)
        if position is None:
            return
        last_index = self._first_open_indices.pop()
        if position == len(self._first_open_indices):
            return
        self._first_open_indices[position] = last_index
        self._first_open_positions[last_index] = position
