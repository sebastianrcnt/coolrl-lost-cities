# Advantage Memory Split Performance Optimization

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-profile-advantage-memory-split-2026-05-07.md`

## Question

Why does splitting the advantage memory reservoir by player provide such a dramatic speedup in Deep CFR training, and what was the primary bottleneck in the unified memory implementation?

The speedup is primarily due to eliminating an $O(N)$ linear scan during the sampling phase of advantage network optimization. In the unified implementation, every sampling request for a specific player's advantages required filtering the entire reservoir, which becomes prohibitively expensive as memory capacity scales to millions of samples.

## Code reference

The split is implemented in `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py` by initializing two separate reservoir memories:

```python
# Lines 239-241
self.advantage_memories = [
    ReservoirMemory(self.config.memory.advantage_capacity) for _ in range(2)
]
```

During the traversal phase, samples are routed to the per-player memory in `_add_advantage_samples` (line 321):

```python
self.advantage_memories[sample.player].add(sample, self.rng)
```

The bottleneck in the previous unified implementation was located in `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py`, within the `sample` method (lines 62-67):

```python
candidates = (
    self._samples
    if player is None
    else [sample for sample in self._samples if sample.player == player]
)
```

When training player $P$'s advantage network, the trainer must sample batches of advantages specifically for that player. If the reservoir is shared, `player=P` is passed to `sample()`, triggering the list comprehension. With a default capacity of 2,000,000 samples and 64 updates per iteration, this results in over 128 million object inspections per iteration for advantage training alone.

## Analysis

Deep CFR alternates between a traversal phase (where advantages are collected) and an optimization phase (where networks are updated). Because advantage networks are player-specific, we only ever need to sample advantages for one player at a time during the `_train_advantage` loop.

By splitting the memories at insertion time (during traversal), we move the $O(1)$ routing logic to a phase that already handles samples individually. This allows the sampling phase to treat its per-player reservoir as a pure pool of valid candidates, defaulting `player` to `None` and using `self._samples` directly. This transforms the sampling cost from $O(N \cdot K)$ to $O(B \cdot K)$, where $B$ is the batch size and $K$ is the number of updates.

Empirical results from the source profile show that sampling time for a single player dropped from approximately 1.65 seconds to 0.05 seconds per iteration—a ~30x improvement. Overall iteration time for non-evaluation steps dropped from 9.1 seconds to 5.8 seconds (~35% reduction).

## Practical implication

- **Scalability:** We can now scale `advantage_capacity` to the limits of system RAM without incurring a linear penalty in training time.
- **Update Frequency:** The reduction in training overhead allows for more `advantage_updates_per_iteration` (K) if needed for better convergence, as the fixed cost of sampling is now negligible.
- **Unified Strategy Memory:** Note that `strategy_memory` remains unified because the strategy network is shared (or trained on all-player data) in the current implementation. If the strategy network were also split per player, a similar optimization would apply.

## References

- `docs/archive/deep-cfr-profile-advantage-memory-split-2026-05-07.md`
- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`
- `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py`