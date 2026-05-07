# Deep CFR Performance Profile: Advantage Training Bottlenecks

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-profile-2026-05-07.md`

## Question
Which components of the Deep CFR training loop dominate execution time, and how does performance scale as the training run progresses and memory buffers grow?

## Analysis
Profile results from initial training runs indicate that **advantage network training** is the primary bottleneck in non-evaluation iterations, accounting for over 55% of total iteration time (e.g., 5.06s out of 9.13s). 

The profile reveals a significant scaling issue: the time spent in advantage training grows roughly linearly with the iteration count. Specifically, the time spent sampling from advantage memory (`time/advantage_player_X_sample_seconds`) increased from ~0.5s at iteration 1 to ~8.2s at iteration 10. During this window, the total advantage memory size grew from approximately 43,000 to 205,000 samples.

### Bottleneck: Linear Scan in Memory Sampling
The scaling bottleneck originates in the `ReservoirMemory.sample` implementation. When training on a shared advantage memory that requires player-specific filtering at sample time, the implementation performs a linear scan over the entire memory buffer to identify valid candidates.

With `advantage_updates_per_iteration` set to 256, the trainer performs 256 full scans of the growing memory buffer per player, per iteration. At iteration 10 (200k samples), this results in over 50 million object inspections per player per iteration, explaining the jump from sub-second sampling to nearly 10 seconds.

## Code Reference
The training loop is orchestrated in `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`:
- `run_iteration` (line 330): Coordinates traversal, advantage training, and strategy training.
- `_train_advantage` (line 1014): Records sampling time and performs the update steps.

The sampling bottleneck occurs in `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py` (line 64):
```python
def sample(self, batch_size, rng, *, player=None):
    candidates = (
        self._samples
        if player is None
        else [sample for sample in self._samples if sample.player == player]
    )
    # ...
```
The list comprehension on line 67 triggers the full traversal of `self._samples` whenever `player` is specified.

## Practical Implication
To maintain stable iteration times in long runs, **advantage memories must be split by player** at the trainer level. This ensures that `sample()` is called with `player=None` on a pre-filtered buffer, reducing the sampling operation to a constant-time index selection (plus the cost of object retrieval). 

Post-split profiling (`docs/archive/deep-cfr-profile-advantage-memory-split-2026-05-07.md`) verified that this architectural change reduces iteration 10 sampling time from ~8.2s to ~0.12s, effectively decoupling training performance from total memory size.

## References
- `docs/archive/deep-cfr-profile-2026-05-07.md` (Baseline Profile)
- `docs/archive/deep-cfr-profile-advantage-memory-split-2026-05-07.md` (Optimization Result)
- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`
- `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py`

# Deep CFR Performance Profile: Advantage Training Bottlenecks

**Last verified:** 2026-05-08
**Source:** `docs/archive/deep-cfr-profile-2026-05-07.md`

## Question
Which components of the Deep CFR training loop dominate execution time, and how does performance scale as the training run progresses and memory buffers grow?

## Analysis
Profile results from initial training runs indicate that **advantage network training** is the primary bottleneck in non-evaluation iterations, accounting for over 55% of total iteration time (e.g., 5.06s out of 9.13s). 

The profile reveals a significant scaling issue: the time spent in advantage training grows roughly linearly with the iteration count. Specifically, the time spent sampling from advantage memory (`time/advantage_player_X_sample_seconds`) increased from ~0.5s at iteration 1 to ~8.2s at iteration 10. During this window, the total advantage memory size grew from approximately 43,000 to 205,000 samples.

### Bottleneck: Linear Scan in Memory Sampling
The scaling bottleneck originates in the `ReservoirMemory.sample` implementation. When training on a shared advantage memory that requires player-specific filtering at sample time, the implementation performs a linear scan over the entire memory buffer to identify valid candidates.

With `advantage_updates_per_iteration` set to 256, the trainer performs 256 full scans of the growing memory buffer per player, per iteration. At iteration 10 (200k samples), this results in over 50 million object inspections per player per iteration, explaining the jump from sub-second sampling to nearly 10 seconds.

## Code Reference
The training loop is orchestrated in `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`:
- `run_iteration` (line 330): Coordinates traversal, advantage training, and strategy training.
- `_train_advantage` (line 1014): Records sampling time and performs the update steps.

The sampling bottleneck occurs in `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py` (line 64):
```python
def sample(self, batch_size, rng, *, player=None):
    candidates = (
        self._samples
        if player is None
        else [sample for sample in self._samples if sample.player == player]
    )
    # ...
```
The list comprehension on line 67 triggers the full traversal of `self._samples` whenever `player` is specified.

## Practical Implication
To maintain stable iteration times in long runs, **advantage memories must be split by player** at the trainer level. This ensures that `sample()` is called with `player=None` on a pre-filtered buffer, reducing the sampling operation to a constant-time index selection (plus the cost of object retrieval). 

Post-split profiling (`docs/archive/deep-cfr-profile-advantage-memory-split-2026-05-07.md`) verified that this architectural change reduces iteration 10 sampling time from ~8.2s to ~0.12s, effectively decoupling training performance from total memory size.

## References
- `docs/archive/deep-cfr-profile-2026-05-07.md` (Baseline Profile)
- `docs/archive/deep-cfr-profile-advantage-memory-split-2026-05-07.md` (Optimization Result)
- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`
- `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py`