# Deep CFR Runtime: Legacy vs. Current Implementation

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-legacy-runtime-comparison-2026-05-07.md`

## Question

How does the current Deep CFR implementation's performance compare to the legacy `../coolrl` codebase, and what architectural changes drove the observed speedups?

## Code reference

- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py` (lines 337, 341, 347, 357): Core training loop timers for `time/traversal_seconds`, `time/advantage_train_seconds`, `time/strategy_train_seconds`, and `time/evaluation_seconds`.
- `src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py` (line 380): Batched action selection for evaluation games, allowing multiple environments to share a single GPU forward pass.
- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py` (line 891): `_evaluate_parallel` method implementing multi-process evaluation across different opponents.
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`: Cython-optimized traversal logic providing the high-throughput foundation for MCCFR.

## Analysis

The current implementation demonstrates a significant performance leap over the legacy `../coolrl` system, with end-to-end wall time improvements ranging from 1.7x to 2.1x depending on the evaluation configuration. This speedup is the result of a deliberate shift toward batched GPU inference and parallelized environment execution.

### Training Iteration Throughput
Normal training iterations (excluding evaluation) improved from approximately 11.5 seconds to 5.8 seconds (~1.97x speedup). This gain is primarily attributed to optimizations in the traversal and network update phases:

- **Traversal:** Improved from 7.16s to 3.16s (2.27x faster). This is driven by the Cythonized traversal loop and efficient management of worker chunking, which minimizes the overhead of Python-to-Cython transitions.
- **Optimization:** Advantage and strategy training phases together improved from ~4.25s to ~2.65s (1.6x faster). The improvement here is largely due to more efficient tensor materialization from the replay buffers, reducing the time the GPU spends waiting for host-side data preparation.

### Evaluation Efficiency
Evaluation was a major bottleneck in the legacy system, averaging 18.6 seconds per session. The current implementation offers two primary modes of improvement:

- **Batched Sequential:** By grouping evaluation games into chunks (default `batch_size: 64` in `evaluate.py:380`), the overhead of single-state GPU inference is mitigated. This reduces evaluation time to 14.8s (1.25x faster).
- **Opponent-Parallel:** Parallelizing evaluation across multiple opponents (via `_evaluate_parallel` in `trainer.py:891`) further reduces wall-clock time to 6.4s. This represents a 2.9x speedup over the legacy evaluation average.

### End-to-End Comparison
Using a simple cadence model of one evaluation every five iterations, the total wall time for a 5-iteration block dropped from ~76.2s in the legacy system to ~35.6s with parallel evaluation. This 2.14x overall speedup allows for more frequent checkpoints and faster hypothesis testing without increasing the total training budget.

## Practical implication

- **Strict Superiority:** The current implementation is significantly more efficient than the legacy code, establishing it as the definitive platform for all Lost Cities Deep CFR research.
- **Bottleneck Distribution:** Despite these gains, traversal remains the largest phase (roughly 60% of iteration time). Future optimizations should prioritize batched traversal inference or "interleaved" execution to further leverage GPU compute during the traversal phase.
- **Evaluation Scaling:** The 2.9x speedup in evaluation enables more frequent, high-fidelity monitoring (e.g., evaluating against a full suite of heuristic bots every 10 iterations) with minimal impact on total training time.

## References

- `docs/archive/deep-cfr-legacy-runtime-comparison-2026-05-07.md`
- `docs/performance.md`
- `docs/research/deep-cfr-batched-evaluation.md`