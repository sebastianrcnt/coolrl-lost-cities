# Batched and Parallel Evaluation in Deep CFR

**Last verified:** 2026-05-08, commit `b0b3855`
**Source:** docs/archive/deep-cfr-batched-evaluation-2026-05-07.md

## Question

Evaluation of Deep CFR strategies against heuristic opponents can be a major bottleneck during training, especially when using CUDA for network inference. How can batched inference and parallel execution be leveraged to minimize this cost without introducing synchronization overhead?

## Code reference

- `src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py`, `StrategyNetPolicy.select_actions_batch` (line 174): Implements batched policy network inference, allowing multiple games to share a single GPU forward pass.
- `src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py` (line 220): Performs batched entropy calculation directly on the GPU using Torch tensors.
- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`, `DeepCFRTrainer._evaluate_parallel` (line 891): Orchestrates parallel evaluation across different opponents using `ProcessPoolExecutor`.

## Analysis

The primary bottleneck in CUDA-based evaluation is the overhead of launching small GPU kernels for single-state network inference. By batching evaluation games, we can saturate the GPU's compute units more effectively. Empirical results from May 2026 show that increasing the evaluation batch size to 64 reduced evaluation time from approximately 61.8 seconds to 14.8 seconds per iteration.

A critical refinement in the batched implementation was the handling of policy entropy. Initial versions that calculated entropy per-row on the CPU incurred significant synchronization penalties because each row required a GPU-to-CPU transfer. Moving the entropy calculation into the Torch post-processing pipeline—specifically calculating it directly on the `probs_tensor` (line 220)—ensures that the computation remains on the device and only the final results are transferred back to the host in bulk.

Once network inference is batched, the remaining bottleneck often shifts to the CPU-bound logic of heuristic opponents (e.g., `safe_heuristic_strict`). Parallelizing the evaluation across multiple workers allows the trainer to evaluate against multiple opponents simultaneously. For a single iteration profile, using 4 parallel workers reduced wall-clock evaluation time from 14.8 seconds to 6.4 seconds, achieving a ~2.3x speedup.

## Practical implication

- **Enable Batching:** For GPU-accelerated training, always configure `evaluation.batch_size` (typically 64 or 128) to minimize kernel launch overhead and maximize throughput.
- **Consolidate Device Operations:** Keep post-inference operations (legal masking, softmax, entropy) in Torch tensors to avoid blocking the GPU with frequent host-device synchronizations.
- **Parallelize Opponents:** Set `evaluation.num_workers` to match the number of opponents being evaluated (up to the available CPU cores). This effectively hides the latency of slower heuristic bots behind the network inference of others.

## References

- `docs/archive/deep-cfr-batched-evaluation-2026-05-07.md`
- `docs/performance.md`