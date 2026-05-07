# Deep CFR Performance Optimization and Scaling

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-performance-experiments-2026-05-07.md`

## Analysis of Trainer and Traversal Bottlenecks

Performance experiments on the baseline Deep CFR implementation (3-layer, 512-hidden MLP) reveal a clear trade-off between implementation complexity and hardware utilization. For models at this scale, the dominant overhead is not the raw floating-point operations on the GPU, but rather the Python-side dispatch and coordination logic. 

Three specific optimization attempts—`torch.compile`, Automated Mixed Precision (AMP), and GPU forward profiling—converged on the same conclusion: the current model is too small to benefit from standard PyTorch optimization "magic" without architectural changes to how data is fed to the model.

### Dispatch Overhead vs. Kernel Fusion

Both `torch.compile` and AMP (fp16) resulted in small performance regressions (approx. 5-18%) on the `default.yaml` and `smoke.yaml` configurations.

- **`torch.compile` regression:** While compilation reduces kernel launch overhead and enables fusion, the `DeepCFRMLP` architecture (`src/coolrl_lost_cities/games/classic/deep_cfr/networks.py:35`) is shallow enough that the bookkeeping overhead of the compiled wrapper exceeds these gains. Furthermore, because traversal runs in separate CPU multiprocessing workers, they do not benefit from the trainer's compiled networks unless specifically re-compiled or shared in a serialized format.
- **AMP regression:** Running the trainer optimization loops (`_train_advantage` at `trainer.py:1010` and `_train_strategy` at `trainer.py:1080`) in fp16 via `torch.autocast` proved counter-productive. The overhead of `GradScaler` bookkeeping and the frequent casting required for loss stability (e.g., computing squared loss in fp32 to maintain precision) outweighs the throughput win of lower-precision matrix multiplications at this model size.

**Rule of Thumb:** Re-evaluate these optimizations only when the model scale increases significantly (e.g., `hidden_size >= 1024` or `num_layers >= 6`).

### The Case for Batched Traversal

While the trainer is throughput-limited by dispatch, the traversal phase is the primary wall-clock bottleneck, accounting for approximately 60% of iteration time. Profiling the `DeepCFRMLP` forward pass (`scripts/profile_gpu_forward.py`) shows that the GPU is massively underutilized during standard recursive traversal (batch size = 1).

| Batch Size | μs per State | Efficiency vs. BS=1 |
| :--- | :--- | :--- |
| 1 | 80.07 | 1.00× |
| 64 | 1.46 | 54.75× |
| 256 | 0.34 | 232.03× |

A typical traversal visits ~360 nodes, providing a natural batching window. The transition from recursive traversal to an **interleaved scheduler** (`traversal.scheduler: interleaved`) leverages this by grouping policy requests across multiple concurrent traversals.

## Interleaved Traversal Architecture

The interleaved scheduler solves the "sequential bottleneck" of recursive MCCFR by decoupling state expansion from policy evaluation. Instead of waiting for a single forward pass per node, it maintains an explicit stack of active traversals and batches their network requests.

### Performance Gains

Comparing the recursive baseline against the interleaved scheduler (8 CPU workers, chunk size 64) on the `default.yaml` configuration:

- **Traversal wall-clock:** ~2.1× speedup.
- **Total iteration time:** ~1.5× speedup.
- **Node throughput:** Increased from 17.6k nodes/s to 31.3k nodes/s.

The 8-worker interleaved path is more effective than a single-process CUDA path because it preserves high CPU-side game-state throughput (encoding/decoding) while still benefiting from GPU batching.

## Practical Implications

- **Default Scheduler:** Interleaved traversal is the preferred default. The recursive path remains as a fallback (`traversal.scheduler: recursive`) for debugging or for verification of byte-identical RNG ordering.
- **Training Stability:** Interleaved traversal uses per-context RNG streams. While it matches recursive statistics in expectation, it does not produce identical sample ordering.
- **Future Scaling:** The next major throughput win lies in **Optimization Priority #5** (batched traversal inference), which will move the batched forward passes to a dedicated inference server, further reducing the worker-to-GPU coordination overhead.

## References

- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`: Implementation of AMP and training loops.
- `src/coolrl_lost_cities/games/classic/deep_cfr/networks.py`: `DeepCFRMLP` architecture.
- `configs/deep_cfr/default.yaml`: Configuration for interleaved scheduler.
- `scripts/profile_gpu_forward.py`: GPU forward pass micro-benchmarks.