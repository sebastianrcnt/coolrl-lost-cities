# Option A Benchmark and Structural Ceiling

**Last verified:** 2026-05-08, commit `8bbed31`
**Source:** `docs/archive/option-a-bench-result-2026-05-07.md`

## Question

Why did the centralized inference server (Option A) result in a 5x traversal regression (0.21x speedup) despite the GPU being ~230x faster at raw forward passes than the CPU?

Short answer: **the sync-blocking nature of the current traversal recursion prevents batching.** Because every worker thread waits for a single-row policy response before proceeding, the realized GPU batch size is limited by the worker count, which is too small to amortize the IPC and shared-memory synchronization overhead for small models.

## Code reference

The sync-blocking boundary is located in `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`. In `_regret_matching_policy_c` (around line 475) and `_policy_from_strategy_network` (around line 552), the traversal recursion crosses into PyTorch:

```cython
with torch.inference_mode():
    x = torch.as_tensor(info_state, dtype=torch.float32, device=self.device).unsqueeze(0)
    advantages = networks[player](x).squeeze(0).detach().cpu().numpy().astype(np.float32)
```

When `traversal.inference_backend` is set to `server` in `configs/deep_cfr/default.yaml`, the `networks[player]` call is intercepted by a `NetworkProxy` (instantiated in `workers.py`, around line 91). This proxy posts a request to the `InferenceServer` and blocks until a response is received via a per-slot event.

The server's batching logic in `src/coolrl_lost_cities/games/classic/deep_cfr/inference_server.py` (around line 221) reports the realized batch size:

```python
handles.stats_queue.put(BatchStatsMessage(batch_size=len(batch), group_count=len(groups)))
```

## Analysis

The 2026-05-07 benchmark results (`scripts/bench_inference_backend.py`) showed that with `num_workers: 8`, the mean batch size reported by the server was only **~7.2–7.9**. Because the server further splits batches by network kind and index to avoid mixing weights in a single forward pass, the actual GPU group size was roughly **4 rows**.

Comparing the per-state costs from the microbenchmark (`scripts/profile_gpu_forward.py`):

| Realized Batch | μs/state (GPU) |
| ---: | ---: |
| 1 | 80.07 |
| 4 | 20.30 |
| 64 | 1.46 |
| 256 | 0.34 |

At a batch size of 4, the GPU compute time (~20μs) is negligible compared to the IPC round-trip cost (queue post, context switch, event wakeup), which is on the order of **100–200μs** per call. This overhead is compounded over ~200k policy calls per iteration, leading to the observed jump from ~10.8s (local CPU) to ~51.6s (remote GPU) for the traversal phase.

The "structural ceiling" is that `batch_window_us` and `max_batch` tuning cannot improve performance if there are no additional in-flight requests to coalesce. With 8 workers blocking synchronously, the server will never see the 64+ requests needed to reach the high-efficiency SIMD regime.

## Practical implication

Option A is deferred for the current small MLP models (512x3). The `local` backend remains the default in `configs/deep_cfr/default.yaml`.

To unlock the projected GPU gains, the traversal must be restructured to drive batch sizes up. This leads to two primary paths:
1.  **Option B (Interleaved Traversal):** Refactor the Cython traversal into a state machine that can advance multiple traversals concurrently per worker. Each worker would suspend at a policy call, batch its own requests, and resume continuations once the results return.
2.  **Option C (Vectorized Traversal):** Re-implement traversal as a single-process operation on large tensors.

Option B is the chosen next step because it preserves the Cython game-rule logic while breaking the sync-blocking boundary. Option A's plumbing remains in the codebase as it will become beneficial if (a) the model size increases to the point where forward compute exceeds IPC cost, or (b) Option B successfully drives the server's batch size toward 64+.

## References

- `docs/research/batched-traversal-inference-decision.md`
- `experiments/traversal_policy_boundary/bench_policy_boundary.py`
- `docs/archive/post-a-optimization-calculus-2026-05-07.md`
