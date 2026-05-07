# Post-A Optimization Calculus

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/post-a-optimization-calculus-2026-05-07.md`

## Question

Why do architectural optimizations like `torch.compile` and TensorRT integration yield negligible or even negative returns in the current Deep CFR implementation, and what specific triggers will shift them from "distractions" to "critical path" requirements?

## Code reference

The current baseline configuration is defined in `configs/deep_cfr/default.yaml`:

```yaml
network:
  hidden_size: 512
  num_layers: 3
```

Performance is measured via `scripts/bench_inference_backend.py`, which targets the traversal and evaluation inference paths. Current benchmarks (recorded in `docs/performance.md`) show that the wall-clock time is dominated by traversal (60%) and advantage/strategy training (39%), with evaluation occupying a small amortized share at the default `eval_every: 25`.

## Analysis

In the current development phase, kernel fusion and specialized inference engines are dispatch-bound rather than compute-bound. For a small model (3 layers, 512 hidden), the time spent executing the actual linear layers and activations is comparable to the overhead of the Python-to-C++ dispatch and CUDA kernel launch latency.

### The Small-Model Regression
Empirical results in `docs/archive/deep-cfr-performance-experiments-2026-05-07.md` show that `torch.compile` on the trainer's networks regressed performance. This occurs because the compile-time overhead and the fusion of very small kernels do not amortize effectively; the overhead of the optimized dispatch path is greater than the execution time of the unoptimized kernels. Similarly, TensorRT on the inference-server forward pass would only shave ~20–40μs off a call that already takes ~90μs, providing an iteration-level gain of less than 1%.

### The Phase Shift
The "calculus" shifts when two factors change the bottleneck profile:

1.  **Model Scaling:** Increasing to ~1024 hidden units and ~6 layers pushes the model into kernel-bound territory. Per-call time scales with FLOPs, while batching gains and dispatch overhead remain relatively fixed. At this scale, the 1.5–2.0× speedup provided by TensorRT or `torch.compile` fusion begins to dominate the wall-clock time.
2.  **Evaluation Density:** Evaluation is pure inference. As we move toward denser evaluation (e.g., `eval_every: 5` and `evaluation.games: 1000`), evaluation can grow to occupy 50% or more of the total iteration time. Since the inference server handles both traversal and evaluation, any gain in the forward path (such as TensorRT) applies directly to this large time slice.

When combined, these shifts can make the same tools that are currently performance-neutral deliver a ~1.5× total iteration speedup.

## Recommended Sequencing

To avoid misleading benchmark data and wasted integration effort, optimizations must follow the growth of the model and evaluation load:

1.  **Baseline Validation**: Confirm the "Option A" (batched traversal inference) multipliers using `scripts/bench_inference_backend.py`.
2.  **Model Selection**: Determine the target model size for production runs. This is the prerequisite for all subsequent optimization work.
3.  **Trainer Optimization**: Re-measure `torch.compile` on the trainer *only after* the model size is increased.
4.  **Inference Optimization**: Integrate TensorRT into the inference server once evaluation load or model size makes the forward pass a significant (>10%) share of wall-clock time.

## Practical Implication

Avoid premature optimization with `torch.compile` or TensorRT on the current small-model baseline. These tools should be treated as "Model Scale" features rather than "Algorithm" features; their value is unlocked by the compute intensity of the configuration, not the correctness of the implementation.

## References

- `docs/archive/deep-cfr-performance-experiments-2026-05-07.md` (Small-model regression data)
- `docs/archive/option-a-bench-result-2026-05-07.md` (Batched traversal benchmarks)
- `docs/performance.md` (Current runtime bottleneck profile)