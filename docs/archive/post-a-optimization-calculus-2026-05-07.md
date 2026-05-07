# Post-A Optimization Calculus (2026-05-07)

**Source:** Extracted from `docs/performance.md` § "Post-A Optimization
Calculus (forward-looking, 2026-05-07)" on 2026-05-08.

Forward-looking sequencing recorded *before* Option A was benched. Has
not been measured. See
`docs/archive/option-a-bench-result-2026-05-07.md` for the actual bench
which deferred Option A — some assumptions below ("once Option A
lands") have to be re-evaluated in light of that result.

---

Once Option A lands, the bottleneck shape changes. This section records the
expected sequencing for follow-up work. It is forward-looking and has not been
measured yet — verify against bench numbers after A is benchmarked.

## Why compile / TensorRT are negligible *today* but become meaningful later

Today (small model: 3-layer, 512 hidden):

- `torch.compile` on the trainer's networks already regressed (see the
  2026-05-07 experiment in
  `docs/archive/deep-cfr-performance-experiments-2026-05-07.md`).
  The model is too small for kernel fusion to beat compile dispatch
  overhead.
- `torch.compile` / TensorRT on the inference-server forward (post-A) would
  shave ~30–50% off ~90μs/call → ~50–70μs/call. With forward share of an iter
  reduced to <1% by A's batching, the iter-level multiplier is ~1.00–1.01×.
  Negligible.

Two compounding shifts can flip this:

1. **Larger model.** Going from 512 hidden / 3 layers to ~1024 hidden /
   ~6 layers pushes the forward call out of dispatch-bound territory into
   kernel-bound territory. Compile fusion and TensorRT both deliver real
   1.5–2× on the forward call itself once the kernel is large enough to
   amortize launch overhead. Forward share of iter time also rebalances upward
   because per-call time scales with FLOPs while batching gain is fixed.
2. **Denser, larger evaluation.** Moving toward `eval_every: 5` and
   `evaluation.games: 1000` makes evaluation about half of iteration wall-clock
   (see the amortized eval table in `docs/performance.md`). Eval is pure
   inference, so TensorRT on the inference-server's forward path applies
   directly.

When both shifts happen together, an illustrative future iter (rough order of
magnitude only):

| Configuration                                  | Iter time (rough) |
| ---                                            | ---: |
| Today (small model, eval_every=25)             | 17.85s |
| + A (batched traversal inference)              | ~14s |
| + larger model (≈4× FLOPs), no compile/TRT     | ~50s |
| + dense eval (eval_every=5, games=1000)        | ~70s |
| + compile (trainer) + TensorRT (inference)     | ~45s |

That last row is where compile/TensorRT contributes ~1.5× iter — the same
tooling that is iter-neutral today. The numbers above are illustrative; real
ratios depend on model size, kernel autotune outcomes, and the eval-vs-train
balance.

## Tooling split

- **TensorRT**: applies only to inference (no backward). Targets:
  - inference-server forward in traversal,
  - inference-server forward in evaluation.
  Both are served by the same A-era server, so a single TensorRT integration
  covers both.
- **`torch.compile`**: applies to trainer's advantage/strategy training
  (forward+backward+optimizer). The 2026-05-07 regression on a small model
  does **not** generalize — it must be re-measured on whatever larger model
  config we settle on. Do not conclude "compile is bad" from the small-model
  data point.

## Recommended sequencing

Do this in order. Skipping ahead is the failure mode that creates misleading
"compile/TRT didn't help" data.

1. **Now**: benchmark A (`scripts/bench_inference_backend.py`) and confirm the
   `local` vs `server` multipliers on `home` and `remote`. Validate the iter
   1.2–1.3× / traversal 1.5–2× working estimate.
2. **Next**: experiment with a larger network config. Measure compute vs
   learning-curve trade-off with the existing toolchain (no compile/TRT yet).
   This step decides the model size that future optimizations target.
   It is also the prerequisite for revisiting AMP, `torch.compile`, and
   TensorRT: all three are dispatch-overhead-bound on the current small model.
3. **Then**: re-measure `torch.compile` on the trainer at the chosen model
   size. The earlier regression was size-bound; expect a different result.
4. **Then**: integrate TensorRT into the inference server (covers traversal
   and eval forward simultaneously). Bound the gain by the post-step-2
   `policy_network_seconds` share, not the headline TensorRT speedup.
5. **In parallel with 2–4**: if denser eval is operationally useful, raise
   `evaluation.games` and lower `evaluation.eval_every`. This step does not
   require code changes but sharply increases the value of step 4.

Out of scope until A bench numbers are in: Option C, `nogil` threading, async
inference client, compiled encoding.
