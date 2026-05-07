# Deep CFR Performance Experiments (2026-05-07)

**Source:** Extracted from `docs/performance.md` § "Experiments" on
2026-05-08 to honor the "dated experiment records → docs/archive/"
routing rule from AGENTS.md.

Bundles four trainer- and traversal-side optimization experiments
performed 2026-05-07 on the small `default.yaml` model
(3-layer, 512-hidden). Three of them (`torch.compile`, AMP, GPU forward
profiling) surfaced the same underlying truth — the model is too small
for kernel-fusion / lower-precision wins to amortize their dispatch
overhead. The fourth (Option B interleaved traversal) passed and became
the new default.

## `torch.compile` on trainer networks (regression)

Wrapped both advantage networks and the strategy network with
`torch.compile()` at trainer construction time. Implementation also
required a `_clean_state_dict()` helper to strip the `_orig_mod.` prefix
that compiled modules add to `state_dict()`, plus a `_orig_mod`-routed
path for `load_state_dict()` so multiprocessing traversal workers and
checkpoint restoration could keep using the uncompiled `DeepCFRMLP`
class.

Measurement (8 iterations on `default.yaml`, eval and checkpoint
disabled, iteration 1 dropped as compile warm-up):

| | iter mean | 1000-iter projection |
| --- | ---: | ---: |
| Baseline (no compile) | 17.93s | 4.98h |
| `torch.compile` on trainer nets | 18.79s | 5.22h |
| Effect | +0.86s (+4.8%) | +14 min |

Net result: regression. Two reasons:

- Traversal is ~60% of iteration time and runs in CPU multiprocessing
  workers that reconstruct networks from cleaned `state_dict`s, so they
  bypass the compiled wrapper entirely.
- `DeepCFRMLP` (512-hidden, 3-layer) is small enough that the compiled
  call dispatch overhead exceeds the kernel-fusion benefit.

Implementation preserved on branch `experiments/torch-compile` for
revisiting if the trainer model grows substantially or after the
batched-traversal-inference work in Optimization Priorities #5 lands —
that is the change that would put compile on the dominant phase, not
just on the trainer's optimization steps. Not enabled on `main`.

## AMP on trainer networks (regression)

Wrapped the trainer optimization phases with `torch.autocast(fp16)` and
`torch.amp.GradScaler`: `_train_advantage` and `_train_strategy` now run
their network forward/backward/optimizer step through the AMP path when
`run.use_amp=true` and the trainer device is CUDA.

Safety mitigations included in the implementation:

- `GradScaler.unscale_(optimizer)` is called before `clip_grad_norm_`.
- Non-finite loss guard increments `amp/nonfinite_loss_count` and skips the
  bad step instead of applying it.
- Advantage squared loss computes `diff.float().square()` so the loss
  reduction is fp32 even when the forward path is autocast to fp16.
- Strategy logits are cast back to fp32 before `masked_fill` and
  `log_softmax`.
- Metrics now expose `amp/grad_scale` and `amp/nonfinite_loss_count`.

Measurement used the small `smoke.yaml` config with synthetic replay-memory
samples via:

```bash
uv run python scripts/bench_amp_trainer.py \
  --config configs/deep_cfr/smoke.yaml \
  --runs 3 \
  --warmup 1 \
  --device cuda
```

| | mean ms/call | speedup vs fp32 |
| --- | ---: | ---: |
| fp32 | 3.22 | 1.00× |
| AMP (fp16) | 3.92 | 0.82× |

Net result: regression. This matches the same dispatch-overhead-vs-kernel
benefit dynamic as the `torch.compile` regression above: the current trainer
model and smoke workload are too small for AMP's lower-precision kernels to
pay back autocast and scaler bookkeeping overhead.

The full `default.yaml` 100-iteration A/B was intentionally skipped. Given
the small-model regression and the matching `torch.compile` precedent on the
same model family, there is no current evidence that spending GPU time on the
longer A/B would produce a different decision. The infrastructure is kept
merged but default-off: `run.use_amp=false` remains the default, and
re-enabling is a one-field config flip.

Re-measure AMP only after the model grows to at least `hidden_size >= 1024`
or `num_layers >= 6`. At that point run both the fast
`scripts/bench_amp_trainer.py` micro-bench and the formal 100-iteration
fp32-vs-AMP A/B. If AMP still provides less than 5% speedup at that larger
model size, keep it default-off and raise the next re-measure trigger to an
even larger model.

## GPU forward profiling for batched traversal (decision support)

To decide whether Optimization Priorities #5 (batched traversal inference) is
worth implementing, profiled `DeepCFRMLP` from `default.yaml`
(input_dim=365, output_dim=22, hidden=512, 3 layers, ReLU) on an RTX 3090 in
`eval()` + `inference_mode`, with 10-iter warm-up and 1000-iter measurement
per batch size. Script: `scripts/profile_gpu_forward.py`.

| Batch size | μs/call | μs/state | Speedup vs bs=1 |
| ---: | ---: | ---: | ---: |
| 1 | 80.07 | 80.074 | 1.00× |
| 4 | 81.20 | 20.299 | 3.94× |
| 16 | 91.30 | 5.706 | 14.03× |
| 64 | 93.61 | 1.463 | 54.75× |
| 256 | 88.34 | 0.345 | 232.03× |
| 1024 | 161.95 | 0.158 | 506.30× |

Policy-call supply from
`runs/tmp/2026-05-07_181155_deep-cfr-default/metrics.jsonl`: mean
`traversal/nodes` ≈ 205,810 over 280 traversals/player → ~368 policy calls per
traversal (rough upper bound on batchable states), ~200k per iteration across
560 traversals.

Verdict: **Priority #5 is worth pursuing.** Per-state cost drops from 80 μs at
bs=1 to 0.34 μs at bs=256 (>230×). The available supply of ~368 states per
traversal sits comfortably in the bs=64–256 range where μs/call plateaus near
90 μs. End-to-end gain will be bounded by encoding and worker-GPU coordination
overhead, but the GPU forward is not the limiter once batching is in place.

## Option B interleaved traversal (pass)

Implemented a non-default interleaved traversal scheduler behind:

```yaml
traversal:
  scheduler: interleaved
```

The recursive Cython path remains the default. The current interleaved path is
a Python explicit-stack production prototype, guarded to the narrow case
`sampling_mode=outcome`, `opponent_policy=network`,
`cutoff_value_mode=score_diff`, `cutoff_rollouts=0`, and
`inference_backend=local`.

Measurement used `configs/deep_cfr/default.yaml` with evaluation and
checkpointing disabled, 10 iterations, first 2 iterations dropped as warm-up.
Because interleaved currently supports only `opponent_policy=network`, the
recursive baseline used the same opponent-policy override.

Commands:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.max_iterations=10 \
  --set run.experiment_name=option-b-recursive-network-10i \
  --set traversal.opponent_policy=network \
  --set checkpoint.save_latest=false \
  --set checkpoint.save_every=0 \
  --set evaluation.eval_every=0

uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.max_iterations=10 \
  --set run.experiment_name=option-b-interleaved-workers-10i \
  --set traversal.scheduler=interleaved \
  --set traversal.opponent_policy=network \
  --set traversal.num_workers=8 \
  --set traversal.worker_chunk_size=64 \
  --set traversal.interleave_width=64 \
  --set traversal.interleave_max_batch=128 \
  --set traversal.progress_every_traversals=0 \
  --set checkpoint.save_latest=false \
  --set checkpoint.save_every=0 \
  --set evaluation.eval_every=0

uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.max_iterations=10 \
  --set run.experiment_name=option-b-interleaved-cuda-single-10i \
  --set traversal.scheduler=interleaved \
  --set traversal.opponent_policy=network \
  --set traversal.num_workers=0 \
  --set traversal.interleave_width=64 \
  --set traversal.interleave_max_batch=128 \
  --set traversal.progress_every_traversals=0 \
  --set checkpoint.save_latest=false \
  --set checkpoint.save_every=0 \
  --set evaluation.eval_every=0
```

Result paths:

- `runs/2026-05-07_225044_option-b-recursive-network-10i`
- `runs/2026-05-07_225342_option-b-interleaved-workers-10i`
- `runs/2026-05-07_225542_option-b-interleaved-cuda-single-10i`

Warm-up-excluded means:

| Mode | iter s | traversal s | traversal speedup | iter speedup | nodes/s | batch mean | batch max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recursive, 8 workers, chunk 8 | 16.49 | 10.22 | 1.00× | 1.00× | 17.6k | 1.0 | 1 |
| interleaved, 8 workers, chunk 64 | 10.81 | 4.92 | 2.08× | 1.53× | 31.3k | 35.2 | 64 |
| interleaved, single CUDA process | 11.51 | 5.35 | 1.91× | 1.43× | 24.7k | 36.7 | 64 |

Net result: **PASS for the Phase 3 traversal-speed gate.** The best candidate
is the 8-worker interleaved path with larger worker chunks. It reaches real
batches near the target regime (`max_batch_size=64`) and more than doubles
traversal wall-clock versus the recursive network-opponent baseline.

The single-process CUDA path confirms that GPU forward is no longer the
dominant cost once batching works (`interleaved/forward_seconds` averaged
0.64s versus 4.53s for CPU worker forward), but it gives up multiprocessing
game-state throughput and is slower end-to-end than 8 interleaved CPU workers.

Important caveat: multi-traversal interleaving uses per-context RNG streams, so
exact recursive-batch RNG ordering is intentionally not preserved. The Phase 2
single-traversal parity test matches recursive stats and sample target
checksums under identical RNG seed. This means the default now favors the
measured traversal-speed win over byte-identical sample ordering. If future
learning curves show unexplained drift, first compare against the recursive
fallback:

```bash
--set traversal.scheduler=recursive \
--set traversal.worker_chunk_size=8 \
--set traversal.progress_every_traversals=10
```

Follow-up: `average_strategy` support was added after the initial Phase 3
network-opponent A/B so the interleaved path can run the actual default opponent
policy. A 10-iteration throughput check with default opponent policy,
evaluation/checkpoint disabled, and the same 8-worker chunk-64 interleaving
settings produced warm-up-excluded means:

| Mode | iter s | traversal s | batch mean | batch max |
| --- | ---: | ---: | ---: | ---: |
| interleaved, default `average_strategy` | 10.61 | 4.85 | 28.3 | 64 |

Run: `runs/2026-05-07_230419_option-b-interleaved-average-strategy-10i`.
This follow-up unblocked making interleaved traversal the default. The default
switch was made with the caveat above rather than waiting for a long-run A/B.
