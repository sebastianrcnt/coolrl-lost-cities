# Deep CFR Performance Notes

This document tracks current runtime bottlenecks for the active Deep CFR
training path. The numbers below are observational, not a benchmark contract.

## Current Default Runtime

Source run:

```text
runs/tmp/2026-05-07_171535_deep-cfr-default/metrics.jsonl
```

The run used `configs/deep_cfr/default.yaml` with CUDA enabled. At the time of
inspection, completed metrics covered iterations 70 through 95. The training
process was still running, so later rows may differ.

Non-evaluation iterations averaged:

| Metric | Mean | Share |
| --- | ---: | ---: |
| `iteration_seconds` | 17.85s | 100% |
| `traversal_seconds` | 10.71s | 60% |
| `advantage_train_seconds` | 4.72s | 26% |
| `strategy_train_seconds` | 2.39s | 13% |
| `memory_add_seconds` | 1.25s | 7% |
| `checkpoint_seconds` | 0.02s | <1% |

`batch_tensor_seconds` averaged 4.60s. This is not an additional phase; it is
included inside the advantage and strategy training timers. It measures the cost
of building per-step tensors from sampled replay entries, including `np.stack`
and transfer to the trainer device.

One evaluation iteration was present in the inspected window:

| Iteration | `iteration_seconds` | `evaluation_seconds` |
| ---: | ---: | ---: |
| 75 | 40.30s | 18.29s |

With the default `evaluation.eval_every: 25`, an 18.29s evaluation amortizes to
about 0.73s per training iteration. This makes evaluation noticeable in logs but
not the primary long-run wall-clock bottleneck.

If the same 100-game evaluation is run more often, the amortized cost becomes:

| `evaluation.eval_every` | Amortized eval cost |
| ---: | ---: |
| 25 | 0.73s/iteration |
| 10 | 1.83s/iteration |
| 5 | 3.66s/iteration |

If `evaluation.games` increases from 100 to 1000, evaluation cost should be
expected to grow roughly linearly unless fixed overhead or batching effects
dominate. Using the observed 18.29s eval as a rough base, a 1000-game eval would
be about 183s:

| `evaluation.games` | `evaluation.eval_every` | Rough amortized eval cost |
| ---: | ---: | ---: |
| 1000 | 25 | 7.3s/iteration |
| 1000 | 10 | 18.3s/iteration |
| 1000 | 5 | 36.6s/iteration |

At that point evaluation becomes a first-order wall-clock concern.

## Current Bottleneck Shape

Default training is split between CPU-heavy traversal and CUDA-backed network
training:

- Traversal is the largest measured phase at roughly 60% of non-eval iteration
  time.
- Advantage and strategy training together are roughly 40% of non-eval
  iteration time.
- Tensor materialization is a major part of training time, so reducing pure GPU
  compute alone cannot remove all of the training cost.

The default traversal settings are:

```yaml
traversal:
  traversals_per_player: 280
  num_workers: 8
  worker_chunk_size: 8
```

This gives 560 traversals per iteration, split into 70 worker batches.

## Device Use

The trainer constructs the advantage and strategy networks on `run.device`.
`configs/deep_cfr/default.yaml` sets:

```yaml
run:
  device: cuda
  use_amp: false
```

Traversal workers currently reconstruct networks on CPU:

```python
device = torch.device("cpu")
```

So, in the default multiprocessing traversal path, setting `run.device: cuda`
accelerates the trainer's optimization steps and trainer-device evaluation, but
does not move traversal worker inference to GPU.

On ROCm systems, AMD GPUs may still appear through PyTorch's `cuda` device API
if a compatible ROCm build is installed. The project does not have a separate
AMD-specific device path.

## AMP Status

`run.use_amp` exists in configuration, but automatic mixed precision is not
currently wired into the training loop. There are no active `autocast` or
`GradScaler` calls in the Deep CFR trainer.

Practical implication: setting

```bash
--set run.use_amp=true
```

should be treated as a no-op until the trainer implements AMP explicitly.

If implemented, AMP would mainly target the network optimization phases:

- `advantage_train_seconds`
- `strategy_train_seconds`

It would not directly reduce traversal CPU time or replay tensor
materialization overhead.

## Batching Status

There are two separate meanings of batching in the current codebase.

Implemented:

- Worker batching via `traversal.worker_chunk_size`.
- Evaluation policy batching via `evaluation.batch_size`.
- Optimization batching via `optimization.advantage_batch_size` and
  `optimization.strategy_batch_size`.

Not implemented:

- Batched network inference inside traversal.

Traversal policy evaluation currently encodes one state and runs a single-row
network call, effectively `batch_size == 1`, then converts the result back to
CPU/Numpy for Cython-side policy logic. This means traversal is not structured
to feed many states to the GPU in a single inference call.

Evaluation batching is already implemented. During evaluation, active games that
need a policy-network action are grouped into chunks of
`evaluation.batch_size`, then evaluated together on the evaluation device.
Default config uses:

```yaml
evaluation:
  batch_size: 64
```

This makes evaluation much more suitable for GPU inference optimization than
traversal is today. The remaining question is whether policy-network forward
time is actually the dominant part of evaluation.

## Evaluation Breakdown

The current source emits evaluation metrics as:

```text
eval/<opponent>/<metric>
```

The inspected run uses an older flattened scheme:

```text
eval_<opponent>_<metric>
```

For that run, iteration 75 had `evaluation_seconds = 18.29s`. Evaluation was
parallelized by opponent, so per-opponent `elapsed_seconds` values overlap and
must not be summed as wall-clock time. The slow safe-heuristic opponents
dominated the eval wall-clock.

Representative per-opponent breakdown:

| Opponent | Elapsed | Network | Postprocess | Opponent act |
| --- | ---: | ---: | ---: | ---: |
| `random` | 0.57s | 0.18s | 0.25s | 0.06s |
| `passive_discard` | 0.36s | 0.13s | 0.17s | 0.00s |
| `safe_heuristic` | 14.54s | 2.65s | 3.34s | 7.57s |
| `safe_heuristic_loose` | 11.20s | 2.55s | 3.30s | 4.63s |
| `safe_heuristic_strict` | 15.94s | 2.51s | 3.14s | 9.22s |
| `noisy_safe` | 1.68s | 0.40s | 0.58s | 0.57s |

The important read is that safe-heuristic evaluation is not primarily GPU
network forward time. `opponent_act_seconds` and policy post-processing are
larger than `policy_network_seconds` for the slowest opponents.

Useful eval runtime keys to inspect:

```text
eval_<opponent>_elapsed_seconds
eval_<opponent>_policy_network_seconds
eval_<opponent>_policy_encoding_seconds
eval_<opponent>_policy_postprocess_seconds
eval_<opponent>_policy_legal_mask_seconds
eval_<opponent>_opponent_act_seconds
eval_<opponent>_apply_action_seconds
eval_<opponent>_diagnostics_seconds
eval_<opponent>_final_scoring_seconds
```

For newer runs, replace `eval_<opponent>_<metric>` with
`eval/<opponent>/<metric>`.

## Evaluation Optimization Options

The practical eval tuning levers are:

1. Tune `evaluation.batch_size`.
   Try 128 or 256 if GPU memory allows. This helps most when
   `policy_network_seconds` is a large fraction of opponent elapsed time.

2. Tune `evaluation.num_workers`.
   Multiple workers parallelize opponents, but they can also split GPU work
   across processes and duplicate model copies. Compare 1, 2, and 4 workers for
   CUDA eval instead of assuming the largest value is fastest.

3. Split light and full evaluation.
   A useful schedule would run a small opponent/games set frequently and the
   full opponent suite less often. The current config has one eval schedule, so
   this would require a feature change.

4. Reduce frequent opponents.
   The safe-heuristic opponents dominate wall-clock in the inspected run. For
   frequent checks, evaluate against one or two representative opponents and run
   the full suite less often.

5. Add a diagnostics-light mode.
   Current eval records many action-quality and expedition diagnostics. The
   measured `diagnostics_seconds` is small in the inspected run, but a basic
   win-rate/score-only mode would still make frequent large evals simpler and
   cheaper.

6. Consider asynchronous evaluation.
   A separate process can evaluate checkpoints while training continues. This
   does not reduce total compute, and it can contend for GPU if run on the same
   device, but it removes eval pauses from the trainer wall-clock.

7. Consider eval-only AMP or compiled inference.
   This is simpler than training AMP because evaluation has no backward pass.
   It should be measured against `policy_network_seconds`; it will not reduce
   opponent policy time or game-state transition time.

## TensorRT Assessment

TensorRT is not an obvious high-priority optimization for the current default
training loop.

Reasons:

- The largest phase is traversal, and default multiprocessing traversal runs on
  CPU workers.
- Traversal network inference is single-state, control-flow-heavy, and crosses
  between encoded state arrays, PyTorch tensors, and CPU/Numpy outputs.
- TensorRT mainly helps inference, while the CUDA-backed trainer phases are
  training steps with backward passes and optimizer updates.
- Evaluation can benefit from inference optimization in principle, but default
  evaluation is only every 25 iterations. Even making evaluation much faster has
  limited effect on long-run average iteration time.

For evaluation specifically, TensorRT is more plausible because GPU batching is
already implemented. It would replace or wrap the strategy-network forward pass
with a precompiled inference engine. Its maximum impact is bounded by
`policy_network_seconds`, not by total eval time.

In the inspected eval row, the slow safe-heuristic opponents spent about
2.5-2.6s in policy-network forward but 4.6-9.2s in opponent action selection and
about 3.1-3.3s in policy post-processing. That means TensorRT could help eval,
especially for larger `evaluation.games`, but it is not expected to collapse the
18.29s eval to a tiny number by itself.

TensorRT becomes more attractive if:

- `evaluation.games` is raised substantially, such as 1000 games.
- `evaluation.eval_every` is reduced to 5 or 10.
- `policy_network_seconds / elapsed_seconds` rises after batch-size and worker
  tuning.

TensorRT may also become relevant for traversal after a larger traversal
redesign that batches many policy-needed states into GPU inference requests.

## Optimization Priorities

Based on the current metrics, the more plausible performance work is:

1. Improve traversal throughput.
   Tune worker count and chunk size, then profile the Cython traversal hot path.

2. Reduce training tensor materialization cost.
   `batch_tensor_seconds` is a large part of train time. More contiguous replay
   storage or tensor-ready sampled batches may help more than model-kernel
   tuning alone.

3. Implement and test AMP.
   This should be gated by `run.use_amp` and measured against loss stability and
   wall-clock, since it only targets the optimization phases.

4. Consider `torch.compile` for the trainer networks.
   This should be measured separately from traversal because the default
   training loop has substantial non-kernel overhead.

5. Consider batched traversal inference only as a structural project.
   This is the path that could make GPU inference accelerators more meaningful,
   but it requires changing traversal scheduling, not just swapping the network
   backend.

6. For eval-heavy runs, optimize the safe-heuristic opponents and policy
   post-processing before assuming TensorRT is the main lever.
   The inspected eval row shows those costs dominate the slowest opponents.

## Experiments

### `torch.compile` on trainer networks (2026-05-07, regression)

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

### AMP on trainer networks (2026-05-07, regression)

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

### GPU forward profiling for batched traversal (2026-05-07, decision support)

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

### Option B interleaved traversal (2026-05-07, pass)

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
checksums under identical RNG seed. Longer learning-curve A/B is still required
before considering a default switch.

Follow-up: `average_strategy` support was added after the initial Phase 3
network-opponent A/B so the interleaved path can run the actual default opponent
policy. A 10-iteration throughput check with default opponent policy,
evaluation/checkpoint disabled, and the same 8-worker chunk-64 interleaving
settings produced warm-up-excluded means:

| Mode | iter s | traversal s | batch mean | batch max |
| --- | ---: | ---: | ---: | ---: |
| interleaved, default `average_strategy` | 10.61 | 4.85 | 28.3 | 64 |

Run: `runs/2026-05-07_230419_option-b-interleaved-average-strategy-10i`.
This prepares the long-run default-config A/B, but it does not replace it:
learning-curve stability still has to be measured before any default switch.

## Batched Traversal Inference: Design Decision (2026-05-07)

Three structural options were considered for Priority #5:

- **A. Central inference server.** Workers stay in multiprocessing and
  reconstruct nothing on GPU. A separate server process owns the model, batches
  incoming policy requests across workers, runs GPU forward, returns logits.
  Worker traversal logic and the Cython recursion are untouched.
- **B. Per-worker batching.** Each worker interleaves multiple traversals
  internally to form its own batches. GPU process count = worker count, so model
  copies and GPU contention scale with workers. Batching efficiency is bounded
  by per-worker in-flight count.
- **C. Single-process vectorized traversal.** Drop multiprocessing entirely.
  Main process runs N traversals lockstep with explicit recursion stacks,
  forming a natural batch dimension across traversal instances. Existing
  recursive traversal can be kept and a new `traversal/batched.{py,pyx}` added
  as a parallel backend gated by config; existing code is not modified.

### Decision: A

Reasons:

- **Hardware fit dominates.** A central inference server keeps multiprocessing,
  so all available CPU cores stay productive on game logic. C is single-process,
  so on a 32-core remote machine with a weak GPU it wastes 31 cores while the
  weak GPU caps batching gains; A is strictly better there. On a 6-core / RTX
  3090 box A and C are competitive but uncertain — C only wins when GPU forward
  is the dominant share of traversal, and game logic in CFR traversal is not
  negligible.
- **C is not the "ultimate" answer on multi-core machines.** A truly maximal
  design would combine C's batched GPU forward with `nogil` threaded game
  logic, which is strictly more complex than C alone. Plain C, by being
  single-process, gives up CPU parallelism that the existing multiprocessing
  path already exploits.
- **A is mostly additive.** New modules: `inference_server.py`,
  `inference_client.py`, shared-memory tensor pool, weight-sync hook. Existing
  touches are small: worker policy call site (one line), worker spawn/teardown
  (server start/stop), trainer (periodic weight push). Cython traversal
  recursion, game engine, replay/training paths are unchanged.
- **The hard part is IPC tuning, not code volume.** Latency budget vs GPU
  forward, weight-staleness window, backpressure, and shared-memory tensor
  layout. Code is small; the design surface is concentrated in one place.

### IPC: what crosses the process boundary

Only the encoded policy input and its response cross IPC:

- Forward request: encoded state vector, ~365 floats ≈ 1.5KB.
- Forward response: action logits, ~22 floats ≈ 88 bytes.

Game state, traversal recursion stack, event log, CFR regret/strategy
accumulators, and chance-node sampling history all stay inside the worker
process. The policy network consumes a flat encoded state (`input_dim=365`),
so the server needs no game-tree context to answer a request.

The replay-buffer write path (workers shipping collected regret/strategy
samples to the trainer) is separate, already exists today, and is reflected in
`memory_add_seconds` ≈ 1.25s/iter; A does not add to it.

### IPC mechanism: multiprocessing + shared memory

- **Big payload (state, logits)**: shared-memory tensors. Either
  `torch.multiprocessing` with `tensor.share_memory_()` and a pre-allocated
  buffer pool indexed by slot id, or `multiprocessing.shared_memory.SharedMemory`
  with manual slot management. Pickle is bypassed for the data itself.
- **Control messages (slot index, request id)**: small `Queue`. Pickle still
  happens here but only for ints/tuples, which is sub-microsecond and
  negligible against ~90 μs GPU forward.
- The naive path (`multiprocessing.Queue(tensor)` with default pickle) is the
  one that is slow and is what causes the "Python IPC is slow" reputation.
  With shared memory, multiprocessing IPC is effectively on par with thread
  shared-memory access for tensor traffic.

### Why not Cython `nogil` + threading instead of multiprocessing

Threading would avoid IPC entirely, but it requires the game-engine hot path
to be genuinely `nogil`-clean — no Python objects touched anywhere on the path.
Whether the existing Cython traversal qualifies is unknown and likely no:
auditing and migrating it to be fully `nogil`-clean is a substantial,
high-risk change to existing code, contradicting A's "mostly additive"
property. Additional drawbacks: a single segfault kills all threads;
multi-threaded CUDA usage has subtle context-sharing pitfalls; tooling and
prior art are weaker than for the multiprocessing pattern. Revisit only after
free-threaded Python (PEP 703) stabilizes or if a future profile shows the
shared-memory IPC is itself the limiter.

### Implementation plan

1. Prototype A on the 6-core / 3090 host with a single worker: validate
   end-to-end correctness and measure IPC round-trip latency vs GPU forward.
2. Scale to multiple workers; tune `batch_window_us`, `max_batch`, and
   `sync_every` (weight push frequency).
3. Deploy to the 32-core / weak-GPU remote and confirm CPU-side scaling holds
   and the weak GPU is still the right place to keep the model.
4. Defer C. Re-evaluate only if A's measurements show GPU forward is no longer
   on the critical path and game-logic CPU cost dominates — in that case the
   right next step is C with `nogil` threading, not plain C.

## Post-A Optimization Calculus (forward-looking, 2026-05-07)

Once Option A lands, the bottleneck shape changes. This section records the
expected sequencing for follow-up work. It is forward-looking and has not been
measured yet — verify against bench numbers after A is benchmarked.

### Why compile / TensorRT are negligible *today* but become meaningful later

Today (small model: 3-layer, 512 hidden):

- `torch.compile` on the trainer's networks already regressed (see the
  2026-05-07 experiment above). The model is too small for kernel fusion to
  beat compile dispatch overhead.
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
   (see the amortized eval table earlier in this doc). Eval is pure inference,
   so TensorRT on the inference-server's forward path applies directly.

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

### Tooling split

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

### Recommended sequencing

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

## Option A Bench Result and Structural Ceiling (2026-05-07)

Option A (`traversal.inference_backend: server`) was implemented and
benchmarked. **Result: regression. A is deferred. `default.yaml` stays on
`local`. The implementation is preserved behind the flag for future revisit.**

### Bench numbers

`scripts/bench_inference_backend.py --device cuda --iterations 5 --warmup 1`,
RTX 3090, after a per-call IPC fix (replaced `multiprocessing.Manager()`
queues/events with spawn-context primitives, slot reuse per worker batch,
shared memory confirmed in use for state/response payloads).

| Backend | iter | traversal | adv_train | strat_train | mem_add | batch_tensor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local` | 16.75s | 10.81s | 3.91s | 2.02s | 0.95s | 3.56s |
| `server` | 57.61s | 51.61s | 3.96s | 2.02s | 0.50s | 3.57s |
| Speedup | 0.29× | 0.21× | 0.99× | 1.00× | 1.89× | 1.00× |

Raw: `runs/bench/2026-05-07_193335_inference_backend/results.json`.

Training and eval phases are unchanged (as expected — A only touches the
traversal forward path). The regression is contained in `traversal_seconds`,
which is ~5× worse.

### Diagnosis

The server emits per-flush batch stats. **Mean batch size: ~7.2–7.9, max 8.**
This is the structural ceiling, not a tunable misconfiguration:

- Traversal recursion is **sync-blocking** at the policy call site. Each
  worker has at most one in-flight policy request at a time.
- In-flight requests at the server ≤ `num_workers` = 8.
- The server further splits each batch by `(network_kind, network_index)`,
  so the actual GPU forward group size is roughly half of that — about 4
  rows per group.

Per-state cost at this realized batch size, from the GPU profile table:

| Realized batch | μs/state |
| ---: | ---: |
| 1 | 80.07 |
| 4 | 20.30 |
| 8 (extrapolated) | ~12 |
| 64 | 1.46 |
| 256 | 0.34 |

So the GPU is doing ~12μs per state instead of the projected ~1.5μs at
bs=64. The IPC round-trip per call (queue post + server scheduler + event
wakeup, even with shared-memory payload) is on the order of hundreds of μs
per call, which exceeds both the local CPU forward (~80–200μs at bs=1 on
this small MLP) and the marginal GPU gain. Net: per-call cost roughly
doubles or triples, compounded across ~205k calls/iter, gives the observed
5× traversal regression.

`batch_window_us` and `max_batch` tuning cannot escape this ceiling —
there are simply not 64 concurrent in-flight requests to coalesce when only
8 workers are blocking-sync.

### What this means for the headline GPU profile (`scripts/profile_gpu_forward.py`)

The earlier "230× speedup at bs=256" is a **per-state GPU forward**
microbenchmark, not an end-to-end traversal speedup. Realizing that gain
requires *actually feeding the GPU* with bs=64+ batches. Sync-blocking
multi-worker traversal cannot do that. Reaching the bs=64 regime needs
either:

- Per-worker traversal interleaving (worker advances `worker_chunk_size`
  traversals concurrently, suspending at each policy call — Option B
  shape), which requires turning Cython traversal recursion into a
  resumable state machine. Same scope as a partial Option C, localized to
  worker scope.
- Option C proper (single-process vectorized traversal).

Both require restructuring traversal. Option A's "additive, no traversal
changes" property turned out to also mean "cannot drive the batch size up."

### Clarifying the traversal bottleneck: sync policy boundary, not SIMD

The tempting shorthand is "Python/GIL prevents traversal from using SIMD or
threads." The more precise diagnosis is narrower:

- Lost Cities game mechanics are already mostly Cython C-level operations.
  `legal_actions`, action push/pop, and cached scoring are not Python list
  walks on the hot path.
- The traversal recursion is Cython, but it synchronously crosses back into
  Python/PyTorch at every policy-needed state: encode a single info state,
  run one-row PyTorch forward, copy logits back to CPU/Numpy, then continue
  recursion.
- This boundary makes every traversal worker **sync-blocking**. With
  `num_workers=8`, the inference server can see at most eight in-flight
  requests before per-network splitting, no matter how large `max_batch` is.
- GIL-free threading would help only after the same path is made
  `nogil`-clean or after traversal is restructured so policy calls can be
  batched. Simply "using SIMD" does not address the one-row policy boundary.

So the actionable bottleneck is **policy-call scheduling shape**, not scalar
game-rule arithmetic. The highest-leverage experiment is Option B:
per-worker interleaved traversal, where one worker advances many traversals,
suspends each at a policy request, batches those requests, and resumes the
corresponding continuations.

Microbench evidence (2026-05-07, `configs/deep_cfr/default.yaml`,
`experiments/traversal_policy_boundary/bench_policy_boundary.py`):

| Device | Component | median μs/call | p95 μs/call |
| --- | --- | ---: | ---: |
| CPU | encode + legal | 3.10 | 3.81 |
| CPU | push + pop | 0.15 | 0.22 |
| CPU | policy boundary bs=1 | 111.50 | 125.46 |
| CPU | torch forward bs=64 | 12.84 | 13.14 |
| CUDA | policy boundary bs=1 | 181.30 | 194.77 |
| CUDA | torch forward bs=64 | 2.55 | 2.75 |

This confirms the bottleneck is not Cython game-rule scalar work. The
single-request policy boundary is ~36× larger than encode+legal on CPU, while
CUDA bs=64 forward is ~71× cheaper than the current CUDA bs=1 boundary.

Expected upside is bounded by the fraction of traversal currently spent at
policy calls. Moving realized GPU forward from the current ~4-8 row regime
(~12-20μs/state) to bs=64 (~1.46μs/state) is an ~8-14× improvement on the
forward component, but not on game recursion, sample creation, or replay
writes. For the observed `local` traversal around 10-13s/iter, a realistic
first target is roughly **1.5-3× traversal speedup** if Option B reaches the
bs=64 regime without adding comparable scheduler overhead. Larger claims need
a prototype because traversal has substantial non-forward work.

### Why deferring A (not deleting) is the right call

- The plumbing (server process, shared-memory client, weight sync, config
  flag) is complete and tested. Re-enabling is a config flip.
- The fundamental issue at this model size is that **GPU forward time is
  too small to amortize IPC overhead** at any realistic batch size we can
  drive without restructuring traversal. Bigger model changes that
  arithmetic; the same plumbing then becomes useful.
- The `mem_add_seconds` row showed a real 1.89× win, suggesting the
  shared-memory replay-write path adopted along the way is worth keeping
  even with `local` backend. (Confirm separately; this is a side effect.)

### Re-enable A when one of these holds

1. **Model grows** to ~1024 hidden / ~6 layers (compile/TRT discussion
   above). Forward time scales with FLOPs while IPC overhead is fixed; at
   some point IPC becomes a small fraction.
2. **Per-worker interleaved traversal** ships (Option B-shape refactor).
   Drives realized batch toward 64 and reclaims the profile table's gains.
3. **Eval becomes the dominant phase** (`eval_every: 5`,
   `evaluation.games: 1000`). Eval is not sync-blocking traversal; it is
   already batch_size=64 in eval code. The same inference server can
   serve eval directly without the worker-side ceiling.

### Free-threaded Python (3.13t) note

Free-threaded Python + Cython `nogil` would let many threads (well above
core count) run game logic concurrently in one process, with shared memory
and no IPC. With ~64 threads sync-blocking on policy calls, the server
would naturally see bs=64. **In principle this is the cleanest endpoint.**

In practice as of early 2026:

- Free-threaded Python is an opt-in build (`python3.13t`), still
  experimental, with measurable single-thread overhead.
- PyTorch's free-threaded compatibility is partial.
- Cython `nogil`-cleanliness audit on the existing game engine is still
  required and was the original reason `nogil` threading was deferred in
  the design decision above.
- No project-level adoption pressure on `python3.13t` today.

So free-threaded Python does change the architectural answer, but it does
not unblock A *now*. Track the ecosystem; revisit when (a) `python3.13t`
becomes mainstream or (b) the Cython engine is `nogil`-cleaned for other
reasons.
