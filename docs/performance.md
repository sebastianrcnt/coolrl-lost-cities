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
