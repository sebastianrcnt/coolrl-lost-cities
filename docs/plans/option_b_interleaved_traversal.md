# Plan: Option B Per-Worker Interleaved Traversal

**Status:** Phase 4 long-run A/B is prepared. Default behavior is unchanged;
the interleaved path remains opt-in.
**Owner:** Codex for prototype design and implementation; operator for long-run
benchmarks on `home`.
**Background:** Option A, the central traversal inference server, was implemented
and benchmarked on 2026-05-07. It regressed traversal because the existing
recursive worker path is sync-blocking and can only feed batches near the worker
count, not the GPU-efficient bs=64+ regime. See `docs/performance.md`:
"Option A Bench Result and Structural Ceiling" and "Clarifying the traversal
bottleneck: sync policy boundary, not SIMD."

## Goal

Restructure traversal scheduling so each worker advances many traversal
instances concurrently, suspends each instance at policy-needed states, batches
the pending policy requests, runs one policy forward, and resumes the matching
instances.

The target is to turn the current policy-forward shape:

```text
one traversal -> policy request -> bs=1 forward -> resume
```

into:

```text
N traversal continuations -> collect policy requests -> bs=32..128 forward -> resume
```

without changing CFR math, game rules, replay sample semantics, or public
training CLI behavior.

## Why This Is The Next Optimization

Microbench evidence from `experiments/traversal_policy_boundary/`:

| Device | Component | median us/call |
| --- | --- | ---: |
| CPU | encode + legal | 3.10 |
| CPU | push + pop | 0.15 |
| CPU | policy boundary bs=1 | 111.50 |
| CUDA | policy boundary bs=1 | 181.30 |
| CUDA | torch forward bs=64 | 2.55 |

The game mechanics and encoding are not the dominant cost. The dominant cost is
the one-row Python/PyTorch policy boundary. Option A moved that boundary to a
server process, but because every worker blocks waiting for one response, the
server observed mean batches around 7-8 and regressed end-to-end. Option B is
the first design that directly changes the scheduling shape.

## Non-Goals

- Do not re-enable `traversal.inference_backend: server` as the default.
  Option A remains available but structurally capped until traversal can feed
  larger batches.
- Do not port traversal to Julia, C++, or a new game engine.
- Do not change Deep CFR sampling math, regret targets, strategy-memory
  location, weighting, or replay schema.
- Do not change model architecture.
- Do not implement TensorRT, `torch.compile`, or AMP in this plan.
- Do not remove the existing recursive traversal path until the interleaved path
  has parity and benchmark evidence.

## Design Sketch

The current Cython traversal is recursive and calls policy synchronously. Option
B needs an explicit continuation representation so policy calls become yield
points.

One worker owns a fixed set of active traversal contexts:

```text
TraversalContext
  GameState state
  explicit stack frames replacing recursion
  RNG state
  traverser player
  iteration
  partial node/action values
  pending info_state/legal mask
  output advantage/strategy samples
  TraversalStats
```

Worker loop:

1. Initialize `K` traversal contexts from the worker's assigned seeds.
2. Advance each context until it reaches one of:
   - terminal/cutoff/done,
   - needs policy forward,
   - error.
3. Collect pending policy requests into a batch, grouped by network target:
   advantage player 0/1, strategy network, or league snapshot if enabled.
4. Run batched forward for each group on the worker's selected inference device.
5. Scatter logits/advantages back into the contexts.
6. Resume contexts until all assigned traversals finish.
7. Return the same `(stats, advantage_samples, strategy_samples)` shape as
   `run_cython_traversal_batch`.

The first prototype should keep one worker process and one GPU model copy per
worker if `device=cuda`. That may duplicate VRAM across workers, so the initial
benchmark can run with fewer workers and larger `interleave_width`. A later
hybrid can combine Option B's continuation batching with Option A's central
server if VRAM pressure dominates.

## Config Surface

Add only after the prototype proves parity:

```yaml
traversal:
  scheduler: recursive        # recursive | interleaved
  interleave_width: 64        # traversal contexts advanced per worker
  interleave_max_batch: 128   # cap per forward group
```

Default remains `recursive`.

## Implementation Phases

### Phase 0: Design Spike

- Trace current `_traverse` control flow and enumerate every value that must
  survive across a policy yield point.
- Decide whether to implement the explicit stack in Cython (`.pyx`) or as a
  Python prototype first.
- Identify exact parity surfaces:
  `TraversalStats`, advantage samples, strategy samples, RNG sequence, and
  terminal/cutoff behavior.

Deliverable: short design note appended to this plan before code work starts.

### Phase 0 Design Note (2026-05-07)

The production `_traverse` yield point is the call to `_policy(...)`; all
state below must survive across that yield:

- current `GameState`, traverser, iteration, depth, and per-context RNG state,
- `TraversalStats`,
- policy metadata: `info_state`, legal mask, policy vector, fallback/tie
  metadata,
- selected sampled action, action probability, and any deck-draw chance swap
  index,
- child return value and the parent post-child computation state,
- pending advantage/strategy samples.

The first implementation target is an experiment-only Python prototype, not a
Cython production rewrite. It intentionally uses per-context RNG so interleaved
execution order does not change the random stream for another context. That
lets the prototype assert value/stat/sample parity against a recursive prototype
while measuring realized batch size. Production Cython parity is a later Phase 2
gate because the real path also has heuristic-balanced opponents, average-strategy
opponents, self-play league snapshots, deck-draw chance sampling, external
sampling, and cutoff rollouts.

### Phase 1: Python Prototype, No Production Wiring

- Add an experiment-only traversal prototype under `experiments/` that mimics
  the current traversal semantics with explicit stacks.
- Use small configs (`max_depth`, low traversal count) and compare samples/stats
  to the recursive path under fixed seeds.
- Measure realized batch size and scheduler overhead.

Success gate: sample/stat parity on small deterministic fixtures, and realized
policy batches materially above worker count.

### Phase 1 Prototype Result (2026-05-07)

Prototype location: `experiments/option_b_interleaved_traversal/`.

Command:

```bash
uv run python experiments/option_b_interleaved_traversal/prototype_interleaved.py \
  --traversals 64 \
  --interleave-width 32 \
  --max-depth 8 \
  --max-nodes 512 \
  --device cuda
```

Result on RTX 3090 host:

| Device | Mode | total s | forward s | scheduler s | batch mean | batch max | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | recursive | 0.084 | 0.054 | - | 1.0 | 1 | 1.00x |
| CPU | interleaved | 0.018 | 0.005 | 0.003 | 32.0 | 32 | 4.71x |
| CUDA | recursive | 0.195 | 0.144 | - | 1.0 | 1 | 1.00x |
| CUDA | interleaved | 0.028 | 0.014 | 0.003 | 32.0 | 32 | 6.98x |

Prototype parity: PASS for values, RNG outputs, aggregate traversal stats, and
sample checksum within float tolerance. This proves the scheduling shape can
form large policy batches. It does **not** yet prove production Cython parity.

### Phase 2: Non-Default Production Prototype

- Add an interleaved traversal entry point beside the existing recursive one.
- Keep the existing recursive path untouched and default.
- Wire through `workers.py` only behind `traversal.scheduler: interleaved`.
- Add focused tests for parity on smoke configs.

Success gate: `uv run pytest -q tests/games/classic/test_deep_cfr_trainer.py`
and new interleaved traversal tests pass.

### Phase 2 Result (2026-05-07)

Implemented as a Python explicit-stack production path in
`interleaved_traversal.py`, not a Cython rewrite. This is deliberate: the Phase
1 prototype proved the scheduling shape, while a full Cython state-machine
rewrite would duplicate a large fraction of `traversal.pyx` before we know that
default-config wall-clock speedup survives trainer integration. The recursive
Cython path remains untouched and default.

Config surface added:

```yaml
traversal:
  scheduler: recursive        # recursive | interleaved
  interleave_width: 64
  interleave_max_batch: 128
```

Current interleaved guardrails:

- `sampling_mode: outcome`
- `opponent_policy: network`
- `cutoff_value_mode: score_diff`
- `cutoff_rollouts: 0`
- `inference_backend: local`

Validation:

- single-traversal parity against `run_cython_traversal_batch`: PASS for
  aggregate stats and sample target checksums under identical RNG seed,
- trainer smoke with `traversal.scheduler=interleaved`: PASS,
- multiprocessing worker smoke via CLI: PASS,
- emitted runtime metrics:
  `interleaved/batches`, `interleaved/requests`,
  `interleaved/avg_batch_size`, `interleaved/max_batch_size`,
  `interleaved/scheduler_seconds`, and `interleaved/forward_seconds`.

Important parity note: multi-traversal interleaving uses per-context RNG streams
so request scheduling does not couple one traversal's random stream to another.
That means exact recursive-batch RNG ordering is intentionally not preserved
across multiple simultaneous traversals. Phase 3 must therefore gate on sample
counts, traversal stats, learning stability, and wall-clock speedup, not
byte-identical multi-traversal sample order.

### Phase 3: Benchmark

Benchmark against current `default.yaml`, eval/checkpoint disabled:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --set run.max_iterations=10 \
  --set checkpoint.save_latest=false \
  --set checkpoint.save_every=0 \
  --set evaluation.eval_every=0
```

Compare:

- recursive baseline,
- interleaved with `interleave_width` in `{16, 32, 64, 128}`,
- worker counts in `{1, 2, 4, 8}` as VRAM allows.

Metrics:

- `iteration_seconds`
- `traversal_seconds`
- realized policy batch size mean/p50/p95/max
- scheduler overhead if instrumented
- `advantage_train_seconds`, `strategy_train_seconds` to confirm no unrelated
  drift
- sample counts and traversal stats

Success gate: at least **1.5x traversal speedup** with no sample/stat parity
failure. Stretch target: **3x traversal speedup** if realized batches reach the
bs=64 regime without high scheduler overhead.

### Phase 3 Result (2026-05-07)

Default-config-scale A/B ran for 10 iterations, with evaluation and
checkpointing disabled and the first 2 iterations dropped as warm-up. Because
Phase 2 interleaved traversal currently supports only `opponent_policy:
network`, the recursive baseline used the same override.

Result paths:

- `runs/2026-05-07_225044_option-b-recursive-network-10i`
- `runs/2026-05-07_225342_option-b-interleaved-workers-10i`
- `runs/2026-05-07_225542_option-b-interleaved-cuda-single-10i`

| Mode | iter s | traversal s | traversal speedup | iter speedup | batch mean | batch max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recursive, 8 workers, chunk 8 | 16.49 | 10.22 | 1.00x | 1.00x | 1.0 | 1 |
| interleaved, 8 workers, chunk 64 | 10.81 | 4.92 | 2.08x | 1.53x | 35.2 | 64 |
| interleaved, single CUDA process | 11.51 | 5.35 | 1.91x | 1.43x | 36.7 | 64 |

Gate decision: PASS. The best current candidate is the 8-worker interleaved
path with larger worker chunks. The single-process CUDA path makes forward
cheap but gives up multiprocessing game-state throughput, so it is slower
end-to-end.

### Phase 4: Feature Expansion And Long-Run A/B Prep

Required feature expansion:

- Support `opponent_policy: average_strategy`, matching the default config's
  opponent branch.
- Keep unsupported branches guarded (`self_play_league`, `heuristic_balanced`,
  random rollout cutoffs, external sampling).
- Add parity tests for the average-strategy fixed-opponent branch.
- Verify a default-policy interleaved run starts and emits batch metrics.

Long-run A/B protocol:

- Baseline: `configs/deep_cfr/default.yaml` unchanged.
- Treatment: exactly one structural scheduler change plus the chunk/batch
  settings below.
- Run sequentially, not in parallel on the same GPU.
- Keep default evaluation/checkpoint cadence for the real A/B unless measuring
  pure throughput only.
- Stop and report if learning metrics drift materially despite wall-clock
  speedup.

Treatment command:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.experiment_name=option-b-interleaved-default-ab \
  --set traversal.scheduler=interleaved \
  --set traversal.num_workers=8 \
  --set traversal.worker_chunk_size=64 \
  --set traversal.interleave_width=64 \
  --set traversal.interleave_max_batch=128 \
  --set traversal.progress_every_traversals=0
```

### Phase 4 Result (2026-05-07)

Implemented `average_strategy` support in the interleaved scheduler. Opponent
nodes now use the strategy network as a fixed-opponent policy, matching the
Cython recursive path rather than recording CFR regret at those nodes.

Validation:

- average-strategy single-traversal parity against `run_cython_traversal_batch`:
  PASS,
- trainer smoke with `scheduler=interleaved` and `opponent_policy=average_strategy`:
  PASS,
- focused trainer test suite: PASS.

Default-policy throughput check:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.max_iterations=10 \
  --set run.experiment_name=option-b-interleaved-average-strategy-10i \
  --set traversal.scheduler=interleaved \
  --set traversal.num_workers=8 \
  --set traversal.worker_chunk_size=64 \
  --set traversal.interleave_width=64 \
  --set traversal.interleave_max_batch=128 \
  --set traversal.progress_every_traversals=0 \
  --set checkpoint.save_latest=false \
  --set checkpoint.save_every=0 \
  --set evaluation.eval_every=0
```

Result path: `runs/2026-05-07_230419_option-b-interleaved-average-strategy-10i`.
Warm-up-excluded means: `iteration_seconds=10.61s`,
`traversal_seconds=4.85s`, `interleaved/avg_batch_size=28.3`,
`interleaved/max_batch_size=64`.

## Risks

- **State-machine complexity.** Recursive CFR control flow has many local
  values. Mitigation: prototype with small depth and exhaustive parity before
  optimizing.
- **RNG drift.** Interleaving changes operation order. Mitigation: store RNG
  state per traversal context and define parity against the recursive path only
  where ordering is intentionally preserved. If exact ordering is impossible,
  require distributional/sample-count parity and document the break.
- **VRAM duplication.** Per-worker GPU models may not fit at larger model sizes.
  Mitigation: start with fewer workers and larger interleave width; revisit a
  central server only after Option B proves the scheduling benefit.
- **Sample memory pressure.** More active contexts mean more pending samples.
  Mitigation: stream completed samples out of contexts as soon as a traversal
  finishes.
- **Scheduler overhead cancels batching.** Mitigation: benchmark light and heavy
  modes; record realized batch size and overhead explicitly.

## Decision Tree

- **Parity fails in Phase 1/2:** stop. Do not optimize. Document the exact
  mismatch.
- **Parity passes, realized batch remains <16:** Option B did not change the
  structural ceiling enough. Reconsider Option C or a deeper traversal rewrite.
- **Parity passes, realized batch >=64, speedup <1.5x:** batching worked but
  non-forward work dominates. Keep recursive default and document.
- **Parity passes, traversal speedup >=1.5x:** keep interleaved behind config,
  run longer learning-curve A/B.
- **Longer A/B is stable and speedup persists:** consider making
  `traversal.scheduler: interleaved` the default.

## Definition Of Done

- Plan reviewed and Phase 0 design note added.
- Prototype proves whether explicit continuation batching can preserve traversal
  semantics.
- Bench results are added to `docs/performance.md`.
- Default behavior remains unchanged until parity and benchmark gates pass.
