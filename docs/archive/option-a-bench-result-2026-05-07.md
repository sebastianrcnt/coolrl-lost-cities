# Option A Bench Result and Structural Ceiling (2026-05-07)

**Source:** Extracted from `docs/performance.md` § "Option A Bench
Result and Structural Ceiling" on 2026-05-08.

Records the post-implementation Option A benchmark (regression: 0.21×
traversal), the diagnosis of the sync-blocking policy boundary as the
real ceiling, and the criteria under which to revisit Option A.

**Related:**
- Design rationale: `docs/research/batched-traversal-inference-decision.md`
- Forward-looking sequencing (pre-bench): `docs/archive/post-a-optimization-calculus-2026-05-07.md`
- Trainer-side experiments from the same date: `docs/archive/deep-cfr-performance-experiments-2026-05-07.md`

---

Option A (`traversal.inference_backend: server`) was implemented and
benchmarked. **Result: regression. A is deferred. `default.yaml` stays on
`local`. The implementation is preserved behind the flag for future revisit.**

## Bench numbers

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

## Diagnosis

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

## What this means for the headline GPU profile (`scripts/profile_gpu_forward.py`)

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

## Clarifying the traversal bottleneck: sync policy boundary, not SIMD

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

## Why deferring A (not deleting) is the right call

- The plumbing (server process, shared-memory client, weight sync, config
  flag) is complete and tested. Re-enabling is a config flip.
- The fundamental issue at this model size is that **GPU forward time is
  too small to amortize IPC overhead** at any realistic batch size we can
  drive without restructuring traversal. Bigger model changes that
  arithmetic; the same plumbing then becomes useful.
- The `mem_add_seconds` row showed a real 1.89× win, suggesting the
  shared-memory replay-write path adopted along the way is worth keeping
  even with `local` backend. (Confirm separately; this is a side effect.)

## Re-enable A when one of these holds

1. **Model grows** to ~1024 hidden / ~6 layers (compile/TRT discussion in
   `docs/archive/post-a-optimization-calculus-2026-05-07.md`). Forward
   time scales with FLOPs while IPC overhead is fixed; at some point IPC
   becomes a small fraction.
2. **Per-worker interleaved traversal** ships (Option B-shape refactor).
   Drives realized batch toward 64 and reclaims the profile table's gains.
3. **Eval becomes the dominant phase** (`eval_every: 5`,
   `evaluation.games: 1000`). Eval is not sync-blocking traversal; it is
   already batch_size=64 in eval code. The same inference server can
   serve eval directly without the worker-side ceiling.

## Free-threaded Python (3.13t) note

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
  the design decision.
- No project-level adoption pressure on `python3.13t` today.

So free-threaded Python does change the architectural answer, but it does
not unblock A *now*. Track the ecosystem; revisit when (a) `python3.13t`
becomes mainstream or (b) the Cython engine is `nogil`-cleaned for other
reasons.
