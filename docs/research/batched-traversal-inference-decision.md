# Batched Traversal Inference: Design Decision (A vs B vs C)

**Last verified:** 2026-05-07
**Source:** `docs/performance.md` § "Batched Traversal Inference: Design Decision"
(extracted to `docs/research/` on 2026-05-08).
**See also:** `docs/archive/option-a-bench-result-2026-05-07.md` for the
post-implementation bench, which deferred Option A. The design rationale
below is captured *as it stood at decision time* — Option A's eventual
regression sharpens but does not invalidate the framework: the bench
later showed that the sync-blocking policy boundary, not the IPC
plumbing, was the actual ceiling.

## Problem

Traversal currently calls the policy network one row at a time. The GPU
profile (`scripts/profile_gpu_forward.py`) shows >230× per-state
speedup at bs=256 vs bs=1, with `~368` policy-needed states per
traversal as the available supply. We need a structural change to
reach that batch regime.

## Three structural options

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

## Decision: A

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

## IPC: what crosses the process boundary

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

## IPC mechanism: multiprocessing + shared memory

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

## Why not Cython `nogil` + threading instead of multiprocessing

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

## Implementation plan (as of decision time)

1. Prototype A on the 6-core / 3090 host with a single worker: validate
   end-to-end correctness and measure IPC round-trip latency vs GPU forward.
2. Scale to multiple workers; tune `batch_window_us`, `max_batch`, and
   `sync_every` (weight push frequency).
3. Deploy to the 32-core / weak-GPU remote and confirm CPU-side scaling holds
   and the weak GPU is still the right place to keep the model.
4. Defer C. Re-evaluate only if A's measurements show GPU forward is no longer
   on the critical path and game-logic CPU cost dominates — in that case the
   right next step is C with `nogil` threading, not plain C.

## Outcome (post-bench, see archive)

The plan was executed and Option A was benchmarked. Result: 0.21×
traversal regression because the *sync-blocking* policy boundary capped
realized batch size at ~7 (not the IPC plumbing). The structural
ceiling diagnosis and the criteria for revisiting A live in
`docs/archive/option-a-bench-result-2026-05-07.md`. Option B
(per-worker interleaved traversal) became the production path because
it actually drives realized batch size up by suspending traversals at
each policy call.
