# Julia Port Evaluation

Tracks evidence for and against porting the Deep CFR training pipeline
from Python/Cython to Julia. Plot/game/parity stays Python regardless;
the candidate scope is the training stack (traversal, networks,
inference).

## Why we are even considering this

`docs/performance.md` "Option A Bench Result and Structural Ceiling"
established that the current sync-blocking traversal in Python
multiprocessing caps realized batch size at `num_workers`. Escaping
that ceiling requires either restructuring traversal (Option B/C) or
moving to a runtime where threads can carry many concurrent traversals
in one process. Julia is the most credible candidate for the latter
(Mojo too immature, free-threaded CPython requires nogil-cleaning our
existing Cython — see `docs/reports/cost_*` triplet).

Decision criteria for going forward:

1. Single-thread compute parity with current Cython (or better).
2. GC behavior under tight CFR-shape recursion is acceptable (low
   pause time, low share).
3. Multi-thread scaling on the same CFR-shape pattern is near-linear,
   demonstrating the GIL-free promise actually holds.
4. ML stack (Flux.jl + CUDA.jl) covers our needs (MLP forward, AD,
   GPU). Our model is small and standard.
5. A real-game-state slice can be ported and compared head-to-head.

## Evidence so far

### 2026-05-07 — Safe heuristic single-thread parity (criterion 1)

Path: `experiments/julia_safe_heuristic/`.

1,838 snapshots in 157.471 ms median (~85.6 μs/call). Action-sequence
parity vs Python. Same order of magnitude as the Cython port of the
same bot (Cython gives ~2.55× over original Python on a 200-game
eval).

**Verdict on criterion 1:** Pass for isolated single-call work.

### 2026-05-07 — CFR-shape recursion toy (criteria 1, 2)

Path: `experiments/julia_cfr_toy/`. See that directory's README for
the full table.

Headline: Julia ~1.9× faster than Cython, 0 MB allocation, 0% GC time
on the hot path. Root regret parity ε ≤ 1e-9.

**Verdict on criterion 1:** Pass. Julia matches or beats Cython on
the CFR-shape pattern.

**Verdict on criterion 2:** Strong pass. Type-stable code produces
zero heap traffic. The GC concern that was the main argument against
Julia adoption did not materialize here.

Caveats: Cython 21.3 MB alloc suggests room for tighter typing;
best-effort Cython could narrow the gap. Toy is not a game.

### 2026-05-07 — CFR-shape multi-thread scaling on the same toy (criterion 3)

Path: `experiments/julia_cfr_toy/` (`bench_cfr_threaded.jl`). Same
100 iters × 1000 traversals total work split across threads. Thread-
local trees, root regrets reduced at the end.

| threads | iter ms | speedup | efficiency |
| ---:    | ---:    | ---:    | ---:       |
| 1       | 0.219   | 1.00×   | 100%       |
| 2       | 0.129   | 1.69×   | 85%        |
| 4       | 0.096   | 2.28×   | 57%        |
| 8       | 0.090   | 2.44×   | 31%        |

**Verdict on criterion 3:** Partial / inconclusive. 8T efficiency is
31% — contention or dispatch overhead dominates at this toy's small
per-thread workload. Julia does improve absolute throughput (2.44×
wall-clock at 8T vs 1T), but not near-linearly. The next scaling test
must increase per-thread work (larger tree or more traversals per
chunk) before a hard conclusion can be drawn. Do not treat this as a
hard ceiling — the toy may simply be too small for 8 threads to amortize
dispatch cost.

## Open evidence (criteria 3, 4, 5)

- **Multi-thread scaling with heavier per-thread work.** Increase tree
  depth or traversals-per-chunk and re-run the 1/2/4/8 thread sweep.
  Current 31% at 8T is likely a toy-size artifact, not a Julia limit.
  This remains the decisive test for the porting decision.
- **Flux.jl + CUDA.jl MLP forward at bs={1, 64, 256}.** Compare to
  PyTorch numbers in `docs/performance.md`. Criterion 4. Not started.
- **Real-game-state slice port.** Port `play_card` + scoring, run on a
  fixed corpus of game states, compare to current Cython. Criterion 5.
  Not started.

## Decision posture

Strong but not yet sufficient. The three completed benchmarks remove the
main risk (GC under recursion) and confirm compute parity. Multi-thread
scaling shows 2.44× wall-clock at 8T but only 31% efficiency — the toy
workload is likely too small per thread to amortize dispatch costs, so
the result is inconclusive rather than negative. A heavier per-thread
workload must be tested before criterion 3 can be marked pass or fail.

After criterion 3 is settled, ML-stack and game-state evidence (criteria
4, 5) determine whether to start a serious port plan.

Do not commit to porting until criterion 3 is conclusively settled with
appropriate per-thread workload.
