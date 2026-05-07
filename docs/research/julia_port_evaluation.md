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
| 1       | 0.223   | 1.00×   | 100%       |
| 2       | 0.130   | 1.71×   | 85%        |
| 4       | 0.097   | 2.29×   | 57%        |
| 8       | 0.093   | 2.40×   | 30%        |

The light workload is too small to settle the question: 1T iter time is
only ~0.2 ms, so thread dispatch overhead can dominate.

Heavy mode keeps the same tree and algorithm but increases traversals
per iteration from 1000 to 50000 (50× work). This raises 1T iter time
to 8.272 ms.

| threads | iter ms | speedup | efficiency |
| ---:    | ---:    | ---:    | ---:       |
| 1       | 8.272   | 1.00×   | 100%       |
| 2       | 5.058   | 1.64×   | 82%        |
| 4       | 2.602   | 3.18×   | 79%        |
| 8       | 1.739   | 4.76×   | 59%        |

**Verdict on criterion 3:** PARTIAL. Heavy 8T efficiency is 59%.
Dispatch overhead was a significant part of the light-mode result, but
the heavier workload still does not reach near-linear 8-thread scaling.
Julia delivers useful throughput scaling (4.76× at 8T), but this is not
the decisive PASS threshold for the threading criterion.

## Open evidence (criteria 4, 5)

- **Flux.jl + CUDA.jl MLP forward at bs={1, 64, 256}.** Compare to
  PyTorch numbers in `docs/performance.md`. Criterion 4. Not started.
- **Real-game-state slice port.** Port `play_card` + scoring, run on a
  fixed corpus of game states, compare to current Cython. Criterion 5.
  Not started.

## Decision posture

Promising but not enough to justify a port yet. The completed benchmarks
remove the main risk (GC under recursion) and confirm compute parity.
Heavy thread scaling upgrades criterion 3 from inconclusive to PARTIAL:
8T is 4.76× faster than 1T, but 59% efficiency is below the near-linear
PASS threshold.

This means Julia remains a credible option, but not a slam dunk. ML-stack
and game-state evidence (criteria 4, 5) must be positive before starting
a serious port plan. If those are positive, criterion 3 should be revisited
on a real traversal slice where each thread has substantially more work
than this toy benchmark.

Do not commit to porting on the current evidence alone.
