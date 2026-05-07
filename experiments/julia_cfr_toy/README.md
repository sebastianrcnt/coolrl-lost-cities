# Julia CFR Toy Benchmark

This experiment compares Julia and Cython on a CFR-shaped hot path: recursive
tree traversal, mutable regret state, branch-heavy legal-action logic, and
counterfactual updates. The toy tree is not a game.

Run Julia only:

```bash
tools/julia/current/bin/julia experiments/julia_cfr_toy/bench_cfr.jl
```

Run Cython plus Julia/Cython parity and comparison:

```bash
uv run python experiments/julia_cfr_toy/bench_cfr_runner.py
```

Run Julia thread-local scaling:

```bash
tools/julia/current/bin/julia --threads=8 experiments/julia_cfr_toy/bench_cfr_threaded.jl
```

The Python runner builds `bench_cfr.pyx` in place when needed, runs the Julia
benchmark in JSON mode, then aborts if the final root regret vectors differ by
more than `1e-9`.

Interpretation:

- `ratio < 1.0`: Julia is faster than Cython for this toy pattern.
- `ratio > 1.0`: Julia is slower than Cython for this toy pattern.
- Julia `gc share` near zero means mutable-state recursion is not creating
  meaningful garbage in this benchmark.

## Results (2026-05-07)

Single run on the host's Julia 1.11.9 + Cython build. 100 iterations × 1000 traversals.

| Lang   | iter mean (ms) | total (s) | alloc (MB) | gc time (s) | gc share |
| ---    | ---:           | ---:      | ---:       | ---:        | ---:     |
| Julia  | 0.21           | 0.02      | 0.0        | 0.00        | 0.0%     |
| Cython | 0.41           | 0.04      | 21.3       | n/a         | -        |
| ratio  | 0.53×          | -         | -          | -           | -        |

Root regret parity verified to ε ≤ 1e-9.

**Interpretation:** Julia ~1.9× faster than this Cython implementation on the
CFR-shape hot path (recursive traversal + mutable regret state +
branch-heavy legal-action logic). Zero allocation, zero GC time on the
Julia side — escape analysis eliminates heap traffic when the code is
type-stable. The GC-pause concern that has been the main argument
against Julia adoption is not realized in this pattern.

**Caveats:**
- Cython 21.3 MB alloc suggests room for a more aggressively typed
  implementation (memoryviews end-to-end). A best-effort Cython could
  narrow the gap to roughly 1.3×–1.9×.
- Toy is not a game. Real Lost Cities CFR has larger state, replay
  buffer interactions, and an existing Cython implementation already
  optimized over time.
- Single seed, single run. Variance unmeasured.
## Thread Scaling (2026-05-07)

Thread-local trees, same 100 iterations × 1000 traversals total work. Each
thread processes its own chunk, then root regrets are reduced. Each threaded
case is checked against a sequential run with the same chunking.

| threads | iter ms | total s | μs/trav | alloc MB | gc s | gc share | speedup | efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.219 | 0.0219 | 0.219 | 0.000 | 0.000 | 0.0% | 1.00× | 100% |
| 2 | 0.129 | 0.0129 | 0.129 | 0.004 | 0.000 | 0.0% | 1.69× | 85% |
| 4 | 0.096 | 0.0096 | 0.096 | 0.004 | 0.000 | 0.0% | 2.28× | 57% |
| 8 | 0.090 | 0.0090 | 0.090 | 0.004 | 0.000 | 0.0% | 2.44× | 31% |

**Interpretation:** 8T efficiency 31% — contention/dispatch overhead dominates
at this tiny per-thread workload. Julia still improves absolute throughput, but
this is not near-linear scaling. The next scaling test should increase per-thread
work before treating this as a hard limit.
