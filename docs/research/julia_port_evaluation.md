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

## Open evidence (criterion 5)

- **Real-game-state slice port.** Port `play_card` + scoring, run on a
  fixed corpus of game states, compare to current Cython. Criterion 5.
  Not started. Skipped for now because criterion 4 failed; per the
  decision rule, the full Julia port is no longer a GO candidate on the
  current evidence.

### 2026-05-07 — Flux.jl + CUDA.jl MLP forward (criterion 4)

Path: `experiments/julia_flux_mlp/`.

Same `DeepCFRMLP` shape as the current Deep CFR default MLP:
365→512→512→512→22, ReLU, identical PyTorch-exported weights loaded
into Flux. Output parity passed with max absolute difference
`5.215e-08`. Timing uses 10 warmup forwards, then 100 timed forwards,
with CUDA synchronized around the timed loop in both runtimes.

| backend | batch | forward ms | μs/state | ratio vs PyTorch |
| ---     | ---:  | ---:       | ---:     | ---:             |
| PyTorch | 1     | 0.0829     | 82.8755  | 1.00×            |
| Flux    | 1     | 0.1669     | 166.8867 | 2.01×            |
| PyTorch | 64    | 0.0927     | 1.4477   | 1.00×            |
| Flux    | 64    | 0.1909     | 2.9836   | 2.06×            |
| PyTorch | 256   | 0.0877     | 0.3427   | 1.00×            |
| Flux    | 256   | 0.1758     | 0.6867   | 2.00×            |

**Verdict on criterion 4:** FAIL. bs=64 is ~2.06× slower than
PyTorch, outside the ±20% PASS band. bs=1 and bs=256 are also ~2×
slower, outside the ±30% bands. Criterion 5 was not run after this
FAIL because the full Julia-port decision rule is already blocked.

### 2026-05-07 — Torch.jl MLP forward retry (criterion 4)

Path: `experiments/julia_torch_mlp/`.

Torch.jl was tested as a possible replacement for Flux/CUDA on the same
DeepCFRMLP forward benchmark. The measurement could not start because
Torch.jl v0.1.3 fails during package load/precompile on this Julia
1.11.9 environment:

```text
UndefVarError: libtorch_c_api not defined in Torch.Wrapper
```

This occurs before model construction or timing, so there is no
Torch.jl forward result to compare against PyTorch.

**Verdict on criterion 4 after Torch.jl retry:** unchanged FAIL. Flux.jl
misses the performance threshold, and Torch.jl is blocked by package
load failure rather than providing a successful re-measurement.

## Pass/fail thresholds (decided in advance)

These are explicit so that the moment a measurement lands, the decision
is automatic. No re-deliberation, no "let's discuss it." If a result
sits on a boundary, treat it as the pessimistic side.

**Criterion 3 (multi-thread scaling)** — already measured on the toy:

- PASS: 8T efficiency ≥ 80%.
- PARTIAL: 50–80%. (Current toy result: 59%.)
- FAIL: < 50%.

Re-measure on the real-game-state slice (criterion 5) when that lands.
The toy figure is informative but not authoritative; real workloads have
more compute per traversal and may scale better.

**Criterion 4 (Flux.jl + CUDA.jl MLP forward)**:

- PASS: bs=64 forward time within ±20% of PyTorch on the same GPU and
  same `DeepCFRMLP` shape (input_dim=365, hidden=512, layers=3,
  output_dim=22). Both bs=1 and bs=256 must also be within ±30% (single-
  state and large-batch matter for traversal-time and eval-time
  respectively).
- FAIL: any of the three batch sizes regresses by more than the band
  above. Single PASS at one batch size is not sufficient.

Methodology: 100-iter timing with 10-iter warm-up, `inference_mode`
equivalent on both sides. Identical weights (export from PyTorch, load
into Flux). Compare per-state latency (μs/state).

**Criterion 5 (real-game-state slice)**:

- Scope: a Julia port of a single representative slice of the Lost
  Cities engine — at minimum `legal_actions` + `apply_action` + scoring
  for end-of-game, against a fixed corpus of ≥1000 game states exported
  from the current Cython engine.
- Action-equivalence requirement: byte-identical `legal_actions` set
  and post-action state for every corpus entry. ε = 0; this is a
  correctness gate, not a numerical one.
- PASS: single-thread per-state cost within ±25% of current Cython, AND
  re-running the heavy threaded benchmark on this slice (1/2/4/8T)
  yields 8T efficiency ≥ 75%. Both conditions required.
- FAIL: single-thread regresses > 50% vs Cython, OR 8T efficiency
  < 60% on the real slice.
- BORDERLINE: between FAIL and PASS — flag in the doc, do not start a
  port; consider whether a different scoping (port traversal recursion
  only, leave game engine in Cython) clears the threshold instead.

## Decision rule

After all of criteria 3 (re-measured on real slice), 4, 5 land:

- All three PASS → start a port plan. Deferred Option A re-enable, AMP,
  compile, TensorRT all get re-evaluated under the new runtime.
- Criterion 5 PASS but criterion 4 FAIL → consider a hybrid: keep
  PyTorch for networks via PythonCall.jl/PyCall, port traversal to
  Julia. Re-evaluate IPC/FFI cost separately.
- Any criterion FAIL beyond the borderline → **stay on Python/Cython,
  pursue Option B (per-worker interleaved traversal)** as the
  GIL-escape path instead. Document the FAIL result, close this thread.
- Criterion 5 BORDERLINE → reduce port scope (recursion-only) and
  re-test.

Cost-of-being-wrong asymmetry: a port is months of work; staying is
zero work but caps us at the current ceiling. So the bar to GO is
deliberately set above 50%; the bar to STAY is permissive. This is on
purpose.

## Decision posture

No full Julia port on the current evidence. The completed benchmarks
remove the main risk (GC under recursion) and confirm compute parity,
but criterion 3 is only PARTIAL and criterion 4 is FAIL. The Torch.jl
retry did not reverse criterion 4 because Torch.jl failed to load in
this Julia environment. Per the decision rule, a full port would spend
months to replace a PyTorch GPU path that is already ~2× faster than
Flux for the exact model shape we use.

Recommended next path: stay on Python/Cython and pursue Option B
(per-worker interleaved traversal) as the GIL-escape path. A narrower
Julia experiment could still be considered later for traversal-only
logic, but it would need an explicit hybrid plan that keeps PyTorch for
networks and separately proves PythonCall/PyCall overhead is acceptable.
For the traversal bottleneck clarification and Option B speedup envelope,
see `docs/performance.md` "Clarifying the traversal bottleneck: sync
policy boundary, not SIMD."

Criterion 5 remains unrun because criterion 4 already blocks the full
port decision.
