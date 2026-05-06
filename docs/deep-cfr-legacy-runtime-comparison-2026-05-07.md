# Deep CFR Legacy Runtime Comparison 2026-05-07

This note records a runtime summary from the older `../coolrl` Lost Cities
Deep CFR implementation and compares it with the current profiling runs in
this repository.

## Legacy Run Summary

The older run completed metrics through iteration 387. The process had stopped
before completing iteration 388.

Total elapsed time through iteration 387:

`5879.17s`, or about `1h 37m 59s`.

| Segment | Mean | Median | Note |
| --- | ---: | ---: | --- |
| All iterations | 15.19s/iter | 11.44s | Includes eval iterations |
| Non-eval iterations | 11.51s/iter | 11.32s | Normal training iteration |
| Eval iterations | 30.00s/iter | 29.31s | Eval every 5 iterations |
| Evaluation only | 18.61s/eval | 18.04s | Early evals were slower |
| Traversal | 7.16s/iter | 6.98s | 140 traversals/iter |
| Advantage train | 2.81s/iter | 2.78s | Player 0 + player 1 |
| Strategy train | 1.44s/iter | 1.44s | |
| Overall throughput | 5387 nodes/s | 5437 nodes/s | |
| Traversal throughput | 19.8 traversals/s | 20.1 traversals/s | |

Recent 50 iteration window from that run:

| Segment | Mean |
| --- | ---: |
| All iterations | 15.57s/iter |
| Non-eval iterations | 12.21s/iter |
| Recent 20 evals, eval only | 17.31s/eval |
| Recent 20 eval iterations | 29.30s/iter |

Evaluation ran every 5 iterations: 5, 10, 15, ..., 385.

The practical legacy cadence was roughly:

`4 normal iterations + 1 eval iteration ~= 75s per 5 iterations`.

## Current Repo Reference Points

From `docs/deep-cfr-profile-advantage-memory-split-2026-05-07.md`:

| Segment | Current mean |
| --- | ---: |
| Non-eval iterations | 5.832958s/iter |
| Traversal | 3.160975s/iter |
| Advantage train | 1.742895s/iter |
| Strategy train | 0.912324s/iter |

From `docs/deep-cfr-batched-evaluation-2026-05-07.md`:

| Segment | Current value |
| --- | ---: |
| Batched CUDA evaluation | 14.834096s/eval |
| Batched CUDA 1-iter wall time with eval | 21.322356s |
| Batched opponent-parallel CUDA evaluation | 6.420402s/eval |
| Batched opponent-parallel CUDA 1-iter wall time with eval | 12.854383s |

## Rough Comparison

Normal training iterations improved from about `11.51s` to about `5.83s`,
roughly `1.97x` faster.

Traversal improved from about `7.16s` to about `3.16s`, roughly `2.27x`
faster.

Evaluation improved from about `18.61s` to about `14.83s`, roughly `1.25x`
faster for the measured batched CUDA profile. With opponent-parallel eval, the
measured eval time was about `6.42s`, roughly `2.90x` faster than the legacy
eval-only average.

Using the simple cadence model:

Legacy:

`4 * 11.51 + (11.51 + 18.61) = 76.16s per 5 iterations`

Current batched sequential:

`4 * 5.83 + (5.83 + 14.83) = 43.98s per 5 iterations`

Current batched opponent-parallel:

`4 * 5.83 + (5.83 + 6.42) = 35.57s per 5 iterations`

That implies about `1.73x` faster eval-included wall time for the batched
sequential rough comparison, and about `2.14x` faster for the batched
opponent-parallel rough comparison.

## Caveat

The legacy numbers came from a long run through iteration 387. The current
numbers are from targeted profiling runs. The comparison is useful for order of
magnitude and bottleneck direction, not as a strict benchmark under identical
runtime conditions.
