# Research Notes Catalog

This file catalogs permanent research notes in `docs/research/`. `AGENTS.md` is
the workflow document for deciding where new documentation belongs.

## Deep CFR

- [Lost Cities Deep CFR selectivity ideas](lost_cities_selectivity.md) — Consolidates selectivity failure hypotheses, diagnostics, and interventions before expensive architecture changes. (2026-05-10)
- [Deep CFR 512x3 2000-iteration baseline analysis](deep-cfr-baseline-2000-analysis.md) — Finds advantage loss improving while evaluated average-policy quality degrades, requiring current-vs-average diagnostics. (2026-05-08)
- [Batched and parallel evaluation in Deep CFR](deep-cfr-batched-evaluation.md) — Batching games and parallelizing opponents cuts CUDA evaluation overhead after entropy stays on device. (2026-05-08)
- [Deep CFR evaluation profiling](deep-cfr-evaluation-profile-plan.md) — Defines runtime counters that separate policy inference, encoding, opponent logic, and engine costs. (2026-05-08)
- [Deep CFR evaluation performance](deep-cfr-evaluation-profile.md) — Shows serial batch-size-one CUDA evaluation is slower than CPU until evaluation is batched. (2026-05-08)
- [Deep CFR legacy parity and hyperparameter mapping](deep-cfr-legacy-experiment-reproduction.md) — Maps legacy features, architecture, and traversal knobs needed for meaningful reproduction comparisons. (2026-05-08)
- [Deep CFR runtime: legacy vs. current implementation](deep-cfr-legacy-runtime-comparison.md) — Attributes current speedups to Cython traversal, batched evaluation, and opponent-parallel execution. (2026-05-08)
- [Deep CFR performance optimization and scaling](deep-cfr-performance-experiments.md) — Concludes small models regress under AMP or compile; interleaved traversal is the real win. (2026-05-08)
- [Advantage memory split performance optimization](deep-cfr-profile-advantage-memory-split.md) — Splitting advantage reservoirs by player removes linear sample filtering and stabilizes iteration time. (2026-05-08)
- [Deep CFR performance profile: advantage training bottlenecks](deep-cfr-profile.md) — Identifies shared-memory player filtering as the original advantage-training scaling bottleneck. (2026-05-08)
- [Deep CFR regret matching fallback and early over-opening](deep-cfr-regret-fallback-audit.md) — Shows uniform all-negative fallback frequently fires early and over-samples expedition-opening actions. (2026-05-08)
- [Deep CFR reproducibility policy](deep-cfr-reproducibility-policy.md) — Sets debug versus research reproducibility expectations and required matched-seed reporting practice. (2026-05-08)
- [Deep CFR reproducibility](deep-cfr-reproducibility.md) — Traces same-seed multi-worker divergence to completion-order-sensitive reservoir insertion and sampling. (2026-05-08)
- [Deep CFR v0: architectural parity and performance gaps](deep-cfr-v0-gap-vs-coolrl.md) — Confirms functional parity while naming synchronous recursive policy inference as the scaling bottleneck. (2026-05-08)
- [Deep CFR Cython traversal architecture](deep-cfr-v0-plan.md) — Explains mutation-based Cython traversal, chance sampling, and non-leaking information-state encoding. (2026-05-08)
- [Option A benchmark and structural ceiling](option-a-bench-result.md) — Shows centralized inference regressed because sync-blocking recursion capped realized GPU batch size. (2026-05-08)
- [Post-A optimization calculus](post-a-optimization-calculus.md) — Defers compile and TensorRT until model scale or evaluation density makes inference compute-bound. (2026-05-08)
- [Deep CFR package architecture and design rationale](deep-cfr-architecture.md) — Documents the Cython/Python module split that follows traversal hot paths versus orchestration. (2026-05-07)
- [Batched traversal inference decision](batched-traversal-inference-decision.md) — Records the initial Option A rationale and why later benchmarks redirected work toward interleaving. (2026-05-07)
- [Deep CFR v0 subsystem coverage vs. legacy reference](deep-cfr-v0-feature-parity.md) — Establishes that remaining legacy gaps are tooling and performance, not core algorithm correctness. (2026-05-07)
- [Opponent policy: network vs. self-play league](opponent-policy-network-divergence.md) — Explains why a live network opponent collapses while snapshot leagues preserve stationarity. (2026-05-07)
- [Outcome-sampling MCCFR advantage target](outcome-sampling-target.md) — Defends zero unsampled-action targets as the textbook importance-weighted outcome-sampling estimator. (2026-05-07)
- [Regret-matching all-negative fallback](regret-matching-fallback.md) — Keeps uniform as the safe default while documenting argmax-tiebreak's unsettled early-training effects. (2026-05-07)
- [Strategy memory recording location](strategy-memory-location.md) — Concludes traverser-node strategy samples are fine for outcome sampling, but external sampling needs OpenSpiel flags. (2026-05-07)

## Engine / Performance

- [Classic game port architecture](classic-port-notes.md) — Describes the standalone Cython classic engine as the stable rules layer for all consumers. (2026-05-08)
- [Fast engine optimization architecture](fast-engine-next-optimizations.md) — Prioritizes C-level APIs, contiguous allocation, and zero-copy extraction for high-throughput RL. (2026-05-08)
- [Optimization sequencing](optimization_sequencing.md) — Orders runtime, traversal, model-scale, and inference optimizations to avoid invalidating experiments. (2026-05-07)

## Other

- [SO-ISMCTS BC ceiling](ismcts-bc-ceiling-2026-05-11.md) — Finds behavior cloning remains the ceiling under current search and compute budgets. (2026-05-11)
- [Julia port evaluation](julia_port_evaluation.md) — Rejects a full Julia port for now because GPU MLP inference misses the threshold. (2026-05-10)
- [Test coverage strategy for Python and Cython modules](test-coverage-notes.md) — Recommends Python-first coverage and isolated Cython tracing to protect performance artifacts. (2026-05-08)
