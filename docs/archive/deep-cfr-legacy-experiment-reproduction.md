# Deep CFR Legacy Experiment Reproduction Plan

This document tracks what is needed to reproduce the legacy `../coolrl`
experiment below with the same intended semantics in this repository:

`experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability`

The goal is not to run a similar Deep CFR configuration. The goal is to map the
legacy experiment's hyperparameters and feature semantics into this repository's
own config schema so that the training run means the same thing.

Exact legacy YAML compatibility is not required. It is acceptable to create a
new config file under this repository as long as every relevant legacy
hyperparameter is represented explicitly and the differences are documented.

Mapped config in this repository:

`configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml`

## Source Experiment

Legacy config path:

`/home/coolguy/dev/coolrl/experiments/lost_cities/deep_cfr_pure_self_play_zero_pit_poc_full_depth_slot_aware_playability/config.yaml`

Important legacy settings:

1. `seed: 79`
2. `max_hours: 4`
3. `max_iterations: null`
4. `device: CUDA`
5. `use_amp: false`
6. `rules.tier: tier3`
7. `encoding.derived_playability: true`
8. `encoding.slot_aware_playability: true`
9. `network.hidden_size: 256`
10. `network.num_layers: 3`
11. `network.activation: relu`
12. `traversal.traversals_per_player: 70`
13. `traversal.max_depth: null`
14. `traversal.max_nodes_per_traversal: 1000`
15. `traversal.opponent_policy: self_play_league`
16. `traversal.outcome_sampling_epsilon: 0.2`
17. `traversal.outcome_sampling_value_clip: 500`
18. `traversal.outcome_unsampled_regret: zero`
19. `optimization.advantage_batch_size: 1024`
20. `optimization.strategy_batch_size: 1024`
21. `optimization.advantage_updates_per_iteration: 256`
22. `optimization.strategy_updates_per_iteration: 256`
23. `optimization.learning_rate: 3.0e-5`
24. `optimization.weight_decay: 1.0e-4`
25. `optimization.grad_clip: 1.0`

## Required Reproduction Work

### Config

Represent the legacy hyperparameters in this repository's config schema:

1. Top-level:
   - `experiment_name`
   - `seed`
   - `max_iterations`
   - `max_hours`
   - `device`
   - `use_amp`
2. `rules`:
   - `tier`
   - optional direct `LostCitiesConfig` field overrides
3. `encoding`:
   - `derived_playability`
   - `slot_aware_playability`
4. `network`:
   - `hidden_size`
   - `num_layers`
   - `activation`
5. `traversal` semantics:
   - per-player traversal count
   - max nodes per traversal
   - traversal worker chunk size
   - self-play league settings
6. `optimization` semantics:
   - separate advantage and strategy batch sizes
   - separate advantage and strategy update counts
   - `weight_decay`
   - `grad_clip`
7. `evaluation.on_max_steps`
8. `checkpoint`:
   - `save_iteration_interval`
   - `save_latest_only`
   - `progress_interval_seconds`

### Encoding

Port the legacy feature semantics:

1. Base information-state encoding must remain deterministic.
2. Add `derived_playability` color-level features.
3. Add `slot_aware_playability` hand-slot action-local features.
4. Preserve feature order and dimensions from legacy where possible.
5. Add tests for input dimension and known-state feature values.

The slot-aware block is the core of the source experiment. Without this block,
the reproduction is not meaningful.

### Network

Make the MLP configurable:

1. `hidden_size`
2. `num_layers`
3. `activation`

The legacy experiment uses a 3-layer ReLU MLP with hidden size 256.

### Optimization

Match the legacy training knobs:

1. Separate advantage and strategy batch sizes.
2. Separate advantage and strategy update counts per iteration.
3. Adam `weight_decay`.
4. Gradient clipping.

### Traversal

Match legacy traversal semantics and metrics:

1. Per-player traversal count behavior.
2. Full-depth traversal when `max_depth: null`.
3. Max nodes per traversal.
4. Self-play league behavior.
5. Endpoint depth accounting:
   - endpoint depth sum
   - endpoint depth buckets
   - depth bucket width
   - depth bucket max
6. Optional worker progress logging.
7. Optional hotspot profiling.

### Run Loop And Checkpointing

Match legacy runtime behavior:

1. Stop by `max_hours`.
2. Stop by `max_iterations` when set.
3. Allow `max_iterations: null`.
4. Save every N iterations through `save_iteration_interval`.
5. Support `save_latest_only`.
6. Preserve useful config artifacts in the run directory.
7. Keep `metrics.jsonl`, `runtime_progress.json`, and `train.log` useful for
   comparing this run against the legacy report. Exact artifact layout
   compatibility is optional.

### Evaluation And Metrics

Match the legacy evaluation configuration:

1. `evaluation.on_max_steps`.
2. Opponent list:
   - `random`
   - `passive_discard`
   - `safe_heuristic`
   - `safe_heuristic_loose`
   - `safe_heuristic_strict`
   - `noisy_safe`
3. Metric suffixes used by the legacy analysis scripts, including opening
   quality, opened-color distribution, discard take rates, expedition quality,
   policy entropy, and timeout counts.
4. Preserve comparable semantics for the legacy experiment's decision criteria.
5. Exact metric key/schema compatibility is optional unless legacy `analyze.py`
   is reused directly.
6. Confirm whether the current classic evaluator emits every required core
   metric.
7. Add a small adapter or mapping table if this repository keeps different
   metric names.

### Analysis Compatibility

The legacy `analyze.py` expects specific run artifacts and metric keys. Direct
reuse is optional. The required goal is that the generated run data can be
compared against the legacy report through documented metric semantics:

1. Core metric meanings are documented.
2. Latest iteration and latest eval iteration can be inferred.
3. Endpoint-depth metrics are present or explicitly marked as omitted.
4. Evaluation metric prefixes and suffixes are documented when they differ from
   legacy output.
5. If we choose to reuse legacy `analyze.py` directly, then add a compatibility
   adapter for keys and artifact layout.

## Suggested Commit Breakdown

1. Reproduction config schema and mapped experiment preset.
2. Configurable network and optimizer knobs.
3. Derived and slot-aware encoding.
4. Run loop, checkpoint, and evaluation compatibility.
5. Traversal endpoint metrics, progress, and profile compatibility.
6. Metric semantics and optional analysis adapter pass.

## Definition Of Done

The reproduction work is complete when:

1. A mapped config exists in this repository for the legacy experiment.
2. Every relevant legacy hyperparameter is either represented or explicitly
   marked as intentionally irrelevant.
3. A smoke-sized version of the mapped config runs end to end.
4. The full mapped config starts training with the same key semantics.
5. The information-state input dimension matches the legacy slot-aware setup for
   tier3.
6. The core training metrics and evaluation metrics are semantically comparable
   with the legacy experiment report.
