# Deep CFR v0 Gap vs Legacy coolrl

This document compares the current `coolrl-lost-cities` Deep CFR v0
implementation with the legacy Lost Cities Deep CFR implementation in
`../coolrl`.

The current implementation proves that Cython traversal primitives, PyTorch
networks, memory collection, and a one-iteration smoke run can work together. It
is not yet equivalent to the legacy implementation.

## Current v0

Implemented:

1. Cython traversal primitives:
   - `random_rollout_value`
   - `root_action_values`
   - direct `GameState` C API use for legal actions and push/pop restoration
2. Recursive Python Deep CFR traverser:
   - `traverse(state, traverser, iteration, depth)` logic
   - terminal values
   - traverser vs opponent node behavior
   - sampled action recursion
   - sampled action value and node value calculation
   - instantaneous regret collection at traverser nodes
   - strategy-memory collection
   - depth and node-budget cutoffs
3. Advantage-network-driven traversal policies:
   - information-state encoding
   - advantage network forward pass
   - regret matching over legal actions
   - sampled action recursion
4. Minimal Cython information-state encoding.
5. Small PyTorch MLP.
6. Simple in-memory sample storage with legal masks.
7. Legal-mask-aware advantage loss and masked strategy loss.
8. Smoke tests for traversal restoration and trainer execution.

This is now a real single-process Deep CFR v0, but it is still not equivalent to
the legacy implementation.

## Major Algorithm Gaps

### Outcome Sampling Controls

Legacy `coolrl` supports:

1. `outcome_sampling_epsilon`
2. sampled action probability correction
3. optional sampled value clipping
4. unsampled regret modes:
   - `negative_node_value`
   - `zero`

Current v0 has none of these.

### Cutoff Values

Legacy traversal supports:

1. score-diff cutoff values
2. random rollout cutoff values
3. safe-heuristic rollout cutoff values
4. rollout max-step timeouts
5. cutoff stats

Current v0 uses score-diff cutoff values for depth and node-budget cutoffs, but
does not support rollout-based cutoff values.

## Training and Memory Gaps

### Reservoir Memory

Legacy implementation has separate `AdvantageMemory` and `StrategyMemory` with:

1. capacity limits
2. reservoir sampling
3. batch sampling
4. sample merging from traversal workers

Current v0 stores `TrainingSample` objects with legal masks, but still uses
list-like storage rather than true reservoir sampling.

### Config System

Legacy config is split into:

1. rules config
2. network config
3. encoding config
4. traversal config
5. optimization config
6. memory config
7. evaluation config
8. checkpoint config
9. run config

It also supports YAML loading and experiment-level overrides.

Current v0 has only `DeepCFRConfig`.

## Runtime and Operations Gaps

### Checkpointing

Legacy checkpoints include:

1. config
2. Lost Cities rules config
3. iteration
4. input/action dimensions
5. advantage networks
6. strategy network
7. optimizer states
8. self-play league snapshots
9. latest and per-iteration checkpoint files

Current v0 has no checkpoint save/load.

### Evaluation Integration

Legacy implementation can load a strategy checkpoint as a bot and evaluate it
against supported opponents.

Supported legacy eval flow includes:

1. random opponent
2. safe heuristic opponent
3. passive discard opponent
4. noisy/safe variants through the evaluation layer
5. many detailed gameplay metrics

Current v0 does not evaluate during training and does not expose a strategy-net
bot adapter.

### CLI

Legacy implementation has command-line tools for:

1. training
2. evaluation
3. evaluation suites
4. status/progress
5. plotting and visualization
6. traversal benchmarking
7. imitation/pretraining experiments
8. policy-gradient fine-tuning experiments

Current v0 has no CLI.

### Multiprocessing Workers

Legacy traversal can run through multiprocessing worker batches:

1. worker count resolution, including `auto`
2. traversal chunking
3. frozen network state dict transfer
4. worker-local traversal
5. result merging
6. progress logging
7. hotspot profiling

Current v0 runs in-process only.

### Metrics and Logging

Legacy training writes:

1. `metrics.jsonl`
2. `runtime_progress.json`
3. `train.log`
4. traversal nodes/sec
5. cutoff rates
6. endpoint depth buckets
7. advantage and strategy losses
8. evaluation metrics
9. hotspot timing metrics

Current v0 returns a small `IterationMetrics` object.

## Encoding Gaps

Legacy encoding is much richer. It includes:

1. phase one-hot
2. current-player indicator
3. player id
4. hand slots with card-type one-hot and empty-slot flag
5. both players' expedition card counts
6. expedition lengths
7. last numeric rank per expedition
8. discard pile card counts
9. discard pile length
10. discard pile top card
11. public card counts
12. deck ratio
13. turn-count ratio
14. pending-discard one-hot
15. optional derived playability features
16. optional slot-aware playability features

Current v0 includes only a minimal subset:

1. phase flags
2. current player
3. encoded player
4. deck ratio
5. player hand slot features
6. legal action mask

The current encoding should be expanded before serious training runs.

## Self-play League Gap

Legacy implementation supports self-play league opponent selection:

1. current networks
2. recent snapshots
3. older snapshots
4. safe-heuristic anchor
5. snapshot interval
6. maximum snapshot count

Current v0 has no league or checkpoint-snapshot opponent sampling.

## Practical Priority

Recommended implementation order:

1. Add outcome-sampling controls.
2. Add rollout-based cutoff values.
3. Replace list-like memory with reservoir memory.
4. Expand encoding to include public board, discard, and score features.
5. Add checkpoint save/load.
6. Add strategy-net bot adapter and evaluation integration.
7. Add CLI for train/eval/smoke.
8. Add traversal stats and benchmark reporting.
9. Add multiprocessing workers only after the single-process algorithm is
   correct.

The first four items are algorithm-critical. The rest are operationally useful
but should not block improving the learning loop.
