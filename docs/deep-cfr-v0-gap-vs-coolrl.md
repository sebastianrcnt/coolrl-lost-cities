# Deep CFR v0 Status vs Legacy coolrl

This document tracks the Lost Cities Deep CFR functionality in this repository
against the legacy implementation in `../coolrl`.

## Implemented In This Repository

### Traversal

Implemented:

1. Recursive `traverse(state, traverser, iteration, depth)` logic.
2. Terminal value handling.
3. Traverser vs opponent node behavior.
4. Advantage-network-driven policies during traversal.
5. Regret matching over legal actions.
6. Sampled action recursion with node-value calculation.
7. Instantaneous regret collection at traverser nodes.
8. Strategy-memory collection.
9. Depth and node-budget cutoffs.
10. Outcome-sampling epsilon.
11. Sampled action probability correction.
12. Optional sampled value clipping.
13. Unsampled regret modes:
    - `negative_node_value`
    - `zero`
14. Score-diff and rollout-based cutoff values.
15. Random and safe-heuristic cutoff rollout policies.
16. Deck-draw chance sampling with state restoration.

### Training And Memory

Implemented:

1. PyTorch advantage networks.
2. PyTorch strategy network.
3. Legal-mask-aware advantage loss.
4. Masked strategy cross-entropy loss.
5. Reservoir memory with capacity limits.
6. Batch sampling.
7. Legal masks stored with samples.
8. Single-process traversal.
9. Multiprocessing traversal worker batches.
10. Worker result merge in the parent trainer process.

### Encoding

Implemented information-state features:

1. Phase flags.
2. Current player.
3. Encoded player.
4. Deck ratio.
5. Player hand slot features.
6. Public expedition summaries for both players.
7. Public discard summaries.
8. Public card counts.
9. Total score and score diff features.
10. Turn ratio.
11. Pending-discard one-hot.
12. Legal action mask.

The encoding is still compact compared with the legacy feature set, but it now
contains the key public board, discard, score, and legal-action information.

### Runtime Operations

Implemented:

1. Checkpoint save/load.
2. Latest and per-iteration checkpoint files.
3. Config stored in checkpoints and `config.json`.
4. Strategy-net policy adapter.
5. Evaluation against registered classic bots.
6. Training CLI.
7. Evaluation CLI.
8. Traversal benchmark CLI.
9. Local run files:
   - `config.json`
   - `metrics.jsonl`
   - `runtime_progress.json`
   - `train.log`
10. Traversal benchmark metrics.
11. Self-play league snapshots.
12. Self-play league opponent selection from stored snapshots.
13. Safe-heuristic anchor opponent path.

## Remaining Differences From Legacy coolrl

The main Deep CFR v0 gaps listed earlier are now implemented. Remaining
differences are mostly experiment-system maturity and legacy-specific research
extras.

Still smaller than legacy:

1. Config is a single `DeepCFRConfig` dataclass rather than a deeply nested
   YAML-first config tree.
2. Multiprocessing exists, but it is intentionally simple:
   - no auto worker-count resolver
   - no progress callback per worker batch
   - no hotspot timing profile
3. Metrics logging exists, but no plotting/status command exists yet.
4. Checkpoint artifacts are local only; W&B artifact integration is not added.
5. Self-play league is simpler than legacy:
   - snapshot sampling exists
   - safe anchor path exists
   - weighted recent/older/current buckets are not implemented
6. Legacy side experiments are not ported:
   - imitation/pretraining commands
   - policy-gradient fine-tuning commands
   - legacy visualization helpers

These remaining items are not blockers for running and iterating on Deep CFR v0.

## Suggested Next Steps

1. Add W&B/JSONL tracker abstraction on top of the existing local run files.
2. Add a status/plot command that reads `metrics.jsonl`.
3. Add richer benchmark output comparing single-process and multiprocessing.
4. Add weighted self-play league bucket selection if experiments need it.
5. Add W&B checkpoint artifacts after checkpoint quality is stable.
