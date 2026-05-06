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

Important implementation note:

- The active training path now calls `deep_cfr/traversal.pyx`.
- The old Python recursive `deep_cfr/traverser.py` path has been removed from
  mainline code.
- The rules engine (`game.pyx`), encoding (`encoding.pyx`), regret-matching math
  (`cfr_math.pyx`), and Deep CFR tree-walking loop now have Cython
  implementations.

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
14. Weighted current/recent/older/anchor self-play league buckets.
15. Safe-heuristic imitation pretraining.
16. Policy-gradient fine-tuning.
17. Single-vs-multiprocessing benchmark comparison.

## Remaining Differences From Legacy coolrl

The main Deep CFR v0 gaps listed earlier are now implemented. Remaining
differences are mostly experiment-system maturity and legacy-specific research
extras.

Still smaller than legacy:

1. Config is YAML-first and nested through Pydantic, but only one smoke preset
   exists under `configs/deep_cfr/`.
2. Multiprocessing exists, but it is intentionally simple:
   - no progress callback per worker batch
   - no hotspot timing profile
3. Metrics logging exists, but no plotting/status command exists yet.
4. Checkpoint artifacts are local only; W&B artifact integration is not added.
5. Legacy visualization helpers are not ported.

These remaining items are not blockers for running and iterating on Deep CFR v0.

## Performance-Critical Gap

This repository was split out to pursue much higher Lost Cities training
performance. From that perspective, the main remaining gap is not feature
parity with legacy `../coolrl`; it is the traversal backend.

Current state:

1. `GameState` mutation, legal-action generation, apply/undo, and cached scoring
   are implemented in Cython.
2. Information-state encoding and regret matching have Cython modules.
3. Full Deep CFR traversal now runs through `traversal.pyx`.
4. PyTorch policy inference and reservoir memory sample materialization still
   cross the Python boundary.
5. Traversal is still recursive inside Cython. The Python recursion-limit guard
   is no longer the main execution path, but an explicit iterative scheduler is
   still a future optimization.

Recommended performance roadmap:

1. Continue moving the traversal hot path away from Python object boundaries:
   - C-level legal action enumeration
   - C-level push/pop undo
   - terminal, depth cutoff, and node-budget cutoff
   - traverser/opponent node handling
   - outcome sampling
   - sampled action value correction
   - instantaneous regret calculation
   - strategy sample collection
   - traversal stats collection
2. Reduce Python boundary costs with batched memory writes.
3. Add batched network inference for policy calls.
4. Replace the recursive Cython DFS with an explicit Cython traversal scheduler.
5. Run multiple traversal contexts concurrently so policy-needed states can be
   encoded and evaluated in batches.

Python iterative traversal is not the preferred performance path. It would
remove Python recursion-limit risk, but it would keep most Python object and
callback overhead in the hot loop. For performance, the next serious step is a
Cython batched iterative traversal scheduler.

## Suggested Next Steps

1. Add batched memory writes from the Cython traversal engine.
2. Add benchmark output for recursive Cython traversal vs batched iterative
   traversal once the scheduler exists.
3. Add batched policy inference.
4. Add an explicit Cython iterative traversal scheduler.
5. Add a status/plot command that reads `metrics.jsonl`.
6. Add worker progress logging and hotspot timing profile.
7. Add W&B checkpoint artifacts after checkpoint quality is stable.
