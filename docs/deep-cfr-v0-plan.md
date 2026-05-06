# Deep CFR v0 Plan

The first Deep CFR target is an end-to-end training pipeline that runs on the
classic Lost Cities game and exercises the Cython game-state hot path. It does
not need to be a final research-grade implementation.

## Goals

1. Add a real Deep CFR package under
   `coolrl_lost_cities.games.classic.deep_cfr`.
2. Keep traversal in Cython so it can call `GameState` C APIs directly:
   `_legal_actions_c`, `_unified_legal_actions_c`, `push_action`,
   `pop_action`, score caches, and deck-sampling helpers.
3. Train minimal PyTorch advantage and strategy networks from traversal output.
4. Save and load checkpoints for networks, optimizer state, config, and
   training counters.
5. Run a small smoke test that completes at least one training iteration on a
   tiny workload.
6. Add a small benchmark for Cython rollout/traversal steps per second.

## Non-goals

- Multi-machine distributed training.
- Highly optimized replay-memory storage.
- Exploitability calculation.
- Perfect feature encoding.
- Full ISMCTS integration.
- Large experiment orchestration.

## Proposed Package Shape

```text
src/coolrl_lost_cities/games/classic/deep_cfr/
  cfr_math.pyx
  cfr_math.pxd
  encoding.pyx
  encoding.pxd
  traversal.pyx
  traversal.pxd
  config.py
  memory.py
  networks.py
  trainer.py
  checkpoints.py
```

`cfr_math.pyx` and `encoding.pyx` already exist as scaffolding. The next major
piece is `traversal.pyx`.

## Chance Handling

Lost Cities has hidden information from deck order and opponent hand contents.
For v0, model chance by sampling compatible deck order through `GameState`
mutation instead of cloning Python snapshots.

Initial implementation:

1. Use a deterministic RNG seed per traversal.
2. Shuffle the remaining deck region with C-level deck swaps.
3. Apply and undo actions with `push_action` and `pop_action`.
4. Do not expose opponent hand or exact unseen deck order in the information
   state encoding.

This is enough for smoke tests and speed work. More precise public-belief or
particle sampling can come later.

## Encoding v1

The current encoding is intentionally minimal. Deep CFR v0 should expand it
without leaking hidden information.

Include:

- Current phase.
- Current player.
- Traversing player.
- Player hand.
- Both players' public expeditions.
- Public discard piles.
- Remaining deck ratio.
- Cached score or score diff.
- Legal action mask.

Do not include:

- Opponent hand contents.
- Exact hidden deck order.
- Any future card identity that the acting player cannot infer.

The encoding should keep a stable `input_dim` for a given config and expose both
Python wrappers for tests and C-level buffer writes for traversal.

## Traversal v0

The first traversal does not need every Deep CFR detail. It should prove that the
Cython control flow is viable.

Steps:

1. Add `traversal.pyx/.pxd`.
2. Implement a deterministic random rollout helper that uses C-level legal
   actions and `push_action`/`pop_action`.
3. Add external-sampling Deep CFR traversal using regret matching.
4. Emit advantage-memory rows with `(info_state, iteration, action_advantages)`.
5. Emit strategy-memory rows with `(info_state, iteration, action_policy)`.
6. Add tests for determinism, push/pop restoration, terminal handling, and
   action legality.

The first version may return Python objects at module boundaries. Inner loops
should stay Cython-native.

## Python Training Layer

`trainer.py` should own orchestration:

1. Build or load networks.
2. Run traversal workers.
3. Append samples to advantage and strategy memories.
4. Train advantage networks per player.
5. Train the average strategy network.
6. Periodically evaluate using the existing policy/evaluation layer.
7. Periodically checkpoint.

`memory.py` can start simple with bounded Python/NumPy buffers. Replace it only
after profiling.

## Validation

Required v0 checks:

1. Cython build succeeds.
2. Unit tests for math, encoding, chance sampling, and traversal restoration.
3. A tiny one-iteration trainer smoke test.
4. A benchmark that reports rollout/traversal steps per second.

The first useful benchmark is not model quality. It is whether traversal is
actually using the Cython hot path instead of rebuilding Python masks and
snapshots.
