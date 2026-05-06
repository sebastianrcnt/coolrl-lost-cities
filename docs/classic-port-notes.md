# Classic Port Notes

This repository starts as a focused extraction of the Lost Cities game from the
legacy `coolrl` repository. The first target is the classic two-player card
game, without the earlier training-oriented tiers.

## Current Direction

- Implement the classic Lost Cities rules first.
- Treat classic as the initial concrete game under `coolrl_lost_cities.games`.
- Do not carry over `tier0` through `tier3`; those were useful for experiments,
  but they should not shape the first public game API.
- Keep backend selection available. The Python/Cython and Rust implementations
  should remain swappable behind a small backend boundary.
- Keep the GUI and Rust implementation in scope for the port.
- Keep RL and training code out of the first extraction.

The expected package shape is roughly:

```text
src/coolrl_lost_cities/
  games/
    classic/
      game.pyx
      env.py
      interfaces.py
      backends/
      bots/
      pygame_pvp.py
      fixtures/
      assets/
      docs/
rust/
  lost-cities-core/
proto/
  lost_cities.proto
```

Tests should live outside the package, roughly under:

```text
tests/games/classic/
```

## Out Of Scope For The First Port

- Deep CFR
- General training infrastructure
- Evaluation loops for learned policies
- Web client
- Legacy experiment configs, checkpoints, logs, exports, and analysis artifacts

Bot-vs-bot helpers can stay with the classic game if they are useful for smoke
tests and local play. Broader policy evaluation can be introduced later with the
training layer.

## Later Training Shape

If training is added later, it should not make Deep CFR the center of the
package. Evaluation and policy interfaces should be general enough for multiple
approaches, with Deep CFR as one implementation.

A possible future shape:

```text
src/coolrl_lost_cities/
  games/
    classic/
  training/
    policies.py
    evaluation.py
    deep_cfr/
    imitation/
    policy_gradient/
```

The game package should expose rules, state transitions, legal actions, scoring,
backend selection, and playable UI. Training code can adapt those pieces later.

## Naming Notes

For now, use `classic` for the five-expedition game. Other variants, such as a
six-expedition version, can be added later if needed. The current port should
avoid adding a variant registry or broad abstraction before there is a second
concrete game to support.
