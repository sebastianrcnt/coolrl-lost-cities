# Classic Game Port Architecture

**Last verified:** 2026-05-08, commit `b0b3855`
**Source:** docs/archive/classic-port-notes.md

The "classic" game port serves as the fundamental layer of the `coolrl-lost-cities` project, representing a focused extraction of the Lost Cities game from the legacy `coolrl` repository. This architectural foundation was established to isolate the core mechanics of the two-player card game from the experimental "tiers" (simplified variants tier0 through tier3) used in earlier research. By centering the repository on the full classic ruleset, the project provides a stable, high-performance API that serves both as a playable game and a rigorous training environment.

## Question: Architectural Scope and Design
How is the Lost Cities "classic" game structured and scoped to serve as the foundation of the repository while remaining decoupled from specific training algorithms?

## Code Reference
- `src/coolrl_lost_cities/games/classic/game.pyx`: The core high-performance game logic and state management.
- `src/coolrl_lost_cities/games/classic/env.py`: The standard environment wrapper for algorithmic interaction.
- `src/coolrl_lost_cities/games/classic/pygame_pvp.py`: The graphical interface for human-to-human play and debugging.

## Analysis and Derivation
The transition from the legacy `coolrl` codebase to this standalone repository involved a deliberate narrowing of scope. The primary design goal was to treat the classic game as the initial concrete implementation under `coolrl_lost_cities.games`, avoiding the complexity of a variant registry or broad abstraction layers before they were strictly necessary.

A key technical decision was the use of an in-process Cython implementation for the game state. By defining `cdef class GameState` (see `src/coolrl_lost_cities/games/classic/game.pyx:217`), the project achieves C-level performance for state transitions, legal action masking, and scoring. This efficiency is critical for compute-intensive search algorithms like Deep Counterfactual Regret Minimization (Deep CFR), where the overhead of pure Python state management would be prohibitive.

The architecture strictly separates the game rules from the training infrastructure. While the classic game provides the necessary hooks for reinforcement learning—such as observation vectors and reward signals—it does not depend on any specific learning library. This separation ensures that the game logic remains verifiable and readable, centered on the rules and state transitions rather than the requirements of a particular neural network architecture.

Furthermore, the "classic" designation is specifically applied to the five-expedition version of the game. This naming convention leaves room for future variants, such as six-expedition versions, without requiring a breaking change to the core package structure. The removal of separate native backends in favor of a single, highly-optimized Cython backend simplifies the build process and ensures consistency between local play and large-scale training runs.

## Practical Implication
The resulting package structure allows developers to interact with the game through multiple interfaces: a raw Cython API for high-performance search, a Gym-like environment for reinforcement learning, and a Pygame-based GUI for manual verification. This modularity means that improvements to the game logic (e.g., scoring optimizations in `score_expedition` at `src/coolrl_lost_cities/games/classic/game.pyx:188`) automatically benefit all downstream consumers, from the training loops to the interactive bots.

## References
- `src/coolrl_lost_cities/games/classic/game.pyx`: Core logic, state cloning, and legal action generation.
- `src/coolrl_lost_cities/games/classic/env.py`: Environment state management and step logic.
- `docs/archive/classic-port-notes.md`: Original design notes regarding the extraction and scoping.