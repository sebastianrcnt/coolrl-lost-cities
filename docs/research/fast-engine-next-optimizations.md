# Fast Engine Optimization Architecture

**Last verified:** 2026-05-08, commit `5c221fb`

Source: `docs/archive/fast-engine-next-optimizations.md`

## Question

How should the Cython-based game engine be optimized to support the high-throughput requirements of reinforcement learning algorithms like Deep CFR?

## Analysis

The core of the Lost Cities implementation (the "fast engine") is built as a Cython extension to provide a high-performance alternative to pure-Python game logic. While the current architecture provides the necessary speed for basic experimentation, it contains several intentional design choices—such as `cpdef` method wrappers and fragmented memory management—that become significant bottlenecks during large-scale self-play and traversal.

### Cython `pxd` API vs. Python Boundaries

The engine currently exposes its logic primarily through `cpdef` methods in `src/coolrl_lost_cities/games/classic/game.pyx`. While these methods are accessible from both Python and Cython, calling them from the Python interpreter incurs conversion overhead for every argument and return value. For algorithms like Deep CFR that visit millions of nodes per iteration, this overhead is prohibitive.

To achieve maximum performance, traversal code must bypass the Python interpreter entirely. This is possible by importing the game state's C-level API directly from `src/coolrl_lost_cities/games/classic/game.pxd`. High-performance modules, such as the Deep CFR trainer, should favor `cdef` methods like `_apply_action_unchecked_c` and `_legal_actions_c` which operate on raw C types and pointers.

### Contiguous Memory Allocation

In its current state, the `GameState` class manages its internal arrays using multiple discrete memory allocations. In `game.pyx` at lines 314–323, the `_configure` method performs approximately eleven separate `malloc` calls to set up the deck, hands, expeditions, and scoring buffers:

```cython
self.deck_cards = <int*>malloc(self.total_cards * sizeof(int))
self.hand_cards = <int*>malloc(2 * self.hand_size * sizeof(int))
# ... and several others
```

This fragmented allocation pattern increases memory management overhead and can lead to poor cache locality during state transitions. Consolidating these buffers into a single contiguous block of memory would not only improve cache performance but also simplify state cloning. A single `memcpy` could duplicate the entire game state, significantly speeding up the recursive branching required by CFR-based traversers.

### Zero-Copy Feature Extraction

A major cost in the current RL pipeline is the construction of observation vectors and action masks. Methods like `unified_legal_mask_np` (line 797 in `game.pyx`) currently build Python lists of booleans before converting them into NumPy arrays. This "build-and-convert" pattern generates excessive Python object churn in the inner loop of training.

The engine must transition toward a "zero-copy" paradigm where observation and mask construction logic writes directly into pre-allocated destination buffers. This pattern is already partially implemented in `src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx` with the `encode_info_state_c` function, which accepts a `float*` pointer. Expanding this approach to all high-frequency data paths—including legal action masks—is essential for eliminating Python-level overhead during training.

## Practical Implication

Performance-critical components—specifically interleaved traversal and batched inference servers—should be designed to interface with the engine at the C level. Future refactoring of the `GameState` should prioritize a single-allocation memory model to enable rapid cloning and cache-efficient updates. All feature extraction logic must move toward buffer-writing APIs to ensure that node traversal speed is limited by algorithmic complexity rather than Python infrastructure.

## References

- `src/coolrl_lost_cities/games/classic/game.pxd`: Direct C-API definitions for the fast engine.
- `src/coolrl_lost_cities/games/classic/game.pyx`: Core state management and `malloc`-based allocation logic.
- `src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx`: Reference implementation for buffer-based feature extraction.