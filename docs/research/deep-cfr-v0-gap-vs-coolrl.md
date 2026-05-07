# Deep CFR v0: Architectural Parity and Performance Gaps

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-v0-gap-vs-coolrl.md`

## Question

What is the implementation status of Deep CFR v0 relative to the legacy `coolrl` reference, and what are the primary architectural bottlenecks remaining for high-performance training?

## Code reference

The core algorithmic components have been ported to Cython to ensure C-level performance for the rules engine and the tree-walking loop:

- `src/coolrl_lost_cities/games/classic/game.pyx:217`: `cdef class GameState` provides high-speed state mutation, legal action generation, and scoring.
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx:228`: `cpdef traverse` serves as the entry point for the recursive Deep CFR traversal engine.
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx:253`: `cdef _traverse` implements the core recursive tree-walking logic, including traverser/opponent node handling and outcome sampling.
- `src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx:425`: `def encode_info_state` generates the information-state feature vectors required for network inference.
- `src/coolrl_lost_cities/games/classic/deep_cfr/cfr_math.pyx`: Contains optimized regret-matching and advantage calculation primitives.

## Analysis

As of May 2026, the implementation has achieved functional parity with the legacy reference. The migration to Cython has successfully eliminated the Python recursion limit as a primary constraint and significantly reduced the per-node overhead for game rules and state management. The "v0" implementation covers the full suite of required features: recursive traversal, terminal value handling (including rollouts and score-diff cutoffs), reservoir memory management, and PyTorch-based network training.

However, a "performance-critical gap" remains. While the traversal loop is in Cython, it remains **synchronous and recursive**. This architecture incurs significant costs at the Python/C boundary:
1.  **Synchronous Policy Inference:** Each node requiring a policy must wait for a PyTorch forward pass. Because these calls cross back into Python, they cannot be efficiently batched across different branches or traversal contexts, leading to poor GPU utilization.
2.  **Data Materialization:** Writing training samples from Cython-managed buffers into NumPy arrays for the reservoir memory involves frequent boundary crossings.
3.  **Lack of Concurrency:** The recursive depth-first search (DFS) pattern makes it difficult to interleave multiple traversal contexts, which is a prerequisite for effective inference batching.

## Practical implication

The implementation is algorithmically complete and suitable for verifying the correctness of the Deep CFR agent. However, for large-scale training, the current recursive DFS is a bottleneck. The recommended path forward is a **batched iterative traversal scheduler** implemented in Cython. Moving to an explicit stack-based or queue-based scheduler will allow the system to:
-   Interleave multiple traversal contexts.
-   Collect policy requests from many contexts into a single batch.
-   Execute a single large forward pass on the GPU, drastically reducing the impact of the Python/C boundary and maximizing throughput.

Until this transition is made, optimization efforts should focus on reducing the frequency and cost of policy calls rather than micro-optimizing the already efficient Cython rules engine.

## References

- `docs/archive/deep-cfr-v0-gap-vs-coolrl.md`: Original status and gap analysis.
- `docs/research/deep-cfr-v0-feature-parity.md`: Detailed subsystem coverage report.
- `docs/research/batched-traversal-inference-decision.md`: Architectural decision record for the next-generation inference server.