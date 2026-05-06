# Fast Engine Follow-up Optimizations

The current fast engine exposes Python wrappers for testing and debugging, but
serious traversal code should use the Cython `fast.pxd` API directly.

Deferred work:

1. Add an internal undo stack with `push_action()` / `pop_action()` so Python
   callers can avoid tuple allocation when they need nested search.
2. Keep traversal legal-action generation caller-buffer based. Consider a
   reusable Python-wrapper action buffer only if wrapper profiling shows
   `legal_actions()` allocation is material.
3. Consider direct NumPy or feature-buffer output for RL pipelines instead of
   building Python lists and converting later.
4. Consider a single contiguous allocation for state arrays after profiling the
   simpler separate-allocation layout.
