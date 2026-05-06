# Fast Engine Follow-up Optimizations

The current fast engine exposes Python wrappers for testing and debugging, but
serious traversal code should use the Cython `fast.pxd` API directly.

Deferred work:

1. Consider direct NumPy or feature-buffer output for RL pipelines instead of
   building Python lists and converting later.
2. Consider a single contiguous allocation for state arrays after profiling the
   simpler separate-allocation layout.
