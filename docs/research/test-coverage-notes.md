# Test Coverage Strategy for Python and Cython Modules

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/test-coverage-notes.md`

## Question
How should the project manage test coverage reporting, particularly for performance-critical Cython extensions, without compromising development speed or contaminating standard build artifacts?

## Analysis
The project utilizes a hybrid architecture where core game logic and algorithmic traversals are implemented in Cython (`.pyx`) for performance, while high-level coordination and configuration are in Python. Standard coverage tools (e.g., `coverage.py`) effectively track Python execution but require specific build-time instrumentation to observe line-level execution within Cython modules.

Enabling Cython tracing introduces several complications:
1.  **Performance Degradation**: Instrumenting tight loops in `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx` or `encoding.pyx` with `linetrace` can result in significant overhead, making large-scale tests or simulations prohibitively slow.
2.  **Artifact Contamination**: A `build_ext --inplace --force` command with tracing enabled overwrites the optimized `.so` files. If these artifacts are accidentally committed or used for benchmarking, they will report misleadingly slow performance.
3.  **Build Complexity**: It requires conditional logic in `setup.py` to toggle `compiler_directives` and `define_macros` based on environment variables.

## Practical Implication
The project adopts a "Python-first, Cython-selective" coverage policy to balance visibility with performance.

### 1. Default Python-Only Coverage
For routine development and CI, coverage is restricted to Python modules. This provides high-level assurance of test execution without impacting the speed of the Cython core. The standard reporting command is:

```bash
uv run --with coverage coverage run --source=src/coolrl_lost_cities -m pytest tests/games/classic
```

### 2. Isolated Cython Tracing
When verification of Cython logic paths is required, it should be performed in an isolated environment (such as a separate `git worktree`) to prevent optimized build artifacts from being overwritten in the main development branch. 

To enable tracing, `setup.py:12` would need to be modified (ideally via an environment variable like `CYTHON_COVERAGE=1`) to include:

```python
# setup.py (proposed modification)
extensions = cythonize(
    [...],
    compiler_directives={
        "linetrace": True,
        # ... other directives
    },
    define_macros=[("CYTHON_TRACE", "1")]
)
```

Additionally, a `.coveragerc` file must include the Cython plugin:

```ini
[run]
plugins = Cython.Coverage
source = src/coolrl_lost_cities
```

### 3. Recommendations
*   **Maintain Fast Defaults**: Keep the main tree "fast and boring." Avoid enabling Cython tracing by default.
*   **Targeted Audits**: Use Cython coverage only when introducing new complex logic in modules like `cfr_math.pyx` or `traversal.pyx` to ensure edge cases are exercised.
*   **External Trace**: Use a dedicated CI job or script for Cython coverage reporting rather than manual developer runs.

## References
- `setup.py`: Extension definitions for `game.pyx`, `cfr_math.pyx`, `encoding.pyx`, `traversal.pyx`, and `heuristic_cy.pyx`.
- `docs/archive/test-coverage-notes.md`: Initial profiling and snapshots.
- [Cython Documentation: Debugging and profiling](https://cython.readthedocs.io/en/latest/src/userguide/debugging.html)