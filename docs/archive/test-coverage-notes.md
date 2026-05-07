# Test Coverage Notes

Current quick coverage command, excluding Cython line coverage:

```bash
uv run --with coverage coverage run --source=src/coolrl_lost_cities -m pytest tests/games/classic
uv run --with coverage coverage report -m
uv run --with coverage coverage html -d htmlcov
```

Latest snapshot:

- `98 passed`
- `69%` total coverage
- The report is effectively Python-only because the existing `.pyx` extensions
  were not built with Cython tracing.

Deferred ideas:

1. Add a tiny coverage helper script if this command becomes common.
2. Keep normal coverage Python-only by default, so the existing Cython build
   artifacts stay untouched.
3. If `.pyx` line coverage becomes useful, run it in a separate `git worktree`
   or fresh clone. Avoid `build_ext --inplace --force` in the main working tree,
   because it can overwrite the normal `.so` files with tracing builds.
4. Gate Cython tracing behind an environment variable such as
   `CYTHON_COVERAGE=1`.
5. For tracing builds, enable both Cython line tracing and coverage's Cython
   plugin:

```python
compiler_directives={
    "linetrace": True,
    # existing directives...
}
define_macros=[("CYTHON_TRACE", "1")]
```

```ini
[run]
plugins = Cython.Coverage
source = src/coolrl_lost_cities
```

Rough performance expectation:

- Python-only coverage: usually modest overhead.
- Cython tracing build without coverage: likely noticeable but manageable.
- Cython tracing plus coverage: can be several times slower, and tight
  traversal or encoding loops may be much worse.

Practical default: keep the main tree fast and boring. Measure `.pyx` coverage
only when there is a specific question about the Cython paths.
