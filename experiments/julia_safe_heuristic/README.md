# Julia Safe-Heuristic Experiment

This experiment keeps Julia out of the production bot registry. Python exports
`GameState.to_snapshot()` parity cases with expected Python actions, then Julia
reads those snapshots and computes actions in-process.

Generate a corpus:

```bash
uv run python scripts/export_safe_heuristic_snapshots.py \
  --output runs/tmp/safe_heuristic_snapshots.jsonl \
  --seeds 50
```

Run parity:

```bash
tools/julia/current/bin/julia --project=experiments/julia_safe_heuristic -e 'using Pkg; Pkg.instantiate()'
tools/julia/current/bin/julia --project=experiments/julia_safe_heuristic \
  experiments/julia_safe_heuristic/test/parity.jl \
  runs/tmp/safe_heuristic_snapshots.jsonl
```

Run throughput benchmark:

```bash
tools/julia/current/bin/julia --project=experiments/julia_safe_heuristic \
  experiments/julia_safe_heuristic/bench/bench_snapshots.jl \
  runs/tmp/safe_heuristic_snapshots.jsonl
```

The benchmark measures pure Julia action selection over already-loaded JSON
records. It does not measure Python-to-Julia per-turn FFI, which is deliberately
not part of this experiment.
