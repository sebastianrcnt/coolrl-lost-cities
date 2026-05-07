# AGENTS.md

This repository is managed with `uv`. Use `uv run ...` for commands so the
project environment and Cython extensions are built/loaded consistently.

## Project Layout

- `src/coolrl_lost_cities/games/classic/game.pyx`: Cython Lost Cities engine.
- `src/coolrl_lost_cities/games/classic/deep_cfr/`: Deep CFR training,
  traversal, evaluation, analysis, and CLI code.
- `configs/deep_cfr/`: Deep CFR YAML configs.
- `runs/`: generated training runs. This path is gitignored and may be a
  symlink to larger storage.
- `docs/`: profiling notes, migration notes, and experiment documentation.

## Core Commands

Run lint:

```bash
uv run ruff check .
```

Run all tests:

```bash
uv run pytest -q
```

Run focused Deep CFR tests:

```bash
uv run pytest -q tests/games/classic/test_deep_cfr_trainer.py
```

Run the CLI through the console script:

```bash
uv run lost-cities-deep-cfr --help
```

Equivalent module form:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli --help
```

## Deep CFR Training

Main full config:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml
```

Unbounded config:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability_unbounded.yaml
```

Short fixed-iteration run:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --iterations 100 \
  --save-latest-only
```

Use explicit run directories for experiments. Put Deep CFR runs under
`runs/deep_cfr/` and prefix generated run names with the date:

```bash
RUN_DIR="runs/deep_cfr/$(date +%Y-%m-%d_%H%M%S)_deep_cfr_experiment_name"
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir "$RUN_DIR" \
  --max-iterations 100
```

Date-prefixed examples:

- `runs/deep_cfr/YYYY-MM-DD_HHMMSS_deep_cfr_100iter`
- `runs/deep_cfr/YYYY-MM-DD_HHMMSS_deep_cfr_unbounded`

Useful train overrides:

- `--resume`: resume from `<checkpoint-dir>/latest.pt`.
- `--resume PATH`: resume from a specific checkpoint.
- `--exact-resume`: require checkpoint config compatibility for exact resume.
- `--no-save`: disable checkpoint writes.
- `--save-latest-only`: keep only `latest.pt`.
- `--save-iteration-interval N`: archive every N iterations.
- `--set PATH=VALUE`: override arbitrary config fields, e.g.
  `--set traversal.num_workers=4`.

## Long Runs

Run long jobs in a real `tmux` session so the user can attach and stop them.
Do not rely on Codex command sessions for long user-observable training runs.

Start a long unbounded run:

```bash
tmux new-session -s coolrl-deepcfr-unbounded \
  -c /home/coolguy/dev/coolrl-lost-cities \
  'uv run lost-cities-deep-cfr train \
    --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability_unbounded.yaml'
```

Attach later:

```bash
tmux attach -t coolrl-deepcfr-unbounded
```

Detach without stopping:

```text
Ctrl+B, D
```

Stop training:

```text
Ctrl+C
```

Follow logs from another terminal:

```bash
tail -f runs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability_unbounded/train.log
```

The unbounded config intentionally has:

```yaml
run:
  iterations: null
  max_iterations: null
  max_hours: null
checkpoint:
  save_iteration_interval: 100
```

`latest.pt` is updated continuously; archive checkpoints are written every 100
iterations. If disk is tight, prefer `--save-latest-only` or increase
`save_iteration_interval`.

## Evaluation And Analysis

Evaluate a checkpoint:

```bash
uv run lost-cities-deep-cfr eval \
  --checkpoint runs/deep_cfr/<run-name>/latest.pt \
  --opponent random \
  --games 100 \
  --device cpu
```

Save evaluation game records:

```bash
uv run lost-cities-deep-cfr eval \
  --checkpoint runs/deep_cfr/<run-name>/latest.pt \
  --opponent random \
  --games 100 \
  --device cpu \
  --save-games runs/deep_cfr/<run-name>/eval_random_games.json
```

Generate analysis plots from `metrics.jsonl`:

```bash
uv run lost-cities-deep-cfr analyze \
  --run runs/deep_cfr/<run-name>
```

Write plots to a separate directory:

```bash
uv run lost-cities-deep-cfr analyze \
  --run runs/deep_cfr/<run-name> \
  --output-dir runs/deep_cfr/<run-name>/analysis
```

The analyzer reads `metrics.jsonl` and writes PNG files grouped by diagnostic
section. Opponents are compared within each plot using fixed colors. The
`lost-cities-deep-cfr analyze` subcommand uses the analyzer default smoothing
window, currently 1 iteration (no smoothing), and supports `--max-iteration`.

For smoothing controls, run the analyzer module directly:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.analyze \
  --run runs/deep_cfr/<run-name> \
  --smoothing-window 5
```

Use `--no-smoothing` to force no moving average.

Current output files:

- `analysis_01_loss.png`
- `analysis_02_match.png`
- `analysis_03_action.png`
- `analysis_04_gameflow.png`
- `analysis_05_open_quality.png`
- `analysis_06_expedition_outcomes.png`
- `analysis_07_calibration.png`
- `analysis_08_traversal.png`
- `analysis_09_selectivity.png`
- `analysis_final_eval_summary.png`

## Runtime Artifacts

Each training run writes:

- `metrics.jsonl`: structured metrics, one completed iteration per line.
- `train.log`: human-readable timestamped logs.
- `runtime_progress.json`: latest progress snapshot.
- `latest.pt`: latest checkpoint.
- `iteration_*.pt`: archive checkpoints when enabled.
- `config.json`: resolved config for the run.

If a run is stopped mid-iteration, the in-progress iteration may not appear in
`metrics.jsonl`. Analyze the latest completed metric row.

## Notes For Future Agents

- Prefer `rg`/`rg --files` for search.
- Use `apply_patch` for manual edits.
- Do not commit generated run artifacts from `runs/`.
- Cython-generated `.c` files are gitignored; edit `.pyx`/`.pxd` sources.
- Before committing, run `uv run ruff check .` and at least the relevant pytest
  subset. For Deep CFR changes, run
  `uv run pytest -q tests/games/classic/test_deep_cfr_trainer.py`.
