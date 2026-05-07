# AGENTS.md

This repository is managed with `uv`. Use `uv run ...` for commands so the
project environment and Cython extensions are built/loaded consistently.

## Project Layout

- `src/coolrl_lost_cities/games/classic/game.pyx`: Cython Lost Cities engine.
- `src/coolrl_lost_cities/games/classic/deep_cfr/`: Deep CFR training,
  traversal, evaluation, analysis, and CLI code.
- `configs/deep_cfr/`: active Deep CFR YAML configs (kebab-case filenames).
  Currently holds two:
  - `default.yaml`: the canonical "best-known" baseline. Start here, then
    override fields via `--set` for experiments/ablations.
  - `smoke.yaml`: 1-iter sanity check for the training loop.
- `configs/archive/`: retired/historical configs. Don't modify; reference
  if you need to reproduce an old run.
- `runs/`: generated training runs. Gitignored, may be a symlink to larger
  storage. Layout:
  - `runs/archive/`: past runs. **Do not modify or delete.**
  - `runs/tmp/`: smoke, tests, throwaway. Free to `rm -rf` anytime.
  - `runs/<YYYY-MM-DD_HHMMSS>_<kebab-name>/`: real experiments (flat).
  Promote a `runs/<...>` directory to `runs/archive/` with a manual `mv`
  once analysis is complete.
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

The CLI auto-derives the run directory from `run.experiment_name` plus a
timestamp. By default runs land under `runs/tmp/`; pass `--keep` for a real
experiment that should live under `runs/`.

Smoke / throwaway run (lands in `runs/tmp/`):

```bash
uv run lost-cities-deep-cfr train --config configs/deep_cfr/smoke.yaml
# → runs/tmp/<YYYY-MM-DD_HHMMSS>_smoke/
```

Real experiment (lands in `runs/`):

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep
# → runs/<YYYY-MM-DD_HHMMSS>_deep-cfr-default/
```

Variant / ablation (override one field; keep slug informative):

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set training_weighting.mode=none \
  --set run.experiment_name=ablation-no-lcfr
# → runs/<YYYY-MM-DD_HHMMSS>_ablation-no-lcfr/
```

Short fixed-iteration run:

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --set run.max_iterations=100 \
  --set checkpoint.save_every=0
```

Resume (path required, no shortcut):

```bash
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --resume runs/<YYYY-MM-DD_HHMMSS>_<slug>/latest.pt
```

When `--resume` is given, the trainer reuses the resumed checkpoint's parent
directory; no new timestamped folder is created.

Useful train controls:

- `--keep`: real experiment, write under `runs/` (default is `runs/tmp/`).
- `--resume PATH`: resume from a specific checkpoint. `PATH` is required.
- `--set PATH=VALUE`: override config fields. Repeatable, parses values as
  YAML (e.g. `--set traversal.num_workers=4`, `--set run.max_minutes=null`).

Common `--set` overrides:

- `--set run.device=cuda`: set the trainer device.
- `--set run.experiment_name=foo-v2`: change the slug used in the run dir
  name (kebab-case).
- `--set checkpoint.exact_resume=true`: require checkpoint config compatibility.
- `--set checkpoint.save_latest=false --set checkpoint.save_every=0`:
  disable checkpoint writes.
- `--set checkpoint.save_every=0`: keep only `latest.pt` (no archives).
- `--set checkpoint.save_every=N`: archive every N iterations.

## Naming Conventions

- **Directory names, run dirs, config filenames, `experiment_name` values**:
  kebab-case (`deep-cfr-color-shared-512x3.yaml`,
  `runs/2026-05-08_103045_color-attn-v2/`).
- **YAML keys, Python identifiers, config field names**: snake_case
  (unchanged: `hidden_size`, `traversals_per_player`, `experiment_name`).
- The CLI converts `run.experiment_name` to a kebab slug when building the
  run directory, so values may contain spaces or mixed case.

## Long Runs

Run long jobs in a real `tmux` session so the user can attach and stop them.
Do not rely on Codex command sessions for long user-observable training runs.

Start a long unbounded run:

```bash
tmux new-session -s coolrl-deepcfr-unbounded \
  -c /home/coolguy/dev/coolrl-lost-cities \
  'uv run lost-cities-deep-cfr train \
    --config configs/deep_cfr/default.yaml \
    --set run.max_iterations=null \
    --set run.max_minutes=null \
    --keep'
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
tail -f runs/<YYYY-MM-DD_HHMMSS>_<slug>/train.log
```

The unbounded config intentionally has:

```yaml
run:
  max_iterations: null
  max_minutes: null
checkpoint:
  save_every: 100
```

`latest.pt` is updated continuously; archive checkpoints are written every 100
iterations. If disk is tight, set `--set checkpoint.save_every=0` (keep only
`latest.pt`) or increase `save_every`.

## Evaluation And Analysis

Evaluate a checkpoint:

```bash
uv run lost-cities-deep-cfr eval \
  --checkpoint runs/<run-dir>/latest.pt \
  --opponent random \
  --games 100 \
  --device cpu
```

Save evaluation game records:

```bash
uv run lost-cities-deep-cfr eval \
  --checkpoint runs/<run-dir>/latest.pt \
  --opponent random \
  --games 100 \
  --device cpu \
  --save-games runs/<run-dir>/eval_random_games.json
```

Generate analysis plots from `metrics.jsonl`:

```bash
uv run lost-cities-deep-cfr analyze \
  --run runs/<run-dir>
```

Write plots to a separate directory:

```bash
uv run lost-cities-deep-cfr analyze \
  --run runs/<run-dir> \
  --output-dir runs/<run-dir>/analysis
```

The analyzer reads `metrics.jsonl` and writes PNG files grouped by diagnostic
section. Opponents are compared within each plot using fixed colors. The
`lost-cities-deep-cfr analyze` subcommand uses the analyzer default smoothing
window, currently 1 iteration (no smoothing), and supports `--max-iteration`.

For smoothing controls, run the analyzer module directly:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.analyze \
  --run runs/<run-dir> \
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

## Weights & Biases (Optional)

Metrics can be mirrored to W&B. `wandb` is an optional extra; default
installs and runs do not require it.

Install:

```bash
uv sync --extra wandb
```

Run with W&B:

```bash
# Offline: no login, writes to <run_dir>/wandb/offline-run-*/
uv run lost-cities-deep-cfr train --config <...> --wandb --wandb-mode offline

# Online: requires `uv run wandb login` once, then real-time upload
uv run lost-cities-deep-cfr train --config <...> --wandb
```

W&B data is stored **per run** at `<run_dir>/wandb/`, not at a global
`runs/wandb/`. Each training run gets its own subfolder, so moving or
deleting a run directory carries its W&B data along with it.

Sync offline runs to wandb.ai later:

```bash
wandb sync runs/<run-dir>/wandb/offline-run-*
```

Flags:

- `--wandb`: enable W&B mirroring.
- `--wandb-project <name>`: defaults to `coolrl-lost-cities`.
- `--wandb-name <name>`: W&B run name; defaults to `run.experiment_name`.
- `--wandb-mode {online,offline,disabled}`: default `online`.
- `--wandb-tag <tag>`: tag the run; repeatable.

W&B is purely additive — `metrics.jsonl` remains the source of truth, and
`analyze` reads `metrics.jsonl`, not W&B. Disabling W&B never breaks
training, resume, or analysis.

### Notes and tags

Use `--wandb-notes` for the run's *purpose* (free-form prose) and
`--wandb-tag` for *categories you might filter on later* (short kebab-case
keywords, repeatable). Otherwise use them however you like. Just avoid:

- Tags that duplicate `config` (`lr-1e-4`, `traversal-280`) — W&B already
  indexes config fields.
- Tags that are unique per run (`test-1`, `2026-05-07`) — that's the run
  name and timestamp's job.
- Tag-as-sentence (`tested-bigger-traversal-with-lcfr`) — that belongs in
  `--wandb-notes`.

**Notes length**: 3–5 lines, commit-message-body length. Should answer
*why* (hypothesis), *what* (key config delta), and *baseline* (run/iter
to compare against). Long analyses go in `docs/` and are linked from
notes; don't paste them in.

### Comparing two runs

Default is **sequential, single seed**. Run baseline first, then the
treatment with exactly one config change, both with the same `run.seed`.
Tag both with a shared hypothesis tag (e.g. `--wandb-tag lr-bump`) so
they show up together in W&B's Compare Runs view.

Do **not** run multiple seeds per condition unless explicitly asked —
that doubles or quintuples wall-clock and isn't the default protocol.
Single-seed comparison is enough to surface a signal; multi-seed is a
follow-up to confirm it.

Do **not** run two trainings in parallel on the same GPU — VRAM/SM
contention slows both unevenly and breaks the comparison.

## Notes For Future Agents

- Prefer `rg`/`rg --files` for search.
- Use `apply_patch` for manual edits.
- Do not commit generated run artifacts from `runs/`.
- Cython-generated `.c` files are gitignored; edit `.pyx`/`.pxd` sources.
- Before committing, run `uv run ruff check .` and at least the relevant pytest
  subset. For Deep CFR changes, run
  `uv run pytest -q tests/games/classic/test_deep_cfr_trainer.py`.
