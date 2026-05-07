# Plan: Model-Size Experiment (Keystone for Model-Scale Optimizations)

**Status:** Ready to execute
**Owner:** Operator (runs grid on `home`); Codex (adds configs and runner script)
**Background:** See `docs/performance.md` → "Post-A Optimization Calculus",
"Option A Bench Result and Structural Ceiling", and "Clarifying the traversal
bottleneck: sync policy boundary, not SIMD" for the sequencing rationale. AMP,
`torch.compile`, and TensorRT are gated on the outcome of this experiment.
Traversal batching itself is now tracked separately in
`docs/plans/option_b_interleaved_traversal.md`; model-size growth does not fix
the current sync-blocking traversal scheduler by itself.

## Goal

Identify a `network` config (hidden_size × num_layers) where:

1. Learning-curve win-rates improve meaningfully over the current `default.yaml` baseline (hidden=512, layers=3) by iteration 200, AND
2. Per-call forward time is large enough that AMP / `torch.compile` / TensorRT overhead is amortized — the documented threshold is `hidden_size >= 1024` or `num_layers >= 6`.

The experiment must produce either a recommended new `network` config or a documented null result. Its output determines which model-scale optimization plans ship next.

## Non-goals

- Do NOT implement AMP, `torch.compile`, or TensorRT here. Those are separate plans gated on this experiment's outcome.
- Do NOT re-enable Option A (`traversal.inference_backend: server`). Option A
  has already been implemented and benchmarked; it is structurally capped by
  sync-blocking traversal. Revisit it only after Option B-style traversal
  interleaving can feed larger batches.
- Do NOT change traversal, replay buffer, or evaluation architecture.
- Do NOT change eval cadence (`eval_every`, `evaluation.games`) — those are separate variables.
- Do NOT change encoding (`input_dim` stays 365).
- Do NOT add transformers, residual blocks, or other architectural changes. Only MLP width and depth vary.

## Config wiring verification

**Confirmed: no code changes required.**

`NetworkConfig` in `src/coolrl_lost_cities/games/classic/deep_cfr/config.py` exposes `hidden_size` and `num_layers` directly (lines 68–72). `DeepCFRMLP.from_config` in `src/coolrl_lost_cities/games/classic/deep_cfr/networks.py` passes both fields through to `_build_mlp` (lines 49–66). Changing `hidden_size` and `num_layers` in a YAML file is sufficient — no `networks.py` or `config.py` edits needed.

## Success criteria

1. At least one tested config produces win-rate trajectories vs `safe_heuristic_strict` that are **clearly outside seed noise** compared to the current baseline at iteration 200 — OR a clear documented null result (no size in the tested range improves the curve).
2. `iteration_seconds`, `traversal_seconds`, `advantage_train_seconds`, `strategy_train_seconds`, and `policy_network_seconds` (eval) are captured for each tested size and written to `docs/performance.md`.
3. A recommended `network` config emerges from the data, OR the experiment documents why the current size should be kept, with specific rationale.

## Experiment grid

| Config name | `hidden_size` | `num_layers` | Notes |
| --- | ---: | ---: | --- |
| `model-size-512x3` | 512 | 3 | Current `default.yaml` baseline — run first as reference |
| `model-size-768x4` | 768 | 4 | Mid bump |
| `model-size-1024x6` | 1024 | 6 | AMP/compile/TRT threshold per `docs/performance.md` |
| `model-size-1536x8` | 1536 | 8 | Optional stretch — only if 768x4 and 1024x6 both run cleanly within memory |

All other config fields are identical to `default.yaml`. Configs live at:

```
configs/deep_cfr/model-size-512x3.yaml
configs/deep_cfr/model-size-768x4.yaml
configs/deep_cfr/model-size-1024x6.yaml
configs/deep_cfr/model-size-1536x8.yaml
```

## Per-config measurement protocol

- **Machine:** `home` (RTX 3090).
- **Iterations:** 200 per config.
- **Seeds:** single seed initially (use `run.seed: 79` from `default.yaml` for reproducibility). If a config sits at the boundary of "better / not better," run a second seed for that config and the baseline before deciding.
- **Eval cadence:** keep `eval_every: 25` and `evaluation.games: 100` (defaults). Do not change these — they are held constant across all grid points.
- **Metrics to capture** (all already emitted by the trainer; read from `metrics.jsonl`):
  - `iteration_seconds` — total iteration wall-clock.
  - `traversal_seconds` — traversal phase.
  - `advantage_train_seconds` — advantage network optimization.
  - `strategy_train_seconds` — strategy network optimization.
  - At eval iterations {50, 100, 150, 200}: `eval/<opponent>/win_rate` for all opponents, with special attention to `safe_heuristic_strict`.
  - At eval iterations: `eval/<opponent>/policy_network_seconds` — needed for the AMP/TRT prerequisite check.
- **Memory monitoring:** watch GPU VRAM during the 1024x6 and 1536x8 runs. If a run OOMs or VRAM > 20 GB, reduce `optimization.advantage_batch_size` and `optimization.strategy_batch_size` by half (1024 → 512) and note the change in the results table. Do not adjust traversal settings.

## Files to add

### Step 1 (verify wiring — no files needed)

Confirm that `NetworkConfig.hidden_size` and `NetworkConfig.num_layers` flow through to the built networks by running the smoke config and checking that parameter counts change with different values. No code edit is expected; this step is a verification gate only.

```bash
# Quick parameter count check — should be 2 different numbers
uv run python -c "
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.config import NetworkConfig
cfg_small = NetworkConfig(hidden_size=512, num_layers=3)
cfg_large = NetworkConfig(hidden_size=1024, num_layers=6)
m_small = DeepCFRMLP.from_config(365, 22, cfg_small)
m_large = DeepCFRMLP.from_config(365, 22, cfg_large)
p_small = sum(p.numel() for p in m_small.parameters())
p_large = sum(p.numel() for p in m_large.parameters())
print(f'512x3 params: {p_small:,}')
print(f'1024x6 params: {p_large:,}')
assert p_large > p_small
print('OK')
"
```

If the assertion passes, proceed. If `NetworkConfig` is missing a field or `from_config` ignores `num_layers`, fix the config schema before creating the YAML files. (Expected result: passes without changes.)

### Step 2 — add per-size YAML configs

Create four files under `configs/deep_cfr/`. Each file is a copy of `default.yaml` with only the `network` block and `run.experiment_name` changed. All other blocks — `traversal`, `optimization`, `evaluation`, `memory`, etc. — must be byte-identical to `default.yaml` so comparisons are clean.

**`configs/deep_cfr/model-size-512x3.yaml`** — baseline reference:

```yaml
# Extends default.yaml with explicit model-size label.
# hidden=512, layers=3: current default.yaml baseline.
run:
  experiment_name: model-size-512x3
  seed: 79
  max_iterations: 200
  max_minutes: null
  device: cuda
  use_amp: false

# ... (all other blocks identical to default.yaml) ...

network:
  hidden_size: 512
  num_layers: 3
  activation: relu
```

**`configs/deep_cfr/model-size-768x4.yaml`** — mid bump:

```yaml
run:
  experiment_name: model-size-768x4
  seed: 79
  max_iterations: 200
  ...
network:
  hidden_size: 768
  num_layers: 4
  activation: relu
```

**`configs/deep_cfr/model-size-1024x6.yaml`** — AMP/compile/TRT threshold:

```yaml
run:
  experiment_name: model-size-1024x6
  seed: 79
  max_iterations: 200
  ...
network:
  hidden_size: 1024
  num_layers: 6
  activation: relu
```

**`configs/deep_cfr/model-size-1536x8.yaml`** — optional stretch:

```yaml
run:
  experiment_name: model-size-1536x8
  seed: 79
  max_iterations: 200
  ...
network:
  hidden_size: 1536
  num_layers: 8
  activation: relu
```

Each YAML must be a complete, standalone config (not using YAML anchors or includes) so it is loadable via `uv run lost-cities-deep-cfr train --config <file>` without any `--set` overrides.

### Step 3 — add runner script

Create `scripts/run_model_size_experiment.sh`. The script runs the grid sequentially, one config at a time, in `tmux` to allow detach/reattach. It should:

1. Run each config with `--keep` so results land in `runs/`.
2. Print a separator between runs so the log is easy to scan.
3. Skip the 1536x8 config if the previous run's max VRAM exceeded a threshold (manual check; the script can print a prompt and wait for operator confirmation before proceeding to stretch).
4. After all runs complete, print the `runs/` directory listing so the operator can verify output locations.

Example structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/deep_cfr/model-size-512x3.yaml"
  "configs/deep_cfr/model-size-768x4.yaml"
  "configs/deep_cfr/model-size-1024x6.yaml"
)
STRETCH="configs/deep_cfr/model-size-1536x8.yaml"

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running: $cfg"
  echo "============================================================"
  uv run lost-cities-deep-cfr train --config "$cfg" --keep
done

echo
echo "Mandatory configs done. Runs:"
ls -d runs/*/  2>/dev/null | tail -5

echo
read -rp "Did 1024x6 fit in VRAM cleanly? Run 1536x8 stretch? [y/N] " yn
if [[ "${yn,,}" == "y" ]]; then
  echo "Running stretch: $STRETCH"
  uv run lost-cities-deep-cfr train --config "$STRETCH" --keep
fi

echo
echo "All done. Runs:"
ls -d runs/*/  2>/dev/null | tail -6
```

Run in `tmux` to allow detaching:

```bash
tmux new-session -s model-size-exp \
  -c /home/coolguy/dev/coolrl-lost-cities \
  'bash scripts/run_model_size_experiment.sh 2>&1 | tee /tmp/model_size_exp.log'
```

Attach later:

```bash
tmux attach -t model-size-exp
```

### Step 4 — run the grid (operator-driven)

This step is not for Codex. The operator runs the script on `home` and monitors progress. Expected rough wall-clock per config at 200 iterations (rough order of magnitude only, based on current 17.85s/iter for 512x3):

| Config | Rough iter time | Rough 200-iter wall-clock |
| --- | ---: | ---: |
| 512x3 (baseline) | ~18s | ~1.0h |
| 768x4 | ~25–35s | ~1.5–2.0h |
| 1024x6 | ~50–80s | ~3.0–4.5h |
| 1536x8 | ~100–160s | ~6.0–9.0h |

Total mandatory grid: approximately 6–8 hours. Run with `tmux`; do not rely on an active terminal session.

After all runs complete, collect results:

```bash
# Extract iteration timing and eval win-rates from each run
for run_dir in runs/*model-size*/; do
  echo "=== $run_dir ==="
  uv run python -c "
import json, pathlib, statistics
rows = [json.loads(l) for l in pathlib.Path('$run_dir/metrics.jsonl').read_text().splitlines() if l.strip()]
non_eval = [r for r in rows if not r.get('evaluation_seconds')]
if non_eval:
    times = [r['iteration_seconds'] for r in non_eval]
    print(f'  iter_seconds mean={statistics.mean(times):.1f} n={len(times)}')
eval_rows = {r['iteration']: r for r in rows if r.get('evaluation_seconds')}
for it in [50, 100, 150, 200]:
    if it in eval_rows:
        wr = eval_rows[it].get('eval/safe_heuristic_strict/win_rate', 'n/a')
        print(f'  iter={it} safe_heuristic_strict win_rate={wr}')
"
done
```

### Step 5 — append results to `docs/performance.md`

After the grid completes, append a date-stamped experiment subsection to `docs/performance.md` under the "Experiments" heading. The subsection must include:

- A results table with `iteration_seconds` mean (non-eval) and win-rate vs `safe_heuristic_strict` at {50, 100, 150, 200} for each config.
- A `policy_network_seconds` column from eval rows — this is the key data for the AMP/compile/TRT prerequisite check.
- The recommendation that follows from the decision tree below.

Subsection template:

```markdown
### Model-size experiment (YYYY-MM-DD)

Grid: hidden={512,768,1024,1536} × layers={3,4,6,8} subset. 200 iterations on home
(RTX 3090), seed 79. eval_every=25, evaluation.games=100 (defaults held).

| Config | iter_seconds | policy_network_s (iter 200) | wr@50 | wr@100 | wr@150 | wr@200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x3 (baseline) | | | | | | |
| 768x4 | | | | | | |
| 1024x6 | | | | | | |
| 1536x8 (if run) | | | | | | |

**Recommendation:** <see decision tree below>

Cross-reference: archived AMP implementation plan
`docs/plans/archive/amp_trainer.md`, compile plan
`docs/plans/torch_compile.md`, and Option B traversal plan
`docs/plans/option_b_interleaved_traversal.md`.
```

## Decision tree

Apply this logic after the grid completes:

### Branch A — a size unlocks the curve AND iter time is acceptable

**Condition:** at least one config at or above 768x4 shows win-rate trajectories vs `safe_heuristic_strict` that are clearly outside seed noise vs 512x3 baseline at iteration 200, AND `iteration_seconds` at that size is ≤ 3× the baseline (i.e., ≤ ~54s/iter).

**Action:**
1. Recommend that config as the new `network` default.
2. Update `default.yaml` `network` block in a follow-up commit.
3. Trigger AMP re-measurement: run `scripts/bench_amp_trainer.py` with the new model size. The implementation exists and is default-off after the 2026-05-07 smoke regression; see `docs/plans/archive/amp_trainer.md` for the original implementation plan.
4. Trigger `torch.compile` re-measurement per `docs/plans/torch_compile.md`.
5. Trigger TensorRT evaluation for batched eval/inference surfaces.
6. Do not treat this as sufficient to re-enable Option A. Larger models improve
   the IPC/GPU-forward ratio, but the observed traversal ceiling is still
   scheduling shape. Re-benchmark server inference only after Option B or a
   comparable traversal interleaving path can feed bs=64+ batches.

### Branch B — no size unlocks the curve

**Condition:** no config in the tested range shows a clear win-rate improvement over 512x3 at iteration 200 (or improvements are within seed noise on a single seed, confirmed on a second seed for the best candidate).

**Action:**
1. Document the null result in `docs/performance.md`.
2. Recommend staying on `default.yaml` (`hidden=512, layers=3`).
3. Do NOT trigger AMP/compile/TRT re-measurement — if larger models do not improve the learning curve, the compute overhead is not justified regardless of kernel speedup.
4. Flag for future revisit when a more substantial architecture change is under consideration (e.g., residual blocks, encoding improvements).

### Branch C — a size unlocks the curve BUT iter time is unacceptable

**Condition:** a config shows a clear win-rate improvement but `iteration_seconds` is > 3× baseline (> ~54s/iter), making 1000-iteration runs impractical within current operator time budgets.

**Action:**
1. Document the size as a "candidate-pending-optimization."
2. Do NOT update `default.yaml` yet.
3. Prioritize AMP and `torch.compile` specifically to reduce iter time at that size (since the learning-curve gain justifies eventual deployment). Note: this is the one case where AMP/compile work is triggered even though the size is not yet the default — the goal is to make the size affordable.
4. Revisit after AMP/compile land and re-measure iter time. If iter time drops below threshold, promote to default.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| 1024x6 or 1536x8 OOMs on RTX 3090 (24 GB) | Halve `optimization.advantage_batch_size` and `optimization.strategy_batch_size` (1024 → 512) before retrying. Note the change in the results table so the comparison is clean. |
| iter time grows disproportionately at larger sizes | The recommendation logic uses a 3× iter-time cap. If 1024x6 exceeds this, fall through to Branch C rather than Branch A even if win-rates improve. |
| Single seed ambiguity (config sits on the boundary) | Run a second seed (e.g., `run.seed: 42`) for the boundary config and the 512x3 baseline, then re-apply the decision tree. Do not run multiple seeds per config preemptively. |
| Learning-curve comparison confounded by randomness at iter 200 | Use the win-rate trajectory across {50, 100, 150, 200}, not just the final point. A clearly superior curve at all four checkpoints is more convincing than a single point difference. |
| Replay buffer warm-up differs at larger model size | All configs use the same `memory.advantage_capacity` and `memory.strategy_capacity` (2M each). Warm-up dynamics should be similar. If early iterations (< 50) show anomalous behavior, note it but do not change the eval cadence. |

## Out-of-scope

- AMP / `torch.compile` / TensorRT enablement — separate plans in `docs/plans/`.
- Eval cadence changes (`eval_every`, `evaluation.games`).
- Encoding changes (`input_dim` stays 365).
- Non-MLP architectures (transformers, residual blocks, etc.).
- Option B or Option C batched traversal.
- Multi-seed runs as a default — only triggered on boundary configs.

## Definition of done

- [ ] Step 1: parameter-count verification passes (no code changes expected).
- [ ] Step 2: four YAML configs added under `configs/deep_cfr/`, each a complete standalone file.
- [ ] Step 3: `scripts/run_model_size_experiment.sh` added and executable.
- [ ] Step 4: grid run on `home`, `metrics.jsonl` files present for all mandatory configs.
- [ ] Step 5: date-stamped results table and recommendation appended to `docs/performance.md` "Experiments" section, decision tree applied and documented.
- [ ] If Branch A: `default.yaml` `network` block updated in a follow-up commit and downstream re-measurement plans triggered.
