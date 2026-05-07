# Plan: Trainer-side AMP (Automatic Mixed Precision)

**Status:** Archived. Implemented and benchmarked on 2026-05-07; default remains off after smoke-config AMP regression. See `docs/performance.md` "AMP on trainer networks (2026-05-07, regression)".
**Owner:** Codex
**Background:** See `docs/performance.md` → "AMP Status" (currently a no-op flag) and "Post-A Optimization Calculus" (AMP becomes more meaningful only at larger model sizes; today's gains are bounded by the small `DeepCFRMLP` and the dominant traversal phase). Also note the `torch.compile` regression experiment in the same doc — small models punish low-level kernel optimizations because dispatch overhead outweighs fused-kernel gains. AMP can hit the same wall.

## Goal

Wire `run.use_amp` into the Deep CFR trainer's optimization phases (`_train_advantage` and `_train_strategy`) using `torch.autocast` + `torch.amp.GradScaler`. Cut `advantage_train_seconds + strategy_train_seconds` (~7.1s/iter combined, ~40% of a non-eval iteration on the inspected default run) without regressing the learning curve.

## Non-goals

- Do not change traversal worker behavior. Traversal workers run on CPU today; AMP does not apply.
- Do not change the inference server's forward path. The server has its own `inference_server.use_amp` flag (eval-only AMP, no GradScaler). Trainer AMP and server AMP are independent.
- Do not change replay buffer dtype. Samples remain float32 in shared memory and on the host side; only the trainer's forward/backward switches to mixed precision.
- Do not change network dtype, parameter dtype, or optimizer dtype. AMP only autocasts forward; parameters and master weights remain fp32.
- Do not implement bf16 (no GradScaler needed) as the default. fp16 is the primary target; bf16 is a follow-up.

## Success criteria

1. With `--set run.use_amp=true` on `configs/deep_cfr/default.yaml` running on CUDA, `advantage_train_seconds + strategy_train_seconds` drops by at least 15% averaged over 20 non-eval iterations on `home` (RTX 3090) compared to `--set run.use_amp=false` baseline. **Realistic upper bound:** ~30% on `home`. If measured speedup is below 5%, document the result in `docs/performance.md` and leave the flag default-off (analogous to the `torch.compile` regression).
2. With `--set run.use_amp=true`, `loss/advantage` and `loss/strategy` trajectories track the fp32 baseline within seed noise over at least 50 iterations on `default.yaml`. No NaN/Inf appears in `loss/*` rows of `metrics.jsonl`.
3. Eval-side win-rate trajectories (`eval/<opponent>/win_rate0`) under AMP are indistinguishable from the fp32 baseline at iteration 50, 100, and 200 within the noise band of a single-seed comparison.
4. `--set run.use_amp=false` (the default) produces byte-identical training to `main` for the same seed: no AMP-related code path runs.
5. `run.use_amp=true` on CPU is a documented no-op (CUDA not available → skip autocast/scaler) and does not crash. Same for non-CUDA `run.device`.
6. All existing tests pass. New unit test covering the AMP code path passes.

## Key files (current)

- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`:
  - `DeepCFRTrainer.__init__` (lines ~187-258) — instantiates networks, optimizers; this is where the `GradScaler` should be created.
  - `DeepCFRTrainer._train_advantage` (lines ~892-948) — advantage forward+backward+step. AMP wrapping target.
  - `DeepCFRTrainer._train_strategy` (lines ~950-995) — strategy forward+backward+step. AMP wrapping target.
  - `DeepCFRTrainer._batch_tensors` (lines ~853-883) — tensors stay float32 (do not cast inputs).
- `src/coolrl_lost_cities/games/classic/deep_cfr/config.py`:
  - `RunConfig.use_amp: bool = False` (line 25) — flag already exists.
- `src/coolrl_lost_cities/games/classic/deep_cfr/networks.py` — no changes; `DeepCFRMLP` and `ColorSharedNetwork` work under autocast as-is.
- `tests/games/classic/test_deep_cfr_trainer.py` — extend with AMP smoke test.

## New surface

No new files. All changes live in `trainer.py` (and one optional config addition for AMP dtype selection). The plan extends `RunConfig` minimally if we want bf16 selectability; otherwise `run.use_amp` alone is sufficient and dtype is fp16 by default.

Optional `RunConfig` extension (deferred — only add if Step 5 measurement motivates it):

```python
class RunConfig(StrictModel):
    # ... existing fields ...
    use_amp: bool = False
    amp_dtype: str = "float16"   # or "bfloat16"
```

## Where AMP code goes

### Construction (in `__init__`)

After optimizers are created:

```python
self._amp_enabled = bool(self.config.run.use_amp) and self.device.type == "cuda"
self._amp_dtype = torch.float16  # bf16 deferred
self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
```

Single shared `GradScaler` across both advantage and strategy networks is fine — it tracks one global loss-scale that adjusts based on observed Inf/NaN gradients and applies to whichever optimizer it is told to step. Sharing the scaler matches PyTorch's recommended pattern for multi-network training and avoids two competing scale schedules.

### Advantage train loop (`_train_advantage`)

Replace the forward + backward + step block with:

```python
with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._amp_enabled):
    pred = network(x)
    diff = (pred - y).masked_fill(~legal, 0.0)
    if self.config.training_weighting.mode == "none":
        loss = diff.square().sum() / legal.sum().clamp_min(1)
    elif self.config.training_weighting.mode == "lcfr":
        ...
    else:
        ...

optimizer.zero_grad(set_to_none=True)
self._scaler.scale(loss).backward()
if self.config.optimization.grad_clip > 0.0:
    self._scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.optimization.grad_clip)
self._scaler.step(optimizer)
self._scaler.update()
losses.append(float(loss.detach().float().cpu()))
```

Notes:
- `legal.sum().clamp_min(1)` and the weighted-mean denominators are int/float reductions that stay fp32 — `autocast` only casts ops that are on the autocast-allow-list. The division is safe.
- `loss.detach().float().cpu()` — explicit `.float()` to avoid logging an fp16 NaN that prints surprisingly.
- The `unscale_` before `clip_grad_norm_` is mandatory; otherwise the clip threshold is applied to scaled gradients. This is the standard PyTorch AMP idiom.

### Strategy train loop (`_train_strategy`)

Same pattern. The `log_softmax` over masked logits remains numerically stable under fp16 because `masked_fill(~legal, finfo.min)` uses fp32-min before autocast can downcast — verify this; if autocast downcasts the mask fill value, replace with `torch.finfo(self._amp_dtype if self._amp_enabled else torch.float32).min` or apply the mask after autocast. Safer: apply `masked_fill` *outside* the autocast block (operating on the post-cast logits is fine — `masked_fill` is allow-listed).

### Disabled path

When `self._amp_enabled is False`, `torch.autocast(..., enabled=False)` is a true no-op (does not change dispatcher state), and `GradScaler(enabled=False)` makes `scale`, `unscale_`, `step`, and `update` all forward to plain optimizer behavior. So the same code runs in both modes — no branching needed in the hot loop.

## Flag semantics

| `run.use_amp` | `run.device` resolves to | Behavior |
| --- | --- | --- |
| `false` (default) | any | No autocast, no scaler. Behavior identical to current `main`. |
| `true` | `cuda` | Autocast(fp16) + GradScaler active in advantage/strategy train loops. |
| `true` | `cpu` | Logs a one-time warning "AMP requested but device is CPU; running fp32." Trainer behaves as `false`. |

The warning is logged via `self.tracker.log_event(...)` once at run start, after `_amp_enabled` is computed.

Trainer-side AMP is fully independent of `inference_server.use_amp`. Both can be on, both off, or either alone:

- `inference_server.use_amp` → server-process `torch.autocast` around the inference forward inside `inference_mode()`. No GradScaler (no backward pass). Affects traversal forward latency only.
- `run.use_amp` → trainer-process autocast + GradScaler around `_train_advantage` and `_train_strategy` forward+backward+step. Affects optimization phases only.

## Numerical stability considerations

CFR regret targets can have wide dynamic range — both very small (regrets near zero for converged actions) and large in magnitude (early-iteration noisy estimates). fp16's 1e-5 to 6.5e4 representable range is narrow compared to fp32, so two failure modes are realistic:

1. **Gradient overflow → NaN.** Mitigated by `GradScaler`. The scaler observes Inf/NaN in unscaled gradients, skips the step, and halves the scale. Standard.
2. **Forward overflow in the squared-error loss.** `diff.square()` on fp16 inputs can overflow if `(pred - y)` magnitude exceeds ~256. This is upstream of GradScaler — the loss itself becomes Inf in the autocast region and the entire batch is wasted. If this happens, the scaler will skip the step but the loss-scale scheduler will not recover because the issue is in the forward, not the gradient.

Mitigations for (2):

- Compute the squared error in fp32 explicitly: cast `diff` to fp32 via `diff.float()` before `.square()` if measurement shows fp16 overflow on real CFR samples. This is cheap and confined to the trainer.
- Clamp `pred - y` to a wide-but-safe range before squaring: e.g. `diff.clamp(-128, 128)`. Only do this if measurement shows it's needed — clamping silently changes loss semantics.
- Default plan: do not pre-mitigate. Add a NaN/Inf guard (see below); if it triggers in the wild, switch to the explicit `.float()` cast in the loss computation.

### NaN/Inf guard

After computing `loss` in each train step:

```python
if not torch.isfinite(loss):
    self._runtime_metrics["amp/nonfinite_loss_count"] = (
        int(self._runtime_metrics.get("amp/nonfinite_loss_count", 0)) + 1
    )
    optimizer.zero_grad(set_to_none=True)
    continue
```

This makes overflow visible in `metrics.jsonl` per iteration without aborting training. Codex should also expose `_scaler.get_scale()` as `amp/grad_scale` per iteration so we can see the scaler's scale schedule in W&B / metrics plots.

## Validation: A/B learning-curve test

After implementation, run two short trainings with the same seed:

```bash
# Baseline
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.max_iterations=100 \
  --set run.use_amp=false \
  --set run.experiment_name=amp-baseline

# AMP
uv run lost-cities-deep-cfr train \
  --config configs/deep_cfr/default.yaml \
  --keep \
  --set run.max_iterations=100 \
  --set run.use_amp=true \
  --set run.experiment_name=amp-on
```

Compare with `lost-cities-deep-cfr analyze` on each, plus a side-by-side table of:

- `loss/advantage`, `loss/strategy` at iters {25, 50, 75, 100}
- `eval/<opponent>/win_rate0` at iters {50, 100} (eval_every=25 is default)
- `time/advantage_train_seconds`, `time/strategy_train_seconds`, `time/iteration_seconds` means over iters 5-100 (iter 1-4 dropped as warm-up)
- `amp/nonfinite_loss_count`, `amp/grad_scale` (AMP run only)

Acceptance: AMP win-rates within fp32 win-rates ± noise, training-phase wall clock down by ≥15%, no nonfinite losses, scaler scale stable (not collapsing toward 1).

## Implementation steps (ordered, each independently mergeable)

### Step 1: scaler + flag plumbing in `__init__`

- In `DeepCFRTrainer.__init__`, after the optimizer block, compute `self._amp_enabled`, `self._amp_dtype`, `self._scaler`.
- Log a one-time warning if `config.run.use_amp` is requested but `self.device.type != "cuda"`.
- Add an `amp/grad_scale` runtime metric emitted once per iteration in `run_iteration` (read `self._scaler.get_scale()` after train phases).
- No behavior change yet — scaler is created but unused.

### Step 2: wrap `_train_advantage`

- Add `with torch.autocast(...)` around the forward + loss computation.
- Replace `loss.backward()` → `self._scaler.scale(loss).backward()`.
- If `grad_clip > 0`: `self._scaler.unscale_(optimizer)` before `clip_grad_norm_`.
- Replace `optimizer.step()` → `self._scaler.step(optimizer)` + `self._scaler.update()`.
- Add the NaN/Inf guard described above; emit `amp/nonfinite_loss_count` to runtime metrics.

### Step 3: wrap `_train_strategy`

- Same pattern as Step 2.
- Verify `masked_fill(~legal, torch.finfo(torch.float32).min)` semantics under autocast. Either keep the mask fill outside autocast or use a dtype-aware `finfo`.

### Step 4: unit test

- Extend `tests/games/classic/test_deep_cfr_trainer.py`:
  - `test_train_advantage_amp_smoke`: build a tiny trainer with `run.use_amp=True`, `device="cuda"` (skip if CUDA unavailable via `pytest.skip`). Seed memories with synthetic samples. Run one iteration. Assert no exception, finite `loss/advantage`, `amp/grad_scale` present in runtime metrics.
  - `test_amp_cpu_falls_back`: `run.use_amp=True`, `device="cpu"`. Assert iteration runs without error and behaves as fp32 (e.g. compare loss to a fp32 reference run on the same seed).

### Step 5: bench + learning-curve A/B

- Add a short script `scripts/bench_amp_trainer.py` (mirrors `scripts/profile_gpu_forward.py` style):
  - Construct trainer with `default.yaml`.
  - Pre-fill advantage and strategy memories with `optimization.advantage_batch_size * advantage_updates_per_iteration` synthetic samples (so the train phases run end-to-end without needing a real traversal).
  - Run `_train_advantage` and `_train_strategy` 20 times each under `use_amp=False` and `use_amp=True`. Drop the first 2 as warm-up. Print mean ms per call.
- Run the full A/B (Validation section above) and append a date-stamped subsection to `docs/performance.md` → "Experiments" with: hardware, iter-time delta, loss/win-rate parity table, NaN counts, scale schedule, decision (default-on / default-off / disabled).

### Step 6: documentation

- Update `docs/performance.md` § "AMP Status" to point to the new experiment subsection and remove the "treat as a no-op" line if AMP becomes default-on, or leave it and explain why if AMP measurement was a regression.
- No CLAUDE.md / AGENTS.md updates needed — `--set run.use_amp=true` already works syntactically.

## Definition of done

- `run.use_amp=true` on CUDA produces measurable speedup on advantage+strategy train phases on `home`, with the A/B learning curve showing parity within noise over 100 iterations.
- `run.use_amp=false` is byte-identical to `main`.
- AMP code path covered by tests; CPU fallback documented and tested.
- `metrics.jsonl` exposes `amp/grad_scale` and `amp/nonfinite_loss_count` so future runs are self-diagnosing.
- An experiment subsection is appended to `docs/performance.md` recording the speedup and the parity check.
- `uv run ruff check .` passes; `uv run pytest -q tests/games/classic/test_deep_cfr_trainer.py` passes.

## Risks and mitigations

- **Small-model regression (analogous to `torch.compile` 2026-05-07).** `DeepCFRMLP` at hidden=512, 3 layers is small. AMP overhead per call (cast/uncast, GradScaler bookkeeping) may exceed the kernel speedup at this size — the same dynamic that bit `torch.compile`. Mitigation: measure first; if speedup is below 5%, keep default off and document. Preserve the implementation on a branch (`experiments/amp-trainer`) for revisiting if `NetworkConfig.hidden_size` or `num_layers` increases.
- **fp16 overflow in `diff.square()`.** Mitigation: NaN/Inf guard logs the count; if observed, cast `diff` to fp32 before squaring inside the autocast region (`diff.float().square()`). PyTorch will not redowncast it.
- **GradScaler scale collapse.** If many consecutive steps overflow, the scaler can drop to scale=1 and stay there, defeating the point. Mitigation: log `amp/grad_scale` every iteration; if it stays ≤16 for >10 iterations, switch the loss-side cast (above) on.
- **Interaction with `clip_grad_norm_`.** Forgetting `unscale_` before clipping silently changes the effective clip threshold. Mitigation: explicit step in the implementation; covered by code review.
- **Determinism breakage.** Autocast can change op kernels and therefore reduction order; bit-exact reproducibility vs fp32 is not preserved. This is expected. Mitigation: A/B is on learning trajectories within seed noise, not on bit-identity.
- **Inference server contention.** If `inference_server.use_amp` and `run.use_amp` are both on, two autocast regions exist in two processes — they do not conflict. Trainer GradScaler does not affect server inference.

## Bench plan

Two artifacts:

1. `scripts/bench_amp_trainer.py` (new, small) — micro-bench the `_train_advantage` and `_train_strategy` methods in isolation under `use_amp=False` vs `True`. Reports per-call ms with mean + p50 + p95 over 20 runs. This is the fast feedback loop during implementation.
2. End-to-end run pair (above) — the real signal. 100-iter `default.yaml` baseline vs AMP, same seed, single GPU. Compare iter-time and learning curves.

Compare pattern matches the `torch.compile` experiment write-up in `docs/performance.md`. Use a similar table:

| | iter mean | adv+strat mean | adv+strat share | 1000-iter projection |
| --- | ---: | ---: | ---: | ---: |
| Baseline (use_amp=false) | TBD | TBD | TBD% | TBD h |
| AMP (use_amp=true) | TBD | TBD | TBD% | TBD h |
| Effect | TBD | TBD | TBD pp | TBD min |

## Out-of-scope follow-ups (do not start in this plan)

- bf16 dtype option (`amp_dtype: bfloat16`). bf16 sidesteps GradScaler entirely and avoids the fp16 overflow class. Add only if (a) hardware supports it efficiently (Ampere+) and (b) fp16 measurement shows scale collapse or frequent nonfinite losses.
- AMP on the inference server's training-side weight push path. Server already has `inference_server.use_amp` for forward; backward is not its job.
- AMP on evaluation forward. Evaluation runs `eval()` + no-grad; if eval forward becomes a bottleneck (see "Evaluation Optimization Options" #7 in `docs/performance.md`), wrap there separately.
- `torch.compile` retry. The 2026-05-07 regression was size-bound. Re-evaluate only after `NetworkConfig` grows substantially (per "Post-A Optimization Calculus" in `docs/performance.md`); coordinate with that work, not this plan.
- Larger model config experiment. AMP becomes meaningfully more useful at hidden≈1024 / layers≈6 per the post-A calculus. That is a separate model-architecture work item, not an AMP work item.
