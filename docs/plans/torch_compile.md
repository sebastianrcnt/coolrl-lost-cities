# Plan: Revisit `torch.compile` on Deep CFR Trainer (and Inference Server Forward)

**Status:** Conditional — gated on a model-size bump. Do not implement against the current `default.yaml` (512 hidden / 3 layers); a regression has already been measured at that size.
**Owner:** Codex
**Background:** See `docs/performance.md`:
- "Experiments → `torch.compile` on trainer networks (2026-05-07, regression)" — the prior attempt regressed iter time by +4.8% on the small default model. Implementation is preserved on branch `experiments/torch-compile`.
- "Post-A Optimization Calculus (forward-looking, 2026-05-07)" — argues compile/TensorRT become meaningful only after model size grows out of the dispatch-bound regime and/or eval density rises. This plan honors that sequencing.
- "Clarifying the traversal bottleneck: sync policy boundary, not SIMD" — the current traversal bottleneck is scheduling shape, not a compile-able model-kernel problem.

This plan is the follow-up referenced in step 3 of "Recommended sequencing".

## Goal

Re-enable `torch.compile` on the Deep CFR trainer's networks at a model size where kernel work amortizes compile dispatch overhead, and (optionally, secondary) on batched eval/inference-server forward paths. The goal is iter-time speedup with **no learning-curve drift**.

## Non-goals

- Do not enable compile on the current 512-hidden / 3-layer `default.yaml`. The regression is already measured.
- Do not compile the Cython traversal call-site networks (workers' CPU networks under `inference_backend: local`). Their per-call shape is `batch_size == 1` and dispatch-dominated; compile cannot help and recompilation triggers are higher-risk.
- Do not introduce TensorRT here. TensorRT is a separate work item for batched inference/eval surfaces (see `docs/performance.md` § "Tooling split").
- Do not change network architectures. This plan picks up whatever larger model the project settles on in step 2 of the post-A sequencing.
- Do not change the public CLI surface. Compile toggles via config only.
- Do not enable `mode="max-autotune"` by default. It is opt-in for benchmarking.

## Success criteria

1. **Model-size precondition met.** The active `default.yaml` (or the targeted variant) has `network.hidden_size ≥ 1024` *or* `network.num_layers ≥ 6`, *or* an architecture (e.g. `color_shared` with non-trivial `color_attention_layers`) whose per-call forward time exceeds ~150 μs at the trainer's training batch size on the target GPU. If neither condition holds, this plan is **not merged**; the branch is parked.
2. **Iter-time improvement.** With `compile.trainer.enabled: true` on the chosen larger model and the same seed, the 1000-iter projection improves by at least **5%** vs the no-compile baseline on the same machine (measured on `home`). Eval and checkpointing should be disabled for the bench window, matching the protocol used in the 2026-05-07 experiment.
3. **No learning-curve drift.** Over at least 100 iterations with `compile.trainer.enabled: true` vs `false` (same seed, same config), the eval win-rate trajectories against `random` and `safe_heuristic` are within seed noise. If trajectories visibly diverge, the plan does not ship even if iter time improves.
4. **No checkpoint-format break.** Checkpoints saved with compile enabled must load cleanly when compile is disabled, and vice versa. (Handled via `_clean_state_dict()`; see Risks.)
5. **No multiprocessing-worker break.** Whether `inference_backend` is `local` or `server`, traversal workers must continue to receive uncompiled `state_dict`s without `_orig_mod.` prefixes.
6. **(Secondary) Inference-server forward.** If step B below is taken, the server's `policy_network_seconds` decreases by at least 20% at the chosen model size, with no traversal-path correctness regression. If step B does not produce a measurable win, it is left disabled and the plan still ships with step A only.
7. All existing tests pass. Lint clean (`uv run ruff check .`). The relevant Deep CFR test subset passes.

## Why this is gated on a larger model (explicit dependency)

The 2026-05-07 experiment already isolated the failure mode: at 512 hidden / 3 layers / ReLU MLP, the per-call forward is ~80 μs at `batch_size=1` and ~90 μs in the bs≤256 plateau (see "GPU forward profiling for batched traversal", `docs/performance.md`). Compile dispatch overhead, plus the cudagraph/decomposition path, costs more than the kernel-fusion benefit at that size. The trainer's training batches *are* larger than 1 (`optimization.advantage_batch_size` / `optimization.strategy_batch_size`), so the dispatch-dominated regime ends sooner there than for traversal — but the prior measurement shows it still does not pay at 512×3.

Compile becomes meaningful when one of these is true:

- **Wider/deeper MLP.** `hidden_size ≥ 1024` *or* `num_layers ≥ 6`. At that point the per-layer matmul is large enough that fused epilogue gains (linear+activation) and reduced Python-level dispatch overhead exceed compile's per-call cost.
- **Non-trivial attention.** `color_shared` with `color_attention_layers ≥ 2` introduces `nn.TransformerEncoderLayer`, which is one of the architectures where `torch.compile` reliably wins (LayerNorm + softmax + GEMM fusion).
- **Backward+optimizer fusion.** Compile can fuse parts of the optimizer step on larger models. The trainer phase compiles forward+backward+optimizer together; this is where the biggest absolute wins live, but only above the dispatch-bound floor.

**Threshold rule (hard gate):** do not merge this plan unless at least one of the following is true on the config it targets:

- `network.hidden_size ≥ 1024`, or
- `network.num_layers ≥ 6`, or
- `network.kind == "color_shared"` with `color_attention_layers ≥ 2`.

If none hold, park the branch and re-evaluate at the next model-size bump. This explicit gate is required by the "Post-A Optimization Calculus" reasoning in `docs/performance.md`.

## Two compile targets (treated separately)

These have different dispatch profiles and different failure modes. Each is independently mergeable; step A is the primary objective, step B is secondary.

### A. Trainer networks (forward + backward + optimizer)

Wraps `advantage_networks[player]` and `strategy_network` at trainer construction time with `torch.compile(...)`. The compiled wrapper sits in front of the optimization loop in `trainer.py`. Workers (under `inference_backend: local`) and the inference server (under `inference_backend: server`) keep using uncompiled networks built from cleaned `state_dict`s.

Why this target: the trainer's optimization step has fixed batch shapes (no dynamic shapes), runs many steps per iteration, and exercises forward+backward+optimizer — the regime where compile pays best when the kernel is large enough.

### B. Batched inference/eval forward (eval-mode, no grad)

Optional follow-up. The Option A inference server has landed and benchmarked,
but it is structurally capped by sync-blocking traversal and is not the default.
Only compile the server's `model.forward` path after either (a) Option B-style
interleaved traversal can feed meaningful batches, or (b) the target is
evaluation, which already has batching. The server/eval path uses `eval()` +
`inference_mode()`. Batches vary in size, which introduces a dynamic-shape
concern (see Risks).

Why this target is secondary: per `docs/performance.md` § "Why compile / TensorRT are negligible *today* but become meaningful later", small-model inference forward is not the iter-level limiter unless traversal can actually feed batched requests. At larger model sizes or denser evaluation, compile and TensorRT compete for the same role; this plan covers compile, and the TensorRT plan (separate, future) covers TensorRT.

## Compile mode selection

Start with `mode="default"` for both targets. Evaluate `mode="reduce-overhead"` and `mode="max-autotune"` only after a baseline number is in.

- **`default`** — safe. Lowest compile time. First measurement.
- **`reduce-overhead`** — uses CUDA graphs. Helps small-batch regimes by amortizing launch overhead. **Incompatible with multiprocessing in non-trivial ways**: the captured graph holds CUDA stream/state from the capturing process. The trainer process is the only place this mode would be used (workers do not compile); confirm the trainer's compiled call is not entangled with worker process spawn (it should not be — workers are already spawned with cleaned `state_dict`s before any compiled call). Use `reduce-overhead` only on the trainer phase, never on a target that crosses a `mp.spawn` boundary.
- **`max-autotune`** — autotunes kernel selection. Long compile time (minutes). Only worth it on a stable, frozen model config that will be trained for many hours. Run as an A/B against `default` after the headline result is established.

For the inference server (target B), `default` is the only safe mode initially. `reduce-overhead` requires fixed-shape inputs; the server's batch dimension varies up to `max_batch`, so cudagraphs would either recompile per-shape or require pre-padding to `max_batch`. Treat that as a separate experiment.

## Key files

- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py` — owns trainer-side networks, optimizer, and the train loop. This is where target A's `torch.compile(...)` calls go, plus the `_clean_state_dict()` helper and the `_orig_mod`-routed `load_state_dict` shim. Reference implementation lives on branch `experiments/torch-compile` (commit `05cc02a`).
- `src/coolrl_lost_cities/games/classic/deep_cfr/networks.py` — network classes. Not modified by this plan; compile wraps the constructed module from outside.
- `src/coolrl_lost_cities/games/classic/deep_cfr/workers.py` — multiprocessing traversal worker entry. Reconstructs CPU networks from `state_dict`s. Must keep receiving cleaned (no-`_orig_mod.`) state dicts.
- `src/coolrl_lost_cities/games/classic/deep_cfr/inference_server.py` — created by the archived Option A plan. Target B wraps `model.forward` here only after Option B or eval batching provides large enough batches.
- `src/coolrl_lost_cities/games/classic/deep_cfr/config.py` — extend with a `CompileConfig` block.

## Reference implementation (branch `experiments/torch-compile`)

The prior implementation (commit `05cc02a` on `experiments/torch-compile`) provides the working pattern. Specifically:

- `_clean_state_dict()` helper strips the `_orig_mod.` prefix that `torch.compile`'s `OptimizedModule` adds to `state_dict()` keys. All checkpoint writes and all weight pushes to multiprocessing workers / the inference server go through this helper.
- `load_state_dict` on a compiled wrapper is routed via `module._orig_mod.load_state_dict(...)` so it accepts clean state dicts (saved checkpoints have no prefix).

This plan **reuses these helpers verbatim**. Cherry-pick the trainer changes from that branch as the starting point and rebase onto current `main`.

## Config schema

Extend `config.py`:

```python
@dataclass
class CompileTrainerConfig:
    enabled: bool = False
    mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    fullgraph: bool = False           # safer to start False; may flip True after stabilization
    dynamic: bool = False             # trainer batches are fixed shape

@dataclass
class CompileInferenceServerConfig:
    enabled: bool = False
    mode: Literal["default", "reduce-overhead"] = "default"
    fullgraph: bool = False
    dynamic: bool = True              # batch dim varies up to max_batch

@dataclass
class CompileConfig:
    trainer: CompileTrainerConfig = field(default_factory=CompileTrainerConfig)
    inference_server: CompileInferenceServerConfig = field(default_factory=CompileInferenceServerConfig)
```

In `default.yaml`, leave both `enabled: false`. Add an explicit benchmarking variant (e.g. `configs/deep_cfr/default_compile.yaml`) that flips `compile.trainer.enabled: true` and bumps the model size to meet the threshold rule.

## Checkpoint and weight-sync handling (the `_orig_mod.` prefix)

This is the single most error-prone part of the work. The contract is:

1. **Saved checkpoints are always uncompiled-shaped.** Before saving, run `_clean_state_dict(module.state_dict())` to strip `_orig_mod.`. Files saved with compile enabled must load fine when compile is disabled.
2. **Loaded checkpoints are always uncompiled-shaped.** When loading into a compiled wrapper, route through `module._orig_mod.load_state_dict(clean_dict)`.
3. **Weights pushed to multiprocessing traversal workers** (under `inference_backend: local`) are always uncompiled-shaped. Workers reconstruct an uncompiled `DeepCFRMLP` and call `load_state_dict` on it.
4. **Weights pushed to the inference server** (under `inference_backend: server`) are always uncompiled-shaped. The server may *separately* wrap its loaded model with `torch.compile` (target B). The wire format is uncompiled.
5. **Resume from a checkpoint trained with a different `compile.*` setting** must work in either direction. Test this explicitly.

These rules mean compile is purely a runtime optimization; it never appears in any persisted artifact and never crosses a process boundary.

## Trainer integration (target A)

In `trainer.py`, after constructing `advantage_networks[player]` and `strategy_network` and binding their optimizers:

```python
if config.compile.trainer.enabled:
    compile_kwargs = dict(
        mode=config.compile.trainer.mode,
        fullgraph=config.compile.trainer.fullgraph,
        dynamic=config.compile.trainer.dynamic,
    )
    self._advantage_networks = [
        torch.compile(net, **compile_kwargs) for net in self._advantage_networks
    ]
    self._strategy_network = torch.compile(self._strategy_network, **compile_kwargs)
```

Caveats:

- The training step calls `loss.backward()` and `optimizer.step()`. Compile fuses across the forward, but backward and optimizer paths are separately traced via Dynamo's autograd hooks. Confirm by inspecting `torch._dynamo.config.cache_size_limit` is not being hit (recompilation noise).
- All `state_dict()` writes (checkpoint, weight push, eval snapshot) go through `_clean_state_dict()`.
- All `load_state_dict()` reads route through the `_orig_mod` shim when targeting a compiled wrapper.

## Inference-server integration (target B, optional)

In `inference_server.py`, after the server process loads the model on `device` and sets `eval()` + `inference_mode()`:

```python
if compile_cfg.inference_server.enabled:
    self._model = torch.compile(
        self._model,
        mode=compile_cfg.inference_server.mode,
        fullgraph=compile_cfg.inference_server.fullgraph,
        dynamic=compile_cfg.inference_server.dynamic,
    )
```

Caveats:

- Server already runs in a child process spawned via `torch.multiprocessing`. Compile is invoked **inside the child**, never in the parent. This avoids the cudagraph-across-spawn issue.
- The server batches request rows into a `[k, input_dim]` tensor where `k ∈ [1, max_batch]`. Set `dynamic=True` so Dynamo does not recompile per `k`. If recompilation is observed, pad batches up to a small set of bucket sizes (e.g. powers of two) before forward.
- `mode="reduce-overhead"` is **not safe** here without bucketing; cudagraphs require fixed shapes.
- Weight sync: when the trainer pushes a fresh `state_dict`, the server must apply it via `self._model._orig_mod.load_state_dict(clean_dict)` if compile is enabled, else `self._model.load_state_dict(clean_dict)`.

## Implementation steps (ordered, each independently mergeable)

### Step 0: precondition check

- Before doing anything, confirm the active model config (or the variant being targeted) meets the threshold rule above. If it does not, **stop**. This plan does not ship against the small default.

### Step 1: cherry-pick reference implementation (target A)

- Cherry-pick `experiments/torch-compile` (commit `05cc02a`) onto a fresh branch. Resolve any conflicts against current `main`.
- Move the unconditional `torch.compile(...)` calls behind `config.compile.trainer.enabled`. Default is `false`.
- Add the `CompileConfig` and `CompileTrainerConfig` dataclasses to `config.py`. Wire `--set compile.trainer.enabled=true` through.
- Confirm `_clean_state_dict()` and the `_orig_mod`-routed `load_state_dict` paths are correctly invoked on every checkpoint save, every checkpoint load, every weight push to workers, and every weight push to the inference server if target B is enabled.

### Step 2: tests for state-dict round-tripping

- Unit test: build a `DeepCFRMLP`, wrap with `torch.compile`, call `_clean_state_dict(model.state_dict())`, build a fresh uncompiled `DeepCFRMLP`, `load_state_dict` from the clean dict, assert parameter equality.
- Unit test: build a compiled model, `load_state_dict` from a clean dict, assert no error and parameters match.
- Integration test: short training run with `compile.trainer.enabled=true`, save checkpoint, resume training with `compile.trainer.enabled=false`, assert no parameter mismatch on load.

### Step 3: precondition guardrail

- In `trainer.py`, when `compile.trainer.enabled=true`, log a warning at startup if the model fails the threshold rule (hidden < 1024 and layers < 6 and not color-attention). Do not error — operators may want to bench against the threshold — but make the regression risk explicit.

### Step 4: bench (target A)

See "Bench plan" below. This is the gating step. If results do not clear success criterion 2 and 3, do not merge target A.

### Step 5: trainer compile shipped behind config flag

- Once Step 4 is green, enable `compile.trainer.enabled=true` in the larger-model config that became the new `default.yaml` (or the dedicated `default_compile.yaml`). Do not flip it on for any small-model config.

### Step 6: (optional) inference-server forward compile (target B)

- Only if Option B or evaluation batching can feed large enough batches to make the inference-server forward a meaningful target. The archived Option A server exists, but the sync-blocking traversal path did not feed large batches.
- Add `CompileInferenceServerConfig` wiring. Add `torch.compile(...)` invocation inside the server process. Route weight sync through `_orig_mod` when enabled.
- Add a small inference-server bench script (or extend `scripts/bench_inference_backend.py` if it has landed) to measure server-side `policy_network_seconds` with and without compile at the chosen larger model size.
- Ship only if criterion 6 is met. Otherwise leave disabled.

### Step 7: documentation

- Add a date-stamped experiment subsection to `docs/performance.md` recording the bench result at the new model size, mirroring the existing 2026-05-07 entry's structure.
- If the result is a regression at the chosen model size, document it and park the branch again with a note about which threshold to revisit.

## Bench plan

Mirror the 2026-05-07 protocol exactly so results are comparable to the prior data point.

- **Hardware**: `home` (6-core + RTX 3090). Confirm baseline numbers on `remote` separately if/when applicable.
- **Config**: the larger-model variant that meets the threshold rule. Disable eval and checkpointing for the bench window (`--set checkpoint.save_latest=false --set checkpoint.save_every=0` and an eval-disabling override).
- **Length**: at least 8 iterations measured, with iteration 1 dropped as compile warm-up. Replicate the 2026-05-07 table format:

  | | iter mean | 1000-iter projection |
  | --- | ---: | ---: |
  | Baseline (no compile) | … | … |
  | `torch.compile` trainer (mode=default) | … | … |
  | Effect | … | … |

- **A/B**: same seed both runs. Same machine, same GPU, no concurrent jobs (per AGENTS.md).
- **Drift check (criterion 3)**: run a separate 100-iter A/B with `eval.eval_every=25` enabled, same seed, and overlay the eval win-rate trajectories from `metrics.jsonl`. If the trajectories diverge beyond seed noise, target A does not ship even if the iter-time A/B looked good.
- **Mode sweep**: only after `mode="default"` clears the bar, also bench `mode="reduce-overhead"` (trainer only) and `mode="max-autotune"` (trainer only). Record both.

For target B, bench `policy_network_seconds` from the inference-server side with and without compile at the chosen larger model size. The report does not need to wait for a learning-curve A/B — the server is eval-mode only and pushes uncompiled weights every iter, so it cannot drift training.

## Risks

- **Recompilation triggers.** If `torch._dynamo.config.cache_size_limit` is hit, the compiled wrapper falls back to eager and the run silently regresses. Mitigation: log Dynamo recompile events at startup; fail loudly if recompile count exceeds a small threshold during the bench window. Trainer batches are fixed-shape (`optimization.*_batch_size`), so this should not trigger for target A. For target B, set `dynamic=True` and watch for recompilation across batch sizes.
- **`_orig_mod.` prefix leaking into checkpoints or worker state_dicts.** Most likely bug. Mitigation: tests in Step 2 explicitly guard this. The reference implementation already handles it.
- **`reduce-overhead` × multiprocessing.** Cudagraphs in `reduce-overhead` mode capture CUDA stream/context state. They are safe inside the trainer process (no spawn after compile) and inside the inference-server child process (compile happens after spawn). They are **not** safe if compile is invoked before `mp.spawn` and the child inherits compiled state. The plan only invokes compile after spawn boundaries.
- **Dynamic shapes on inference server (target B).** Server batches vary in size up to `max_batch`. Mitigation: `dynamic=True`. Fallback: pad to fixed bucket sizes.
- **Backward path not benefiting.** Compile's biggest theoretical wins on the trainer phase come from fusing forward+backward+optimizer. In practice on simple MLPs the optimizer is already fused (e.g. `torch.optim.Adam(..., fused=True)` if available); compile may add little on top. Mitigation: bench is the answer — if iter time does not move 5%, the plan does not merge. This is a real possibility.
- **CFR variable-length traversal does not feed compile.** Confirmed: traversal call sites use `batch_size == 1` and run on CPU workers (or via the inference server, which is target B not target A). Target A only sees the trainer optimization batches, which are fixed-shape. So dynamic-shape concerns do not apply to target A.
- **GPU contention with eval.** Eval already runs on the trainer's device. Compile increases peak memory during compile/autotune phases (especially `max-autotune`). Mitigation: compile happens once per process at startup; eval runs after warm-up. Bench with eval disabled to isolate iter time, then re-enable for the drift check.
- **AMP interaction.** Trainer AMP is implemented but default-off after the 2026-05-07 smoke regression. If the model-size experiment later makes AMP attractive, re-bench compile with and without AMP because the two interact and prior numbers do not transfer.

## Decision tree

- **Threshold rule fails on the active model.** Park the branch. Do not merge. Re-evaluate at the next model-size bump.
- **Threshold rule passes; bench shows ≥5% iter improvement and no learning-curve drift.** Ship target A behind the config flag, enable on the larger-model config.
- **Threshold rule passes; bench shows iter improvement but learning-curve drift.** Do not ship. Investigate determinism (compile mode, autograd path, optimizer fusion). Likely cause: optimizer-fusion change altering update order. If cause cannot be isolated within reasonable effort, park.
- **Threshold rule passes; bench shows <5% iter improvement.** Do not ship. The cost (state-dict gymnastics, recompile risk, checkpoint compatibility surface) is not justified by sub-5% wins. Park.
- **Threshold rule passes; bench shows regression.** Park the branch with a documented `docs/performance.md` entry. Note the model size at which the regression was observed and the next threshold to try.
- **Target A shipped; target B (inference-server compile) bench shows <20% server-forward improvement.** Leave target B disabled. Revisit alongside the TensorRT plan, since they target the same surface.

## Out-of-scope

- **TensorRT.** Separate plan. TensorRT and `torch.compile` on the inference-server forward are alternatives covering the same surface (target B). This plan covers compile only; the TensorRT plan covers TensorRT.
- **Compiling the Cython traversal call-site networks.** Per the 2026-05-07 experiment, this is dispatch-bound and unhelped by compile. Workers under `inference_backend: local` will keep using uncompiled CPU networks indefinitely.
- **Compiling the encoding path.** Encoding is numpy/Cython, not a `torch.nn.Module`.
- **`torch.export` / AOTInductor.** Out of scope until the model architecture is frozen and an offline-compiled artifact is operationally needed.
- **Multi-GPU.** Not relevant to the current single-GPU setup.
- **`fullgraph=True`.** Default `fullgraph=False` until stabilization. Promoting to `fullgraph=True` is a follow-up after one full training run completes cleanly with compile enabled.
