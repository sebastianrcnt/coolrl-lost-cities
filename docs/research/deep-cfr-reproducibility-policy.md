# Deep CFR Reproducibility Policy

**Last verified:** 2026-05-08, commit `0f85fa8`
**Source:** `docs/research/deep-cfr-reproducibility.md`

## Policy

Deep CFR experiments should target two different kinds of reproducibility:

1. **Debug reproducibility:** short runs should be bitwise reproducible on the
   same machine when using deterministic settings.
2. **Research reproducibility:** reported conclusions should survive repeated
   seeds and reasonable hardware differences, even when exact metric rows do
   not match bit-for-bit.

Do not treat a single seeded multi-worker GPU run as a final research result.
Use it as an exploratory signal unless it is confirmed with matched seeds.

## Hardware Expectations

Exact same metrics are not guaranteed across machines, even with the same CUDA
version. GPU model, driver, PyTorch build, CPU scheduling, multiprocessing
timing, and floating-point reduction order can all affect exact trajectories.

Expected reliability by setting:

| Setting | Expected result |
| --- | --- |
| Same machine, same GPU, same code, `traversal.num_workers=1` | Best current option for bitwise debug checks |
| Same machine, same GPU, same code, multi-worker traversal | Current implementation target: bitwise stable across repeated runs |
| Same GPU/CUDA but different CPU | Current implementation target: same seed should follow the same logical trajectory |
| Different GPU or PyTorch/CUDA build | Exact equality is not expected; use multi-seed conclusions |

## Required Experiment Practice

For exploratory experiments:

- Single-seed runs are acceptable.
- Record the resolved config, run directory, W&B URL when used, and final eval
  metrics.
- Label conclusions as provisional.

For claims worth keeping:

- Use matched seeds across baseline and treatment.
- Use at least 3 seeds; prefer 5 when runtime allows.
- Report mean and standard deviation for the key metrics.
- Keep `run.seed`, commit SHA, GPU, CUDA/PyTorch versions, and
  `traversal.num_workers` visible in the run record.

Example matched-seed comparison:

```text
baseline:  seeds 79, 80, 81
treatment: seeds 79, 80, 81
```

Compare the treatment against the baseline seed-by-seed, then report aggregate
statistics.

## Batch Size Interpretation

Not every batch size is a reproducibility-neutral setting.

Batching that should be algorithm-neutral under the deterministic target:

- traversal/interleaved policy inference batch size, such as
  `traversal.interleave_max_batch`
- evaluation inference batch size, such as `evaluation.batch_size`

These settings should affect throughput, not the logical trajectory. Same code,
seed, hardware stack, and training config should produce the same non-timing
metrics when only these batching knobs change.

Batching that is an algorithmic training setting:

- `optimization.advantage_batch_size`
- `optimization.strategy_batch_size`
- update counts or optimizer schedule parameters

These settings change gradient estimates or optimizer updates. They are
experiment variables, not reproducibility-neutral execution settings. A run with
training mini-batch size 3 is not expected to match one with training mini-batch
size 7.

## Debug Mode

Use debug mode when checking exact reproducibility or suspected regressions:

- `traversal.num_workers=1`
- `evaluation.num_workers=1`
- PyTorch deterministic settings enabled:
  - `torch.use_deterministic_algorithms(True)`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.backends.cuda.matmul.allow_tf32 = False`
  - `torch.backends.cudnn.allow_tf32 = False`
- short `run.max_iterations`, usually 3
- checkpoint writes disabled unless specifically needed
- W&B disabled

The expected check is to run the same command twice and compare non-timing
metrics in `metrics.jsonl`, including:

- `traversal/nodes`
- `memory/advantage`
- `memory/strategy`
- `loss/advantage`
- `loss/strategy`
- eval scores, if evaluation is enabled

Timing fields are not expected to match exactly.

## Implementation Priorities

Current implementation target:

1. Add a small reproducibility smoke script that runs two short debug jobs and
   compares non-timing metrics.
2. Make worker count a performance setting, not an algorithm setting:
   `traversal.num_workers=1`, `4`, and `8` should produce the same non-timing
   metrics on the same code/config/seed.
3. Assign stable traversal IDs, derive RNG streams from traversal IDs, and merge
   samples by traversal order rather than worker, batch, or completion order.
4. Stabilize interleaved scheduler request/context ordering so CPU scheduling
   differences do not change the logical traversal path.
5. Enable PyTorch deterministic settings for reproducibility/debug runs. Keep
   the speed impact explicit when comparing runtime.
6. Re-test same-seed runs across `traversal.num_workers=1`, `4`, and `8`.
7. Re-test on the same GPU/CUDA stack with a different CPU when available.

## Current Interpretation Rule

Until traversal-ID based deterministic scheduling and merge are implemented,
default multi-worker runs with the same seed may diverge. Interpret exact curves
cautiously. For research conclusions, prefer matched multi-seed comparisons over
bitwise row matching.
