# Deep CFR Reproducibility

**Last verified:** 2026-05-08, commit `0f85fa8`

## Summary

Deep CFR training is deterministic for short single-worker checks, but the
default multi-worker traversal path is not bitwise reproducible across repeated
runs with the same seed.

The likely source is multiprocessing result ordering, not evaluation. With
`traversal.num_workers=1`, repeated runs matched exactly for core training and
evaluation metrics across iterations 1-3. With `traversal.num_workers=1`,
inserting evaluation every iteration did not change the training trajectory.

## Evidence

### Multi-worker runs diverged despite matching seed and config

Two 512x3 runs used the same resolved training config except for
`run.experiment_name`, `run.max_iterations`, and `evaluation.eval_every`:

- `runs/2026-05-08_022808_model-size-512x3`
- `runs/2026-05-08_051124_baseline-512x3-2000-dense-eval`

They matched on iteration 1 traversal size and memory size, then diverged from
iteration 2:

| Iteration | Metric | model-size-512x3 | baseline-512x3-2000 |
| --- | --- | ---: | ---: |
| 1 | `traversal/nodes` | 169976 | 169976 |
| 1 | `memory/advantage` | 84664 | 84664 |
| 2 | `traversal/nodes` | 157664 | 173489 |
| 2 | `memory/advantage` | 163292 | 171205 |

This divergence happens before either run reaches its first evaluation point in
the 200-iteration model-size run, so evaluation frequency does not explain the
initial split.

### Single-worker repeated runs matched

Two temporary runs used:

- `traversal.num_workers=1`
- `run.max_iterations=3`
- `evaluation.eval_every=1`
- `evaluation.games=20`
- `evaluation.opponents=[random,safe_heuristic_strict]`
- W&B disabled

Runs:

- `runs/tmp/2026-05-08_152357_repro-single-worker-a`
- `runs/tmp/2026-05-08_152448_repro-single-worker-b`

Core metrics matched exactly:

| Iteration | `traversal/nodes` | `memory/advantage` | `loss/advantage` | `eval/random/win_rate0` | `eval/safe_heuristic_strict/win_rate0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 170822 | 85096 | 803.9475702643394 | 0.75 | 0.10 |
| 2 | 206225 | 187987 | 821.3355012834072 | 0.50 | 0.05 |
| 3 | 142128 | 258823 | 1089.841465830803 | 0.50 | 0.00 |

Timing counters differed, as expected.

### Evaluation did not perturb single-worker training

Two temporary runs compared eval disabled vs. eval every iteration:

- `runs/tmp/2026-05-08_152813_repro-single-worker-no-eval`
- `runs/tmp/2026-05-08_152852_repro-single-worker-with-eval`

With `traversal.num_workers=1`, common non-timing, non-eval fields matched
exactly across iterations 1-3. Evaluation added eval metrics and wall-clock
cost, but did not change:

- `traversal/nodes`
- `memory/advantage`
- `memory/strategy`
- `samples/advantage`
- `samples/strategy`
- `loss/advantage`
- `loss/strategy`

## Likely Cause

The parallel traversal path processes worker results in completion order.
`DeepCFRTrainer._run_traversals_parallel` waits for `FIRST_COMPLETED` futures,
then immediately adds the completed batch's samples into reservoir memory
(`src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py:548`).

Reservoir insertion is order-sensitive because each sample increments `seen`,
and capacity replacement draws from the trainer RNG
(`src/coolrl_lost_cities/games/classic/deep_cfr/memory.py:26`). Even before
capacity is reached, list order affects later sampled batches because memory
sampling draws indices from the same RNG
(`src/coolrl_lost_cities/games/classic/deep_cfr/memory.py:57`).

Therefore two runs can share the same seeds and configs but diverge if worker
completion order differs due to OS scheduling, process timing, or device timing.

## Implications

- Same seed does not guarantee bitwise reproducibility for default
  multi-worker Deep CFR training.
- Short deterministic checks should use `traversal.num_workers=1`.
- Multi-worker experiment comparisons should be interpreted as stochastic
  repeated runs, even when `run.seed` is identical.
- Eval frequency is not currently implicated in training trajectory divergence,
  based on the single-worker eval/no-eval check above.

## Proposed Fix

The current implementation target is stronger than same-worker-count stability:
worker count should be a performance setting, not an algorithm setting. On the
same code/config/seed and same GPU/CUDA stack, `traversal.num_workers=1`, `4`,
and `8` should produce the same non-timing metrics.

Implement that by making the logical traversal stream independent of process
scheduling:

1. Assign a stable traversal ID to every traversal, such as
   `(iteration, player, traversal_index)`.
2. Derive all traversal-local RNG streams from that traversal ID and a purpose
   token, not from worker ID, batch ID, or completion order.
3. Keep a canonical logical traversal list for each iteration. `num_workers`
   should only decide how that list is partitioned for execution.
4. Return samples and stats with their traversal IDs.
5. Buffer completed futures for an iteration.
6. Insert `advantage_samples` and `strategy_samples` into memory in sorted
   traversal ID order.
7. Accumulate stats in the same sorted traversal ID order.
8. Stabilize interleaved scheduler request/context ordering so ready queue and
   policy request processing do not depend on set/dict iteration or worker
   timing.
9. Enable PyTorch deterministic settings for reproducibility/debug runs:
   `torch.use_deterministic_algorithms(True)`,
   `torch.backends.cudnn.benchmark = False`,
   `torch.backends.cuda.matmul.allow_tf32 = False`, and
   `torch.backends.cudnn.allow_tf32 = False`.
10. Re-run the repeated-seed check across `traversal.num_workers=1`, `4`, and
   `8`.
11. If differences remain, inspect remaining CPU-side ordering and PyTorch
    operator-level nondeterminism.

This is a larger change than sorting completed worker batches, but it is the
right target if same GPU/CUDA runs should remain stable across different CPU
machines and different worker counts.

## Batch Size Scope

The deterministic target treats traversal and evaluation inference batch sizes
as execution details:

- `traversal.interleave_max_batch`
- `evaluation.batch_size`

Changing these should not change non-timing metrics once traversal IDs, request
ordering, and result merge order are stable.

Training mini-batch sizes are different. Changing
`optimization.advantage_batch_size` or `optimization.strategy_batch_size`
changes the gradient estimate and optimizer trajectory, so those values remain
ordinary experimental variables. They are not expected to match across runs.
