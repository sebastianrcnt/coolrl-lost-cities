# Deep CFR Batched Evaluation 2026-05-07

Run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_032944_deep_cfr_batched_eval_cuda_postprocess_1iter`

Command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_032944_deep_cfr_batched_eval_cuda_postprocess_1iter \
  --max-iterations 1 \
  --eval-every 1 \
  --save-latest-only
```

The config used `evaluation.batch_size: 64` and `evaluation.device: trainer`.
The base run device was CUDA.

## Summary

The run completed one training iteration and evaluated immediately.

| Metric | Value |
| --- | ---: |
| `iteration_seconds` | 21.322356 |
| `evaluation_seconds` | 14.834096 |
| `traversal_seconds` | 3.850308 |
| `advantage_train_seconds` | 1.778991 |
| `strategy_train_seconds` | 0.847232 |

Compared with the previous batch-size-1 CUDA check, evaluation time changed
from about 61.826s to about 14.834s.

## Opponent Timing

| Opponent | elapsed | avg len | policy turns | network | postprocess | encoding | legal mask | opponent act |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `safe_heuristic_strict` | 3.971 | 1000.0 | 50000 | 0.195 | 0.251 | 0.147 | 0.084 | 3.168 |
| `safe_heuristic` | 3.685 | 997.5 | 49876 | 0.199 | 0.257 | 0.148 | 0.084 | 2.868 |
| `safe_heuristic_loose` | 3.330 | 980.8 | 49038 | 0.194 | 0.250 | 0.145 | 0.082 | 2.536 |
| `noisy_safe` | 2.902 | 955.9 | 47784 | 0.195 | 0.250 | 0.144 | 0.079 | 2.109 |
| `random` | 0.797 | 640.4 | 31968 | 0.154 | 0.204 | 0.095 | 0.053 | 0.212 |
| `passive_discard` | 0.138 | 172.2 | 8562 | 0.032 | 0.042 | 0.025 | 0.014 | 0.004 |

## Totals

| Metric | Value |
| --- | ---: |
| `eval_elapsed_seconds` | 14.824448 |
| `eval_policy_turns` | 237228 |
| `eval_policy_network_seconds` | 0.968006 |
| `eval_policy_postprocess_seconds` | 1.253108 |
| `eval_policy_encoding_seconds` | 0.702534 |
| `eval_policy_legal_mask_seconds` | 0.396494 |
| `eval_opponent_act_seconds` | 10.896567 |
| `eval_apply_action_seconds` | 0.068075 |
| `eval_diagnostics_seconds` | 0.060940 |

## Notes

Batched network inference removed the previous batch-size-1 CUDA network
bottleneck. After batching, safe heuristic opponent action time became the
largest remaining eval cost for safe heuristic opponents.

The first batched implementation exposed a postprocess synchronization cost
from per-row entropy calculation. Moving entropy calculation into torch batch
postprocess reduced that cost before this run.

## Opponent Parallel Evaluation

Run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_033925_deep_cfr_batched_eval_parallel_1iter`

Command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_033925_deep_cfr_batched_eval_parallel_1iter \
  --max-iterations 1 \
  --eval-every 1 \
  --save-latest-only
```

This run used `evaluation.num_workers: 4`, `evaluation.batch_size: 64`, and
`evaluation.device: trainer` on CUDA.

| Metric | Batched sequential | Batched opponent-parallel |
| --- | ---: | ---: |
| `iteration_seconds` | 21.322356 | 12.854383 |
| `evaluation_seconds` | 14.834096 | 6.420402 |
| `traversal_seconds` | 3.850308 | 3.852627 |
| `advantage_train_seconds` | 1.778991 | 1.746623 |
| `strategy_train_seconds` | 0.847232 | 0.819998 |

Opponent elapsed values from the parallel run:

| Opponent | elapsed | opponent act | network | avg len |
| --- | ---: | ---: | ---: | ---: |
| `safe_heuristic_strict` | 4.121 | 3.225 | 0.196 | 1000.0 |
| `safe_heuristic` | 4.029 | 2.863 | 0.281 | 997.5 |
| `safe_heuristic_loose` | 3.807 | 2.631 | 0.286 | 981.6 |
| `noisy_safe` | 3.026 | 2.141 | 0.211 | 951.7 |
| `random` | 1.130 | 0.212 | 0.253 | 639.0 |
| `passive_discard` | 0.417 | 0.004 | 0.114 | 172.2 |

Parallel eval reduced measured eval wall time from about 14.83s to about
6.42s, roughly `2.31x` faster for this one-iteration profile.
