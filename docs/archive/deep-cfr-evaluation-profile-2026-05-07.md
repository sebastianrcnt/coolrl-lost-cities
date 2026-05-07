# Deep CFR Evaluation Profile 2026-05-07

Run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_030634_deep_cfr_profile_eval_breakdown_10iter`

Command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_030634_deep_cfr_profile_eval_breakdown_10iter \
  --max-iterations 10 \
  --save-latest-only
```

## Summary

The run completed 10 iterations. Loss values stayed finite.

Evaluation ran on iterations 5 and 10.

| Iteration | `iteration_seconds` | `evaluation_seconds` |
| ---: | ---: | ---: |
| 5 | 16.473433 | 11.055458 |
| 10 | 16.835262 | 10.855797 |

## Opponent Averages

Values below are averaged across iterations 5 and 10.

| Opponent | elapsed | avg len | policy select | network | postprocess | encoding | legal mask | opponent act |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `safe_heuristic_strict` | 2.184393 | 158.13 | 1.628546 | 1.279717 | 0.175923 | 0.043837 | 0.038600 | 0.518953 |
| `random` | 2.072657 | 186.11 | 1.918007 | 1.515231 | 0.207319 | 0.051491 | 0.037316 | 0.112155 |
| `safe_heuristic` | 1.942907 | 143.23 | 1.471839 | 1.155444 | 0.159592 | 0.039683 | 0.035180 | 0.435870 |
| `noisy_safe` | 1.904932 | 143.71 | 1.493521 | 1.174909 | 0.161256 | 0.039845 | 0.034331 | 0.374162 |
| `safe_heuristic_loose` | 1.829347 | 136.13 | 1.400569 | 1.099200 | 0.151752 | 0.037722 | 0.033497 | 0.394891 |
| `passive_discard` | 1.014156 | 96.82 | 0.982162 | 0.774097 | 0.105579 | 0.026702 | 0.021929 | 0.006383 |

Other averaged step costs were small:

| Opponent | apply action | diagnostics | final scoring |
| --- | ---: | ---: | ---: |
| `safe_heuristic_strict` | 0.005185 | 0.008161 | 0.000569 |
| `random` | 0.006229 | 0.009001 | 0.000632 |
| `safe_heuristic` | 0.004748 | 0.007636 | 0.000581 |
| `noisy_safe` | 0.004818 | 0.007761 | 0.000584 |
| `safe_heuristic_loose` | 0.004473 | 0.007438 | 0.000592 |
| `passive_discard` | 0.002911 | 0.005156 | 0.000477 |

## Notes

`policy_select_seconds` dominated every opponent.

Inside policy selection, `policy_network_seconds` was the largest component.
`policy_postprocess_seconds` was second. `policy_encoding_seconds` and
`policy_legal_mask_seconds` were much smaller.

`opponent_act_seconds` was meaningful for safe heuristic opponents, but was
still smaller than policy network time.

`apply_action_seconds`, `diagnostics_seconds`, and `final_scoring_seconds` were
small in this run.

## CPU vs CUDA Evaluation Check

This check compared `--device cpu` and `--device cuda` on the same base
configuration after the evaluation breakdown metrics were available. The base
configuration was:

`configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml`

Both runs used one training iteration and ran evaluation on that iteration.
The base configuration's evaluation settings were kept at 100 games per
opponent and the six configured opponents.

CPU run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_eval_device_cpu_1iter`

CPU command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_eval_device_cpu_1iter \
  --max-iterations 1 \
  --eval-every 1 \
  --save-latest-only \
  --device cpu
```

CUDA run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_eval_device_cuda_1iter`

CUDA command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_eval_device_cuda_1iter \
  --max-iterations 1 \
  --eval-every 1 \
  --save-latest-only \
  --device cuda
```

Top-level timing:

| Metric | CPU | CUDA | CUDA / CPU |
| --- | ---: | ---: | ---: |
| `iteration_seconds` | 50.969 | 72.501 | 1.42 |
| `evaluation_seconds` | 38.920 | 61.826 | 1.59 |
| `traversal_seconds` | 6.343 | 6.409 | 1.01 |
| `advantage_train_seconds` | 4.054 | 2.884 | 0.71 |
| `strategy_train_seconds` | 1.643 | 1.369 | 0.83 |

Opponent elapsed timing:

| Opponent | CPU elapsed | CUDA elapsed | CUDA / CPU |
| --- | ---: | ---: | ---: |
| `random` | 4.038 | 7.220 | 1.79 |
| `passive_discard` | 0.990 | 1.773 | 1.79 |
| `safe_heuristic` | 8.701 | 13.518 | 1.55 |
| `safe_heuristic_loose` | 8.334 | 13.145 | 1.58 |
| `safe_heuristic_strict` | 9.097 | 13.882 | 1.53 |
| `noisy_safe` | 7.751 | 12.281 | 1.58 |

Opponent breakdown for the CPU run:

| Opponent | elapsed | policy turns | network | network / turn | postprocess | encoding | legal mask | opponent act | avg len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random` | 4.038 | 32170 | 2.401 | 0.075 ms | 0.586 | 0.169 | 0.114 | 0.361 | 644.4 |
| `passive_discard` | 0.990 | 8562 | 0.633 | 0.074 ms | 0.155 | 0.044 | 0.034 | 0.010 | 172.2 |
| `safe_heuristic` | 8.701 | 49756 | 3.650 | 0.073 ms | 0.903 | 0.253 | 0.211 | 3.068 | 995.1 |
| `safe_heuristic_loose` | 8.334 | 48958 | 3.592 | 0.073 ms | 0.886 | 0.248 | 0.207 | 2.798 | 979.2 |
| `safe_heuristic_strict` | 9.097 | 50000 | 3.678 | 0.074 ms | 0.910 | 0.254 | 0.214 | 3.420 | 1000.0 |
| `noisy_safe` | 7.751 | 47276 | 3.501 | 0.074 ms | 0.864 | 0.244 | 0.195 | 2.354 | 945.7 |
| Total | 38.911 | 236722 | 17.456 | 0.074 ms | 4.304 | 1.212 | 0.975 | 12.011 |  |

Opponent breakdown for the CUDA run:

| Opponent | elapsed | policy turns | network | network / turn | postprocess | encoding | legal mask | opponent act | avg len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random` | 7.220 | 32096 | 5.337 | 0.166 ms | 0.716 | 0.179 | 0.129 | 0.384 | 642.9 |
| `passive_discard` | 1.773 | 8562 | 1.364 | 0.159 ms | 0.185 | 0.047 | 0.039 | 0.012 | 172.2 |
| `safe_heuristic` | 13.518 | 49876 | 8.053 | 0.161 ms | 1.097 | 0.274 | 0.239 | 3.139 | 997.5 |
| `safe_heuristic_loose` | 13.145 | 49078 | 7.967 | 0.162 ms | 1.080 | 0.271 | 0.236 | 2.890 | 981.6 |
| `safe_heuristic_strict` | 13.882 | 50000 | 8.034 | 0.161 ms | 1.096 | 0.274 | 0.239 | 3.524 | 1000.0 |
| `noisy_safe` | 12.281 | 47266 | 7.650 | 0.162 ms | 1.044 | 0.262 | 0.219 | 2.421 | 945.5 |
| Total | 61.819 | 236878 | 38.406 | 0.162 ms | 5.218 | 1.307 | 1.101 | 12.370 |  |
