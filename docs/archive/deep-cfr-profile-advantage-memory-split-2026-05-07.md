# Deep CFR Advantage Memory Split Profile 2026-05-07

Run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_025639_deep_cfr_profile_adv_memory_split_10iter`

Command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_025639_deep_cfr_profile_adv_memory_split_10iter \
  --max-iterations 10 \
  --save-latest-only
```

## Summary

The run completed 10 iterations. Loss values stayed finite.

Non-evaluation iterations were iterations 1-4 and 6-9:

| Metric | Before | After |
| --- | ---: | ---: |
| `iteration_seconds` | 9.132692 | 5.832958 |
| `traversal_seconds` | 3.143286 | 3.160975 |
| `memory_add_seconds` | 0.171213 | 0.157649 |
| `advantage_train_seconds` | 5.061859 | 1.742895 |
| `strategy_train_seconds` | 0.911080 | 0.912324 |
| `evaluation_seconds` | 0.000004 | 0.000004 |
| `checkpoint_seconds` | 0.015469 | 0.015756 |
| `batch_tensor_seconds` | 1.613572 | 1.553183 |
| `advantage_player_0_sample_seconds` | 1.652320 | 0.054032 |
| `advantage_player_1_sample_seconds` | 1.625927 | 0.053600 |
| `strategy_sample_seconds` | 0.061489 | 0.063131 |

Evaluation iterations were iterations 5 and 10:

| Metric | Before | After |
| --- | ---: | ---: |
| `iteration_seconds` | 22.676999 | 16.568693 |
| `traversal_seconds` | 3.233497 | 3.051501 |
| `memory_add_seconds` | 0.145742 | 0.146042 |
| `advantage_train_seconds` | 7.502700 | 1.778785 |
| `strategy_train_seconds` | 0.915639 | 0.930619 |
| `evaluation_seconds` | 11.004282 | 10.786219 |
| `checkpoint_seconds` | 0.019737 | 0.020205 |
| `batch_tensor_seconds` | 1.704181 | 1.633090 |
| `advantage_player_0_sample_seconds` | 2.839880 | 0.060441 |
| `advantage_player_1_sample_seconds` | 2.778888 | 0.061004 |
| `strategy_sample_seconds` | 0.067247 | 0.068568 |

## Per-Iteration Notes

At iteration 10, `advantage_player_0_sample_seconds +
advantage_player_1_sample_seconds` changed from about 8.231s to about 0.124s.

At iteration 10, `advantage_train_seconds` changed from about 10.230s to about
1.782s.

Traversal time stayed close to the previous profile.
