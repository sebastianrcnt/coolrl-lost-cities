# Deep CFR Profile 2026-05-07

Run directory:

`/mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_024616_deep_cfr_profile_10iter`

Command:

```bash
uv run python -m coolrl_lost_cities.games.classic.deep_cfr.cli train \
  --config configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml \
  --checkpoint-dir /mnt/2tbhdd/coolrl-lost-cities-runs/2026-05-07_024616_deep_cfr_profile_10iter \
  --max-iterations 10 \
  --save-latest-only
```

## Summary

The run completed 10 iterations. Loss values stayed finite.

Non-evaluation iterations were iterations 1-4 and 6-9:

| Metric | Average |
| --- | ---: |
| `iteration_seconds` | 9.132692 |
| `traversal_seconds` | 3.143286 |
| `memory_add_seconds` | 0.171213 |
| `advantage_train_seconds` | 5.061859 |
| `strategy_train_seconds` | 0.911080 |
| `evaluation_seconds` | 0.000004 |
| `checkpoint_seconds` | 0.015469 |
| `batch_tensor_seconds` | 1.613572 |

Evaluation iterations were iterations 5 and 10:

| Metric | Average |
| --- | ---: |
| `iteration_seconds` | 22.676999 |
| `traversal_seconds` | 3.233497 |
| `memory_add_seconds` | 0.145742 |
| `advantage_train_seconds` | 7.502700 |
| `strategy_train_seconds` | 0.915639 |
| `evaluation_seconds` | 11.004282 |
| `checkpoint_seconds` | 0.019737 |
| `batch_tensor_seconds` | 1.704181 |

## Per-Iteration Notes

`advantage_memory_size` grew from 43,019 at iteration 1 to 204,903 at
iteration 10.

`advantage_player_0_sample_seconds + advantage_player_1_sample_seconds` grew
from about 0.521s at iteration 1 to about 8.231s at iteration 10.

Traversal stayed near 3 seconds per iteration after iteration 1, except for
normal run-to-run variance.
