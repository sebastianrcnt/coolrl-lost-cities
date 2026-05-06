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
