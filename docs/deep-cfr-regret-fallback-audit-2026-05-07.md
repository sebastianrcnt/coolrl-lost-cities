# Deep CFR Regret Fallback Audit, 2026-05-07

Goal: test whether all-negative regret matching fallback is a plausible source of
early over-opening in Lost Cities Deep CFR.

## Code Changes

- Added traversal audit metrics for regret matching fallback decisions.
- Default behavior remains unchanged: `regret_matching.all_negative_fallback: uniform`.
- Added optional fallback mode: `argmax_tiebreak`.
- Added CLI override: `--regret-fallback uniform|argmax_tiebreak`.

Key metrics:

- `traversal_regret_matching_decisions`
- `traversal_regret_fallback_count`
- `traversal_regret_fallback_rate`
- `traversal_regret_fallback_avg_depth`
- `traversal_regret_fallback_depth_bucket_<range>`
- `traversal_regret_fallback_opened_colors_count_<n>`
- `traversal_regret_fallback_action_open_new`
- `traversal_regret_fallback_open_new_selected`
- `traversal_regret_fallback_open_new_selected_rate`
- `traversal_regret_fallback_legal_actions_mean`
- `traversal_regret_fallback_legal_open_new_mean`
- `traversal_regret_fallback_legal_discard_mean`
- `traversal_regret_fallback_legal_draw_deck_mean`
- `traversal_regret_fallback_legal_draw_pile_mean`
- `traversal_regret_fallback_open_new_available_rate`
- `traversal_regret_fallback_open_new_selection_over_availability`
- `traversal_regret_fallback_avg_opened_colors_before_action`
- `traversal_regret_fallback_argmax_tie_rate`
- `traversal_regret_fallback_argmax_tie_size_mean`
- `traversal_regret_fallback_argmax_full_tie_rate`
- `traversal_regret_fallback_open_new_available_color_<color>`
- `traversal_regret_fallback_open_new_selected_color_<color>`

Implementation note: fallback policy state is captured immediately after the
network policy is computed. This avoids child recursion overwriting the
traversal-level fallback flag before the decision is recorded.

## Runs

Baseline long run, analyzed after it had reached iteration 210:

- `runs/deep_cfr/2026-05-07_legacy_align_full_depth_slot_playability`

Short comparison runs:

- `runs/deep_cfr/2026-05-07_regret_fallback_uniform_20iter`
- `runs/deep_cfr/2026-05-07_regret_fallback_argmax_tiebreak_20iter`

Both short runs used the same base config and seed:

- `configs/deep_cfr/deep_cfr_selfplay_full_depth_slot_playability.yaml`
- `seed: 79`
- `iterations: 20`
- `save_latest_only`

The 20-iteration runs below were collected before the expanded fallback timing,
legal-action composition, and tie diagnostics were added. They should be treated
as the first historical audit snapshot. New paired runs are needed to compare
the expanded metrics.

Instrumentation smoke run:

- `runs/deep_cfr/2026-05-07_regret_fallback_metrics_smoke_1iter_v2`

This run confirms the expanded metrics are emitted to `metrics.jsonl`.

## Iteration 20 Snapshot

| metric | uniform | argmax_tiebreak |
|---|---:|---:|
| traversal_regret_matching_decisions | 39,358 | 49,296 |
| traversal_regret_fallback_count | 18,362 | 7,588 |
| traversal_regret_fallback_rate | 0.4665 | 0.1539 |
| traversal_regret_fallback_open_new_selected | 641 | 164 |
| traversal_regret_fallback_open_new_selected_rate | 0.0349 | 0.0216 |
| traversal_regret_fallback_avg_opened_colors_before_action | 4.4325 | 4.6118 |
| eval_random_avg_opened_colors | 2.48 | 2.16 |
| eval_random_5_color_open_count | 49 | 35 |
| eval_safe_heuristic_avg_opened_colors | 2.50 | 2.44 |
| eval_safe_heuristic_5_color_open_count | 50 | 44 |
| eval_passive_discard_avg_opened_colors | 2.33 | 2.32 |
| eval_passive_discard_5_color_open_count | 39 | 41 |
| eval_random_avg_score_diff0 | 42.46 | 33.58 |
| eval_safe_heuristic_avg_score_diff0 | -52.65 | -58.25 |

## Read

The audit confirms that uniform fallback fires frequently in the early run.
At iteration 20, almost half of traversal regret-matching decisions use fallback
under `uniform`.

`argmax_tiebreak` sharply reduces fallback frequency and absolute fallback
open-new selections in this 20-iteration comparison. It also lowers 5-color
counts against random and safe heuristic opponents at iteration 20. The effect is
not uniform across every opponent in this very short run.

This is diagnostic evidence, not enough to promote `argmax_tiebreak` as the
default. A longer 50-100 iteration paired run is still needed before deciding
whether this fixes the plateau without hurting policy quality.
