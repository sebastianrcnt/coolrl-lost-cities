# Regret-Matching All-Negative Fallback

**Last verified:** 2026-05-07, commit `ad0be89`

Source: `docs/archive/deep-cfr-regret-fallback-audit-2026-05-07.md`

## Question

When all action regrets at a traverser node are non-positive (which happens
frequently in early training before the advantage network has learned anything
useful), what policy should regret matching produce? The two candidate fallbacks
are `uniform` (equal probability across legal actions) and `argmax_tiebreak`
(break the all-tied-at-zero case by selecting the highest-index legal action,
effectively a near-deterministic choice). Which is correct, and does it matter
for the over-opening pathology seen in early training?

Short answer: **`uniform` is the theoretically safe default** and remains the
code default; `argmax_tiebreak` reduces fallback frequency and its side-effects
on over-opening, but the audit dataset (20 iterations) is too short to declare
it better overall.

## Code reference

The fallback is controlled by
`regret_matching.all_negative_fallback` in config
(`src/coolrl_lost_cities/games/classic/deep_cfr/config.py`) and implemented in
`src/coolrl_lost_cities/games/classic/deep_cfr/cfr_math.pyx`. The traversal
records per-iteration fallback counts and action composition through a suite of
`traversal_regret_fallback_*` metrics written to `metrics.jsonl`.

## What the audit found

At iteration 20 of an otherwise identical 20-iteration paired run:

| metric | `uniform` | `argmax_tiebreak` |
|---|---:|---:|
| fallback rate | **46.7%** | 15.4% |
| open-new selections during fallback | 641 | 164 |
| open-new selection rate during fallback | 3.49% | 2.16% |
| avg opened colors before fallback action | 4.43 | 4.61 |
| eval vs. Random: avg opened colors | 2.48 | 2.16 |
| eval vs. Random: 5-color open count | 49 | 35 |
| eval vs. Random: avg score diff | +42.5 | +33.6 |

`uniform` fires as the fallback in nearly half of all traversal regret-matching
decisions in this early-training window. During those fallback decisions, open-new
expedition actions are selected at nearly the base rate of their availability —
which is meaningfully elevated relative to an informed policy, because opening
new expeditions is usually risky in Lost Cities.

`argmax_tiebreak` reduces fallback frequency by two-thirds. The open-new
selections during fallback drop to roughly a quarter of the `uniform` count, and
eval games show lower 5-color open counts against both Random and Safe Heuristic
opponents.

## Why uniform can cause over-opening

When the advantage network output is all non-positive, `uniform` assigns equal
probability to every legal action. In Lost Cities, early in a hand, a large
fraction of legal actions are "open a new expedition." Uniform over legal actions
therefore assigns material probability mass to opening new expeditions even when
all trained regrets say "do not do this" (or say nothing, which uniform
interprets as equal preference). This early-game opening bias can propagate into
the strategy network through strategy memory samples collected during traversal.

`argmax_tiebreak` avoids that bias by collapsing the all-negative case to a
near-deterministic choice (highest legal action index), which is arbitrary but
not systematically biased toward opening.

## What the audit does not settle

The 20-iteration window is a diagnostic, not a conclusion. Two important
questions remain open:

1. **Does argmax_tiebreak help past iteration 20?** The reduction in 5-color
   openings at iteration 20 is real, but the score-diff comparison goes
   *against* `argmax_tiebreak` (+42.5 vs. +33.6 vs. Random). This suggests the
   two runs have not yet differentiated in any stable way, and the
   near-deterministic argmax choice may introduce its own early-iteration bias
   (favoring a specific action regardless of game state). A 50–100 iteration
   paired run is the stated next step before changing the default.

2. **Is over-opening caused by the fallback at all?** The fallback affects early
   iterations heavily, but other mechanisms — trajectory truncation, weak
   opponent policy, poor encoding — can also produce the same symptom. The audit
   establishes that `uniform` fallback *contributes* to open-new selections
   during traversal; it does not prove it is the primary driver of the
   over-opening plateau.

## Practical implication

- The code default (`uniform`) is safe and does not bias the algorithm in a
  theoretically incorrect direction. Regret matching is invariant to adding
  constants, so uniform over all legal actions is a valid no-information policy.
- `argmax_tiebreak` is a heuristic correction that may reduce early-game noise.
  It was used as the fallback in the `opponent_policy: network` experiments
  documented in `docs/archive/deep-cfr-opponent-policy-network-divergence-2026-05-07.md`.
- Do not switch the default to `argmax_tiebreak` based solely on the 20-iteration
  audit snapshot. Run a longer paired experiment first.
- The `traversal_regret_fallback_*` metrics are available in `metrics.jsonl` and
  provide the fine-grained action-composition data needed to evaluate any future
  change.
