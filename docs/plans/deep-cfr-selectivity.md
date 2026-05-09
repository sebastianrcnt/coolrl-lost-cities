# Deep CFR Selectivity Investigation

Last updated: 2026-05-09

## Current conclusion

The current Deep CFR baseline is not mainly blocked by model size or LCFR
time-weighting. The strongest signal so far is a selectivity failure: the model
does not reliably learn to distinguish good first opens from bad first opens.

The `outcome_sampling_epsilon=0.05` setting produced a short-run improvement
around 200 iterations, but the improvement did not hold through 500 iterations.
Weighting ablations did not recover the 200-iteration peak, so further LCFR
alpha tuning is lower priority than directly inspecting or changing the
first-open advantage target.

## Baseline symptoms

The 512x3 dense-eval baseline showed improving training losses, but the main
game-quality metrics against `safe_heuristic_strict` did not improve enough to
indicate a useful policy.

Observed pattern:

- Advantage loss can move in the expected direction while eval quality remains
  poor.
- Bad first-open behavior stays high.
- Open quality, measured by score per opened color, stays negative.
- Current-policy evaluation did not rescue the result, so the issue is not only
  average-policy lag.

## Closed hypotheses

### More outcome sampling helps, but only up to a point

Short 200-iteration ablations with `outcome_unsampled_regret=zero`:

| setting | safe strict score diff | win rate | bad open rate | score/opened color |
| --- | ---: | ---: | ---: | ---: |
| `epsilon=0.20` | -57.87 | 0.07 | 0.901 | -8.40 |
| `epsilon=0.10` | -58.66 | 0.05 | 0.913 | -8.63 |
| `epsilon=0.05` | -40.01 | 0.12 | 0.893 | -6.25 |
| `epsilon=0.02` | -53.55 | - | - | - |

Conclusion: `epsilon=0.05` was the best short-run candidate. Lowering to
`0.02` was worse, and larger values were also worse.

### Negative unsampled regret was not sufficient

The `epsilon=0.05` plus `outcome_unsampled_regret=negative_node_value` variant
ended around safe strict score diff `-44.53` at 200 iterations, worse than
`epsilon=0.05` with `zero`.

Conclusion: the negative target suppresses opening, but it does not selectively
preserve good opens.

### The 200-iteration epsilon=0.05 peak did not hold

Confirmation run:

- Run: `runs/2026-05-09_010747_confirm-eps-005-zero-512x3-det-500`
- W&B group: `eps005-confirmation-512x3-v1`
- Config delta:
  - `run.deterministic=true`
  - `run.max_iterations=500`
  - `traversal.outcome_sampling_epsilon=0.05`
  - `traversal.outcome_unsampled_regret=zero`

Safe heuristic strict metrics:

| iteration | win rate | score diff | bad open rate | score/opened color |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0.12 | -40.01 | 0.893 | -6.25 |
| 250 | 0.08 | -46.91 | 0.900 | -7.29 |
| 300 | 0.08 | -45.93 | 0.895 | -6.30 |
| 350 | 0.07 | -54.02 | 0.899 | -7.83 |
| 400 | 0.04 | -49.64 | 0.912 | -7.83 |
| 450 | 0.06 | -49.31 | 0.917 | -6.71 |
| 500 | 0.06 | -61.24 | 0.903 | -8.30 |

Conclusion: `epsilon=0.05` creates a real short-run peak, but the behavior is
not stable through 500 iterations.

### LCFR time-weighting is not the sole cause of the degradation

Weighting ablation group:

- W&B group: `eps005-weighting-ablation-512x3-v1`
- Common config:
  - `run.deterministic=true`
  - `run.max_iterations=300`
  - `traversal.outcome_sampling_epsilon=0.05`
  - `traversal.outcome_unsampled_regret=zero`

Compared runs:

| run | iter 200 diff | iter 250 diff | iter 300 diff | iter 300 win | iter 300 bad open |
| --- | ---: | ---: | ---: | ---: | ---: |
| LCFR alpha=1.0 confirmation | -40.01 | -46.91 | -45.93 | 0.08 | 0.895 |
| `training_weighting.mode=none` | -50.98 | -48.46 | -56.27 | 0.04 | 0.912 |
| `training_weighting.lcfr_alpha=0.5` | -66.81 | -58.97 | -50.77 | 0.10 | 0.891 |

Open-quality comparison:

| run | iter 200 score/opened color | iter 300 score/opened color |
| --- | ---: | ---: |
| LCFR alpha=1.0 confirmation | -6.25 | -6.30 |
| `training_weighting.mode=none` | -7.71 | -7.70 |
| `training_weighting.lcfr_alpha=0.5` | -9.94 | -7.85 |

Conclusion: neither removing time-weighting nor softening LCFR to `alpha=0.5`
beat the original `alpha=1.0` confirmation by the primary score-diff metric at
300 iterations. LCFR may affect stability, but it is not the main lever.

## First-open diagnostic

Diagnostic output:

- `runs/tmp/first_open_advantage_confirm_eps005_200_vs_500.jsonl`

Observed values:

| checkpoint | bad selection | good selection | bad advantage | good advantage | sampled turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| iter 200 | 0.0625 | 0.0582 | -12.82 | -16.60 | 8560 |
| iter 500 | 0.0067 | 0.0067 | -35.42 | -44.32 | 40942 |

Interpretation: by 500 iterations the model strongly suppresses opening
overall. It suppresses good opens along with bad opens, which is the core
selectivity failure.

## First-open target audit

Script:

- `scripts/analyze_first_open_targets.py`

Output:

- `runs/tmp/first_open_target_audit_confirm_eps005_200_vs_500.jsonl`

Method: regenerate short interleaved traversal batches from existing
checkpoints and bucket first-open advantage targets by action quality. Because
`outcome_unsampled_regret=zero` sets unsampled legal actions to zero, the most
informative statistic is the sampled-action target distribution, not the full
legal-candidate target distribution.

Sampled target summary:

| checkpoint | bucket | candidates | policy prob | sampled rate | sampled target mean | sampled target positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iter 200 | good open | 796 | 0.078 | 0.078 | -40.71 | 0.419 |
| iter 200 | bad open | 7142 | 0.081 | 0.081 | -25.65 | 0.424 |
| iter 500 | good open | 800 | 0.064 | 0.058 | 8.39 | 0.435 |
| iter 500 | bad open | 8322 | 0.069 | 0.071 | 12.61 | 0.433 |

Interpretation: the regenerated traversal targets do not rank good opens above
bad opens. At both inspected checkpoints, bad-open candidates receive slightly
higher policy probability and sampled rate than good-open candidates. The
sampled target mean is also better for bad opens than good opens. This points
to a target or metric-alignment problem before model capacity or LCFR tuning.

## First-open counterfactual audit

Script:

- `scripts/analyze_first_open_counterfactual.py`

Output:

- `runs/tmp/first_open_counterfactual_confirm_eps005_200_vs_500.jsonl`

Method: collect first-open candidate states from existing checkpoints, force
each first-open candidate once, and compare the resulting continuation value
against the current policy's best non-open action from the same state.

`delta_open = value(force open) - value(best non-open)`.

Counterfactual summary against `safe_heuristic_strict`:

| checkpoint | bucket | candidates | delta mean | delta median | delta positive | policy prob | selected rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iter 200 | good open | 40 | -26.57 | -26.5 | 0.050 | 0.000 | 0.000 |
| iter 200 | bad open | 460 | -23.15 | -21.0 | 0.130 | 0.036 | 0.037 |
| iter 500 | good open | 30 | -23.43 | -17.5 | 0.167 | 0.067 | 0.067 |
| iter 500 | bad open | 470 | -11.18 | -8.0 | 0.226 | 0.020 | 0.019 |

Interpretation: the heuristic `open_bad` label is not entirely misaligned with
continuation value. Forced bad opens are usually worse than the best non-open
alternative. However, heuristic `open_good` also often loses to best non-open in
these sampled states, so "recoverable eventually" is not the same as "open now."

Combined with the target audit, this points toward target/objective alignment:
the traversal target is not making the bad-open-vs-non-open mistake clearly
negative, even when the counterfactual continuation usually is negative.

## Open questions

1. Does the traversal target itself provide separable labels for good first
   opens versus bad first opens?
2. Is the model receiving too sparse or too noisy a signal at the first-open
   decision point?
3. Would target shaping around first-open decisions improve score/opened color
   without increasing bad-open rate?
4. Is evaluation showing a policy-selection problem, or is the advantage model
   already misranking good and bad opens before strategy extraction?

## Recommended next experiments

### 1. Deeper first-open target audit

Before another long training run, inspect sampled first-open decision records
more directly:

- Group candidate first-open actions into good-open and bad-open buckets.
- Compare target values before model prediction, not only final learned
  advantages.
- Report distributions, not just means.
- Check whether the target ranks good opens above bad opens in the same
  information-state context family.

Success criterion: the target distribution should show a usable separation
between good and bad opens. If it does not, the training target is the blocker.

### 2. First-open replay reweighting

Implemented option:

- `optimization.advantage_first_open_fraction`

Meaning: reserve this fraction of each advantage minibatch for samples where a
first-open action was legally available at the traverser decision. The target is
not changed; only replay sampling frequency changes. This keeps the experiment
on the pure self-play side more than hand-written target shaping.

Initial planned run:

- `run.experiment_name=first-open-reweight-50-512x3-det-500`
- `optimization.advantage_first_open_fraction=0.5`
- `traversal.outcome_sampling_epsilon=0.05`
- `traversal.outcome_unsampled_regret=zero`
- `run.deterministic=true`
- `run.max_iterations=500`

Primary comparison is the `eps=0.05` confirmation run and the pure external
sampling run. Success requires bad-open rate and score/opened color to improve
together without a large score-diff regression.

Result:

- Run: `runs/2026-05-09_235808_first-open-reweight-50-512x3-det-500-indexed`
- W&B group: `first-open-replay-v1`
- Commit: `8217cec` (`Speed up first-open memory sampling`)

The scan-based first implementation was stopped after 12 iterations because
first-open sampling scanned the full replay memory and pushed iteration time
above 60 seconds. The indexed-memory version kept first-open sampling near
0.25 seconds per player at 4M advantage samples and completed 500 iterations.

Final `safe_heuristic_strict` comparison:

| Run | Iter | Score diff | Win rate | Bad open | Score/opened |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline `confirm-eps-005-zero-512x3-det-500` | 500 | -61.24 | 0.06 | 0.903 | -8.30 |
| pure external `pure-external-512x3-det-500` | 500 | -50.38 | 0.06 | 0.925 | -9.79 |
| first-open reweight 50% indexed | 500 | -52.20 | 0.07 | 0.919 | -9.39 |

Conclusion: first-open replay reweighting alone did not solve selectivity. It
improved score diff versus the baseline final checkpoint, but it did not reduce
bad-open rate and made score/opened color worse. The run briefly looked better
around 120-200 iterations, then regressed by 300-500 iterations. This suggests
the replay emphasis is not enough if the underlying target does not separate
good first opens from bad first opens.

### 3. Short open-selectivity ablation

Run a 200-300 iteration ablation only after the target audit identifies a
specific change. Candidate changes include:

- first-open target shaping,
- modified unsampled-open penalty,
- outcome sampling focused on open-relevant branches,
- or strategy/eval selection that separates current and average policy at the
  open decision.

Primary metrics:

- `eval/safe_heuristic_strict/avg_score_diff0`
- `eval/safe_heuristic_strict/win_rate0`
- `eval/safe_heuristic_strict/bad_open_rate`
- `eval/safe_heuristic_strict/score_per_opened_color`

Do not promote to 500+ iterations unless bad-open rate and score/opened color
both improve without degrading score diff.

## Operational notes

- Keep one GPU training run active at a time.
- Use `tmux` plus `.compute.lock` for training.
- Mirror real experiments to W&B.
- Use one W&B group per hypothesis family.
- Do not start a long 2000-iteration run until a short diagnostic run shows
  stable selectivity improvement.
