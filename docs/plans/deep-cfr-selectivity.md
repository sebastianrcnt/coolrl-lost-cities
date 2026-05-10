# Deep CFR Selectivity Investigation

Last updated: 2026-05-10

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
game-quality metrics against `heuristic_cautious` did not improve enough to
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

### regret_matching_epsilon = 1e-4 was the zero-pit countermeasure (prior repo)

이전 레포(`../coolrl`)에서 zero-pit 발생 당시 시도한 epsilon 튜닝 기록
(commit `5a98855`, `da0d4cd`, 2026-05-06). 이건 `outcome_sampling_epsilon`이
아니라 `regret_matching_epsilon`이라는 별도 파라미터.

- **eps1e3** (`regret_matching_epsilon=1.0e-3`): zero-pit timeout은 줄였지만
  safe 계열 avg_diff 개선 X. selectivity floor를 과하게 만들 가능성 의심.
- **eps1e4** (`regret_matching_epsilon=1.0e-4`): floor 더 낮춰서 219 iter
  수동 중단. random 승률 0.86 / score_diff +32, safe 승률 0.03~0.05 /
  score_diff -57~-74. zero-pit timeout은 거의 안 나오지만 selectivity
  실패는 그대로.

`1.0e-4`가 그때 best로 채택되어 현재 `default.yaml`의 기본값으로 남아 있음
(`regret_matching_epsilon: 0.0001`). R0, R1 양쪽 모두 이 설정 사용 중.

**이 lever는 이미 당겨져 있다.** R1에서 zero-pit이 다시 나타난 것은
floor가 풀려서가 아니라, opponent=discard_only가 "do nothing" 균형을
*수학적으로* 안전하게 만든 별개 원인. epsilon 재조정으로 풀리는 구조가 아님.

### past-self 풀만으로는 trap 못 깸 (prior repo)

이전 레포(`../coolrl`) `full_depth` 실험 (commit `33e0368`, 2026-05-06).

설정 (anchor 없음, 순수 past-self pool):

```yaml
self_play_league:
  current_weight: 0.5
  recent_weight: 0.3
  older_weight: 0.2
  max_snapshots: 20
  recent_window: 5
```

가설: opponent로 본인 최신만 쓰는 self-mirror 평형이 selectivity emergence를
막는다면, 과거 자아 snapshot을 섞으면 평형이 깨진다.

결과 (iter 322 종료):

- safe 상대 opened_colors: **4.94~4.96** (거의 전색 opening)
- 5-color opening 빈도: 91% → 93% (iter 110→180, 감소 없음)
- safe avg_diff: eps1e4 baseline 대비 +25점 회복 (recovery skill은 emerge)
- terminal rate 100%, node cutoff 0% (truncation 정상)

판정 (commit 메시지 그대로): "recovery는 self-play로 emerge했지만 selectivity는
현재 league 평형 안에서 emerge하지 않는다."

함의 — past-self를 섞어도 본인 과거 자아는 같은 trap policy의 시간축 평행이동일
뿐이라 self-mirror 평형이 사실상 그대로 유지된다. selectivity는 self-play 가족
내부 다양성으로 풀리지 않는다.

### self_play_league에 heuristic anchor 0.15 주입은 trap을 깨지 못함 (prior repo)

이전 레포(`../coolrl`) `anchor_safe015` 실험 (commit `279d726`, `a77464b`,
2026-05-06).

설정:
- self_play_league에 deterministic `safe_heuristic` anchor를 0.15 weight로
  주입. 나머지 0.85는 current/recent/older self snapshot.
- traversal에서 15%의 opponent role을 휴리스틱이 담당 → self-mirror 평형
  부분 절단.

가설: "self-mirror over-opening 평형을 안chor가 깰 수 있는가."

결과 (1219 iter / 4h 풀 런):

| metric | 값 |
| --- | ---: |
| anchor traversal rate | 15.7% (mechanism 정상) |
| safe 상대 avg_diff | -57.11 |
| opened_colors | 4.83 |
| 5-color opening 빈도 | ~86% |
| random avg_diff | +48.31 (후퇴 없음) |

판정 (commit 메시지 그대로): "anchor pressure 0.15가 over-opening 평형을
깨지 못한 것으로 판정."

함의 — opponent pool에 외부 정책 일부 섞기는 0.15 정도로는 trap을 못 깬다.
더 큰 비율(0.5+)은 사실상 self-play 가설 폐기와 동일하므로 별개 카테고리로
취급해야 한다. discard_only를 anchor로 섞는 변형은 더 약한 신호이므로 같은
weight로는 더 나쁠 가능성이 큼.

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

Counterfactual summary against `heuristic_cautious`:

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

Final `heuristic_cautious` comparison:

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

### 3. First-open target prior (A1) — running 2026-05-10

Hypothesis: target audit shows the regenerated traversal target does not
separate good first opens from bad first opens. With
`outcome_unsampled_regret=zero`, every unsampled first-open candidate is
labeled with target `0`, regardless of whether it is heuristically good
or bad. A small signed prior on unsampled first-open play actions should
break the symmetry without overriding the high-variance sampled-action
target.

Implementation:

- Config field: `traversal.outcome_unsampled_first_open_prior_alpha` (float,
  default `0.0`).
- Plumbed through `InterleavedTraversalConfig` and
  `run_interleaved_traversal_batch`. Cython traversal path unchanged
  (default scheduler is interleaved; pyx path is a follow-up if needed).
- Effect site: `_after_child` in `interleaved_traversal.py`. When
  `is_first_open` and `alpha != 0`, unsampled legal play actions whose color
  has an empty expedition are overwritten in `target` with `+alpha` if the
  visible `recoverable_score` for that color is `>= 0`, else `-alpha`. The
  sampled action's cell stays as `sampled_action_value - node_value`.
- `recoverable_score` mirrors `evaluate._visible_recoverable_summary` for
  the empty-expedition case (no future-card lookahead, hand-only signal).
- Tests: `test_first_open_prior_overrides_unsampled_play_targets_with_signed_alpha`,
  `test_first_open_prior_zero_alpha_is_noop`.

Note on "pure self-play": this prior introduces a hand-written heuristic
into the advantage target. The opponent and rollout policy remain
self-play (no external bot). This is treated as a diagnostic experiment;
if effective, the permanent solution is a counterfactual-based prior
(A2) that recovers full pure self-play.

Initial planned run:

- `run.experiment_name=first-open-prior-alpha5-512x3-det-200`
- `traversal.outcome_unsampled_first_open_prior_alpha=5.0`
- `traversal.outcome_unsampled_regret=zero`
- `traversal.outcome_sampling_epsilon=0.05`
- `run.deterministic=true`
- `run.max_iterations=200`
- W&B group: `first-open-prior-v1`

Primary comparisons:

- `confirm-eps-005-zero-512x3-det-500` (baseline; iter 200 reference)
- `first-open-reweight-50-512x3-det-500-indexed` (replay-side intervention)

Success criterion: `bad_open_rate` and `score_per_opened_color` improve
*together* at iter 200 without large regression in `avg_score_diff0`. If
positive, follow up with a 500-iter run to test stability. If null/regress,
try alpha sweep (e.g. 2.0, 10.0) before abandoning the direction.

Result (2026-05-10):

- Run: `runs/2026-05-10_072718_first-open-prior-alpha5-512x3-det-200`
- W&B: synced online to group `first-open-prior-v1` (run `teuh915r`)
- Commit: see HEAD at run start

`heuristic_cautious` at iter 200:

| Run | Score diff | Win rate | Bad open | Score/opened |
| --- | ---: | ---: | ---: | ---: |
| baseline `confirm-eps-005-zero` (iter 200) | -40.01 | 0.12 | 0.893 | -6.25 |
| **A1 prior α=5.0 (iter 200)** | **-67.54** | **0.02** | **0.796** | **-8.85** |

Other opponents at iter 200: random +32.51 / 0.86, heuristic_noisy -71.52 / 0.07,
heuristic_balanced -81.81 / 0.02, heuristic_aggressive -81.24 / 0.05.

Conclusion: **mixed result, net regression.** The prior did shift behavior
in the intended direction on one axis — `bad_open_rate` dropped from 0.893
to 0.796 (~10% absolute reduction). This is the only metric where the
hypothesis "the prior breaks ranking symmetry" looks supported.

But the overall game-quality metrics regressed: score diff worsened
(-40 → -67), win rate collapsed (0.12 → 0.02), and `score_per_opened_color`
got worse (-6.25 → -8.85). The model fails the success criterion (which
required `bad_open_rate` AND `score_per_opened_color` to improve together).

Two interpretations:

1. **α = 5.0 too strong.** The prior is overpowering the sampled-action
   target rather than acting as a weak symmetry-breaker. The good-open prior
   pushes the model to open *more often*, but those forced opens are bad
   *given the actual game state*, just labeled good by the visible-only
   heuristic. The counterfactual audit already flagged this: heuristic
   `open_good` candidates often lose to best non-open in real continuation.
2. **Heuristic itself misaligned.** Even the right α won't help if the
   sign assigned to candidates is wrong relative to true game value.

Next steps (in order):

- (A1.b) α sweep at 200 iter: α ∈ {1.0, 2.0} to test "weaker prior" hypothesis.
  If α=2.0 still regresses score diff while reducing bad_open, the prior shape
  itself is wrong, not just strength.
- (A2) Counterfactual prior: replace heuristic sign with sign of
  `value(force open) - value(best non-open)` from a small in-traversal
  rollout. Cleaner signal, recovers pure self-play, but more expensive.
- If both fail to improve score diff, the issue is upstream of unsampled
  regret labeling (likely traversal sample distribution or sampled-action
  target variance), and the next experiment family should target those.

### 4. D1 diagnostic: counterfactual with strong post-policy (2026-05-10)

Question: are forced-open continuation values low because the openings
themselves are bad, or because the self-play rollout policy poisons the
post-action play (selection bias)?

Method: re-ran `analyze_first_open_counterfactual.py` on the
`confirm-eps-005-zero-512x3-det-500` baseline checkpoints (iter 200, iter
500), but with `--post-policy heuristic_cautious`. The opponent and
state-collection policy stayed the same; only the policy_player's actions
*after* the forced first action used the strong fixed bot.

Output: `runs/tmp/first_open_counterfactual_d1_strong_post_policy_200_vs_500.jsonl`

`delta_open = value(force open) - value(best non-open)` comparison:

| iter | bucket | n | self-play post | strong post | shift |
| ---: | --- | ---: | ---: | ---: | ---: |
| 200 | open_good | 40 | -26.57 | **-3.35** | +23.22 |
| 200 | open_bad | 460 | -23.15 | **-5.52** | +17.63 |
| 500 | open_good | 30 | -23.43 | **-10.90** | +12.53 |
| 500 | open_bad | 470 | -11.18 | **-8.99** | +2.19 |

`delta_positive_rate`:

| iter | bucket | self-play post | strong post |
| ---: | --- | ---: | ---: |
| 200 | open_good | 0.050 | 0.225 |
| 200 | open_bad | 0.130 | 0.317 |
| 500 | open_good | 0.167 | 0.300 |
| 500 | open_bad | 0.226 | 0.230 |

Verdict — two findings, both important:

**1. Selection bias is real and significant.** Strong post-policy
improves forced-open continuation values by 12–23 points on average. The
self-play policy is meaningfully poisoning rollouts: forced opens look
much less bad once a competent player handles the followup.

**2. Heuristic labels do not separate cleanly even under strong post-policy.**
At iter 200, `open_good` delta_mean is only ~2 points better than
`open_bad` (-3.35 vs -5.52). At iter 500, ranking is essentially flat
(`open_good` -10.9 vs `open_bad` -8.99 — slightly *worse*). Median deltas
agree. Sample size for `open_good` is small (30–40) so noise contributes,
but there is no clean signal that the heuristic `recoverable_score`
classifier matches actual continuation value.

Implications:

- A1 prior was destined to fail — the heuristic sign is at best weakly
  aligned with continuation value, even with optimistic post-policy.
- A2 with self-play rollouts would inherit selection bias and likely
  reproduce the same misranking. A2 with a strong post-policy would
  give clean signs but breaks pure self-play.
- The deeper bottleneck is post-open play quality. Until the trained
  policy plays competently *after* opening, training signals about
  *whether to open* will be biased toward "don't open."

Candidate next directions (decision pending):

- (E1) Train with `cutoff_rollout_policy=heuristic_balanced` instead of `random`.
  Already a config option; gives leaf nodes stronger value estimates
  during traversal. Trades some pure-self-play purity for a stronger
  bootstrap signal. Cheap to test.
- (E2) Investigate post-open behavior directly: forced-open + observe
  next-2-3 turns. Diagnoses *why* the model can't follow up (e.g. always
  discards followup cards, switches color, etc.).
- (E3) Curriculum / staged training that exposes the network to good
  post-open trajectories before forcing it to make first-open decisions.
- A2 deferred unless we adopt a strong post-policy in rollouts.

### 5. E2 diagnostic: post-forced-open behavior (2026-05-10)

Question: D1 showed selection bias is real — *what is the model actually
doing after a forced first-open* that poisons rollouts?

Method: forced each first-open candidate at sampled states, then observed
the policy_player's next 3 decisions (window). Used baseline checkpoints
(`confirm-eps-005-zero` iter 200, iter 500). Categorized each followup
decision relative to the forced-open color.

Output: `runs/tmp/first_open_followup_baseline_200_vs_500.jsonl`
Script: `scripts/analyze_first_open_followup.py`

Per-window mean counts (3 policy_player decisions = ~1.5 game turns;
each game turn includes one play/discard plus one draw):

| iter | bucket | n | held@force | plays(same) | discards(same) | open_other | other_discard | held@end |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | good | 33 | 4.82 | 0.15 | 0.15 | **0.49** | 0.21 | 2.70 |
| 200 | bad  | 367 | 1.97 | 0.03 | 0.12 | 0.10 | 0.75 | 1.08 |
| 500 | good | 22 | 4.45 | 0.14 | **0.46** | 0.14 | 0.27 | 1.23 |
| 500 | bad  | 378 | 2.56 | 0.01 | 0.23 | 0.05 | 0.71 | 1.29 |

Findings:

**1. Model under-plays followup cards even when it holds many.** At iter
200, good-open candidates start with ~4.8 same-color cards in hand. In the
next 3 decisions, the model plays only 0.15 of them on average (3% of
its window). 2.7 cards remain in hand at terminal — nearly 3 cards of
the just-opened color never reach the expedition.

**2. Model opens additional new colors after a forced open.** At iter 200
good bucket, `open_other` = 0.49 in a 3-decision window — the model is
~3× more likely to open *another* new color than to follow up the one it
was forced into. This compounds the recoverable-score drag.

**3. By iter 500 the model actively dumps the forced color.** Good bucket
at iter 500: `same-color discards` 0.46 vs `same-color plays` 0.14. The
model discards followup cards more than 3× as often as it plays them,
despite holding 4.45 of them at force time. This is the strongest single
piece of evidence so far that the post-open value head is poisoned.

**4. Bad-open bucket shows even sharper avoidance.** Hand has ~2
same-color cards, plays effectively zero. Almost all play-phase decisions
are "discard other" (0.71–0.75). The model treats forced opens as bad
news to liquidate rather than commit to.

Interpretation: the trained advantage network *learned* that opens of
these colors are net-negative, so once forced into one, it hedges by
opening other colors and dumping followup cards. That hedging is rational
under the policy's own value estimates but is the exact mechanism that
keeps forced-open continuation values low and keeps the training target
labeling opens as bad. Closed loop.

This combined with D1 means:

- A1/A2 priors operate on first-open *labels*. They cannot fix a model
  that, even after committing to an open, refuses to follow up.
- Any fix needs to either (a) produce stronger leaf values during training
  so the value head learns post-open play matters, or (b) explicitly
  curriculum or prior-shape the *post-open* decision (not the open
  decision).

Updated next-step priorities:

- **(E1) `cutoff_rollout_policy=heuristic_balanced` training ablation.**
  Strongest single lever: gives traversal leaves stronger value estimates
  during training, which should ripple back to "open + follow up" signal.
  Pure-self-play purity dented, but only at cutoff leaves.
- (E1.b) Sanity check: at iter 200 baseline, the same-color discard rate
  is already 0.12 in good bucket — early. So this is not a late-training
  collapse; it's baked in from early iterations. Curriculum-style fix
  would have to start very early.
- A2 effectively dead unless paired with strong post-policy in rollouts.

### 6. Input feature audit and cleanup (2026-05-10)

After D1 + E2 confirmed the bottleneck is *training signal* and not *input
poverty*, audited every feature the model receives. Goal: align the input
representation with a strict pure-self-play definition by removing any
feature that embeds a hand-coded judgment or strategy assumption.

Three tiers found:

- **Tier 1**: pure game state and single-step game rules (hand cards,
  expedition state, discard piles, scores, public histogram, legal-action
  mask, `playable_*`/`dead_*` rule checks, `unknown_remaining_count`,
  etc.). No judgment. Kept.
- **Tier 2**: projection-based features that compute "if I commit and
  play all currently-playable cards, what is the score?" — mechanical
  but assumption-laden. Includes `recoverable_score_no_bonus`,
  `recoverable_margin_no_bonus`, `min_needed_to_break_even`,
  `cards_needed_for_bonus`, `has_bonus_path`. Removed.
- **Tier 3**: explicit hand-coded judgments — `is_bad_open_candidate`,
  `open_risk_score`, `is_safe_continuation`. Same heuristic family used
  to label `bad_open` in evaluation. Removed.

Notable additional finding: the `no_bonus` projection family is
asymmetric. It includes wager multipliers but excludes the +20 bonus,
which systematically *under*-estimates the upside of commitable
expeditions by roughly 20 × P(complete bonus). This bias points in the
same direction as the phase 1 trap ("opens look like loss"), so removing
the projection family is consistent with diagnosing-not-baking-in the
trap.

Implementation:

- `encoding.pyx`: `DERIVED_PLAYABILITY_PER_COLOR` 19 → 15;
  `SLOT_AWARE_PLAYABILITY_PER_SLOT` 12 → 6 (across two cleanup passes).
- Test shape assertions updated.
- Cython module rebuilt.

Input dimension change: 365 → 341 (Tier 3 removal) → **297** (Tier 2
removal). Net 18.6% reduction.

Result is not yet measured. The hypothesis is that with the heuristic
crutches removed, training behavior is a cleaner measurement of what
Deep CFR can do in this game from raw input. The model may stay in the
phase 1 trap longer or fail more visibly, both of which are useful
information.

### 7. ColorSharedNetwork chunked-layout bug (2026-05-10)

While re-examining the archived `2026-05-07_092137_color_shared_attention_1000iter`
run (killed at iter 41), discovered the `ColorSharedNetwork` implementation
in `networks.py` was not actually color-aware. The forward pass split the
input vector into `input_dim // n_colors` contiguous slices and ran them
through a shared encoder. The slice boundaries do not align with the actual
encoding layout — adjacent slices contain phase flags, hand slots,
expedition state, scores, etc. mixed together. The "shared color encoder"
was therefore sharing weights across semantically unrelated chunks, not
across per-color blocks.

This means the prior conclusion that "color_shared / attention archive run
was inconclusive" was charitable. The architecture being measured was a
chunked-input network mislabeled `color_shared`, not a real per-color
shared architecture. We have *no* signal on whether a properly per-color
architecture would help.

Fix landed in this same session:

- Added `compute_lost_cities_color_layout(input_dim)` in `networks.py`. For
  the standard Lost Cities schema (n_colors=5, hand_size=8, n_ranks=9), it
  recognises the four valid `input_dim` values (171, 219, 249, 297 across
  derived/slot-aware flag combinations) and returns per-color and common
  index lists derived from the actual encoding layout.
- Per-color block (39 dims with `derived_playability` on): both players'
  expedition state for that color, discard top metadata, public-histogram
  row, pending-discard one-hot bit, legal-action draw-pile bit, and the
  `derived_playability` per-color block. Slot-aware features stay in
  common because they are slot-major, not color-major.
- `ColorSharedNetwork.forward` now indexes per-color blocks via the layout
  when `input_dim` matches a known schema. For other input dims (unit
  tests, non-Lost Cities use), it falls back to the old chunked slicing
  with a `UserWarning`, preserving backward compatibility for tests but
  making the legacy behaviour visible.
- New unit tests cover both branches and the layout helper.

This is purely an implementation correctness fix; no fair test of the
architecture has been run yet. Fair test deferred — the diagnosis from
sections 4–5 (selection bias, post-open behaviour) suggests that even a
correct color-aware encoder would not break the closed loop on its own.

### 8. Interleaved scheduler honours all_negative_fallback (2026-05-10)

The interleaved traversal scheduler's `_regret_matching` was hard-coded to
spread fallback policy uniformly across legal actions, regardless of the
configured `regret_matching.all_negative_fallback`. The default config has
shipped with `all_negative_fallback: argmax_tiebreak` since
`618d5f8 Promote avg-strategy 1000iter to default.yaml` based on prior
20-iteration audit + 1000-iteration empirical evidence (see
`docs/archive/deep-cfr-regret-fallback-audit-2026-05-07.md`), but the
default scheduler was switched to interleaved in `09bbe7c Make interleaved
traversal the default`, after which the configured fallback mode silently
no-op'd in interleaved code paths.

Fix:

- `_regret_matching(advantages, legal_mask, epsilon, fallback_mode="uniform")`
  in `interleaved_traversal.py` now honours `argmax_tiebreak` by
  concentrating policy mass on the lowest-index tied action (deterministic
  tiebreak; the Cython recursive traverser randomises ties using its
  per-traverser RNG, which the batched policy does not have).
- `BatchedPolicy` accepts `fallback_mode` and threads it through.
- `InterleavedTraversalConfig` carries `all_negative_fallback`.
- `run_interleaved_traversal_batch`, `trainer.py`, `workers.py`, and the
  `analyze_first_open_targets.py` callers pass the field through.

Also bumped `traversal.outcome_sampling_epsilon` in `default.yaml` from
0.2 to 0.05. The 200-iteration sweep (section 1) showed 0.05 produced the
best short-run heuristic_cautious score diff (-40.01 vs -57.87 for
0.20). All recent experimental runs already used 0.05; the default now
matches actual experimental practice.

These changes do not target the diagnosed selection-bias bottleneck. They
align config intent with actual scheduler behaviour and make the default
config reproduce known-best knob settings out of the box.

### 9. Naming, plot curation, and tiered eval cadence (2026-05-10)

Hygiene changes — none target the diagnosed selection-bias bottleneck,
but they make the codebase honestly reflect the pure-self-play stance
and reduce dashboard noise.

Bot family rename (drop the unhelpful `safe_` prefix; suffixes now
describe behaviour):

| Old | New |
| --- | --- |
| `safe_heuristic_loose` | `heuristic_aggressive` |
| `safe_heuristic` | `heuristic_balanced` |
| `safe_heuristic_strict` | `heuristic_cautious` |
| `noisy_safe` | `heuristic_noisy` |
| `passive_discard` | `discard_only` |

Class renames in `bots/`: `SafeHeuristicBot` → `HeuristicBot`,
`SafeHeuristicParams` → `HeuristicParams`, `PassiveDiscardBot` →
`DiscardOnlyBot`, plus the loose/strict parameter constants. Backwards
compatibility was dropped intentionally — no aliases. Active configs,
docs, scripts, tests updated; archive files (read-only by policy)
left intact and may still reference old names.

Analyze plot curation (`deep_cfr/analyze.py`):

- New `analysis_00_core.png` dashboard with 10 heuristic-free metrics
  (loss/{advantage, strategy}; vs `heuristic_cautious`:
  `avg_score_diff0`, `win_rate0`, `avg_opened_colors`,
  `positive_expedition_rate`, `bonus_expedition_rate`,
  `score_per_opened_color`, `policy_entropy`; vs `random`: `win_rate0`).
- Removed `analysis_05_open_quality.png` (bad/weak/good open rates,
  recoverable score) and `analysis_07_calibration.png` (calibration gap,
  recoverable mean) — both relied on the heuristic `recoverable_score`
  classifier we already dropped from inputs.
- Removed `SELECTIVITY_PLOTS` and `plot_selectivity` (heuristic-laden).
- `SUMMARY_EVAL_METRICS` no longer includes `bad_open_rate` or
  `calibration_gap`.

`PlotSpec` gained an optional `opponents` allowlist so the new core
section can pin a specific opponent per panel without restructuring the
existing `plot_section` plumbing.

Tiered evaluation cadence (`EvaluationConfig`):

- Added `extended_opponents: tuple[str, ...]` and `extended_eval_every:
  int = 0`.
- Method `opponents_for_iteration(iteration)` returns the core list at
  every `eval_every`, and appends `extended_opponents` (de-duplicated)
  when `iteration` is also a multiple of `extended_eval_every`.
- `default.yaml` now uses 3 core opponents
  (`random`, `discard_only`, `heuristic_cautious`) every 5 iterations
  and 3 extended opponents (`heuristic_balanced`, `heuristic_aggressive`,
  `heuristic_noisy`) every 50 iterations.
- `random` is the floor sanity. `discard_only` is the zero-pit
  detector / absolute-score reference (its score is always 0, so
  `eval/discard_only/avg_score_diff0` directly equals our model's
  raw average score). `heuristic_cautious` is the ceiling and the
  archive-comparable benchmark used in sections 1–6.

Net effect on ongoing eval cost: ~50% reduction (3 opponents × every
5 iter, plus 6 opponents × every 50 iter, vs the prior 6 opponents
× every 5).

### 10. Short open-selectivity ablation

Run a 200-300 iteration ablation only after the target audit identifies a
specific change. Candidate changes include:

- first-open target shaping,
- modified unsampled-open penalty,
- outcome sampling focused on open-relevant branches,
- or strategy/eval selection that separates current and average policy at the
  open decision.

Primary metrics:

- `eval/heuristic_cautious/avg_score_diff0`
- `eval/heuristic_cautious/win_rate0`
- `eval/heuristic_cautious/bad_open_rate`
- `eval/heuristic_cautious/score_per_opened_color`

Do not promote to 500+ iterations unless bad-open rate and score/opened color
both improve without degrading score diff.

## Operational notes

- Keep one GPU training run active at a time.
- Use `tmux` plus `.compute.lock` for training.
- Mirror real experiments to W&B.
- Use one W&B group per hypothesis family.
- Do not start a long 2000-iteration run until a short diagnostic run shows
  stable selectivity improvement.

## §10. R1 — discard_only opponent diagnostic (2026-05-10)

### Hypothesis

Trap의 본질이 **자기참조 회로(self-similar opponent)**인지 **모델 자체
credit assignment 실패**인지 분리하기 위해, opponent_policy를
`discard_only` (항상 카드만 버리는 fixed bot)로 고정. opponent
dynamics를 거의 0으로 만들고 self-reference를 끊은 환경에서 모델이
trap을 벗어나는지 본다.

### Run

- Run dir: `runs/2026-05-10_173912_r1-discard-only-opp-512x3-det-300`
- W&B group: `r1-discard-only-opp-v1`
- Iters: 300 (R0 끊긴 시점과 같은 길이로 비교)
- Single change vs R0: `traversal.opponent_policy: discard_only`
- All else (capacity 2M, eval games 100, eps=0.05) identical

### Result (iter 300, vs R0 clean-baseline iter 300)

| metric | R0 (over-open) | R1 (zero-pit) | Δ |
| --- | ---: | ---: | ---: |
| score_diff vs cautious | -47.99 | **-69.56** | -21.57 worse |
| opened_colors vs cautious | 4.93 | **2.54** | -2.39 |
| play_action_rate vs cautious | 0.02 | **0.00** | -0.02 |
| pos_expeditions/game cautious | 0.76 | **0.08** | -0.68 |
| neg_expeditions/game cautious | 4.06 | 2.44 | -1.62 |
| avg_game_length cautious | 1053 | **5282** | +4229 (timeout) |
| score_diff vs discard_only | -27.19 | **0.00** | +27.19 |
| opened_colors vs discard_only | 4.98 | **0.00** | -4.98 |
| bad_open_rate vs discard_only | 0.91 | **0.00** | -0.91 |
| score_diff vs random | +49.61 | **-9.30** | -58.91 |
| opened_colors vs random | 4.97 | 1.45 | -3.52 |

### Interpretation: trap shifted, did not break

Trap 본질 진단 결과 **"자기참조 회로"는 충분조건이 아니다.**

- vs `discard_only` opponent: 모델이 **정확히 0개 색을 열고 0점으로 비김**
  (zero-pit). 즉 closed-loop self-reference 깬 후에도 trap이 다른
  형태(zero-pit)로 재출현.
- vs `random`: R0에서 +49 (제압)하던 상대에게 R1은 -9 (패배). random은
  카드를 두기라도 하니까 음수 expedition으로 끝나도 양수 게임 일부 만들어서 점수
  내지만, R1 모델은 0점 (전혀 안 둠).
- vs `heuristic_cautious`: opened_colors 절반(4.93→2.54)으로 줄었지만
  play_action_rate 0% — 연 expedition을 채우지 못함. avg_game_length
  5x 증가 → max_steps timeout 다발. score_diff 22점 더 악화.
- Action 분포 (vs cautious): **discard 50% / draw_pile 50% / play 0% /
  draw_deck 0%**. 즉 "버리고 상대 버린 거 줍기"의 단일 패턴.
- Loss 트렌드: advantage loss 1500 → 3500 (꾸준히 상승), strategy loss
  0.7 → 1.4 (꾸준히 상승). 모델이 점점 더 noisy한 target을 못 fit함.

### What this proves

1. **Trap의 본질은 모델 자체 value function 실패 (credit assignment)**, not
   self-reference. self-reference는 trap 강화 요인이지만 제거해도 다른 형태로
   재출현.
2. **모델이 일관되게 못하는 일은 "expedition에 카드 두기" 행동**. R0에서
   over-open 무차별 5색 + 채우지 못함. R1에서 아예 안 열되 채우지도 않음.
   양 극단 어디든 play_action_rate ≈ 0%.
3. **Outcome sampling + 본인 정책 rollout만으로 long-horizon credit chain을
   못 닫는다.** "open이 좋은가?"의 답은 "이후 카드를 잘 두는가"에 종속이고,
   모델이 카드 두기를 못 배우는 한 open 가치는 노이즈로만 추정됨.

### Recommended next direction

이전에 hygiene 개선(R1 = 4M capacity + 500 eval)을 첫 후보로 두었으나, R1의
명확한 결과로 hygiene으로는 trap 본질 못 풀 것이 거의 확정. **R2는 trap에 직접
작용하는 변경이어야 함.**

후보 (강도 약→강):

1. **Curriculum (작은 게임 → 큰 게임)**: 3색 5랭크 mini Lost Cities로 시작 →
   credit chain 짧아짐 → expedition 사이클이 게임 8턴 안에 닫힘. 모델이
   "카드 두기" 학습 가능성 확보. 풀 게임에 weight transfer.
2. **Opening / playing 네트워크 분리**: closed loop 회로의 한 회선 물리적
   절단. opening 네트워크는 follow-up이 noise여도 자체 신호로 학습 가능.
   아키텍처 작업 큼.
3. **Cutoff rollout heuristic + external opponent (조합)**: opening 가치
   추정에 외부 합리적 정책 끼움. pure self-play 가설 명확히 폐기.

저자 추천: **R2 = curriculum**. 가장 적은 아키텍처 변경으로 credit chain
직접 단축. 작은 게임에서 follow-up 학습 성공이 풀 게임 학습의 핵심 prior가
됨.

