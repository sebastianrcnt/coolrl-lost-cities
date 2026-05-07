# Opponent Policy: Network vs. Self-Play League

**Last verified:** 2026-05-07, commit `ad0be89`

Source: `docs/archive/deep-cfr-opponent-policy-network-divergence-2026-05-07.md`

## Question

Why does `traversal.opponent_policy: network` lead to policy collapse, and what
makes `self_play_league` (the default) stable?

Short answer: using the **currently training network** as its own traversal
opponent violates the stationarity assumption that Deep CFR's convergence proof
rests on. The opponent policy must be fixed (or drawn from a fixed distribution)
within a training iteration; feeding a moving target to the advantage estimator
produces non-stationary regret signals that compound into divergence. A snapshot
pool supplies that fixed diversity.

## Code reference

The opponent policy mode is selected in
`src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx` during opponent
node evaluation. The trainer wires the policy source in
`src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py`, which reads
`traversal.opponent_policy` from config
(`src/coolrl_lost_cities/games/classic/deep_cfr/config.py`).

The snapshot pool that backs `self_play_league` is managed through
`src/coolrl_lost_cities/games/classic/deep_cfr/checkpoints.py` and the
`self_play.max_snapshots` / `self_play.current_weight` /
`self_play.recent_weight` / `self_play.older_weight` config knobs.

## Divergence mechanism

Deep CFR's convergence guarantee is that the **average strategy** (maintained
by the strategy network) converges toward a Nash equilibrium as regret sums
accumulate over many iterations. This holds when the opponent at each traversal
node plays a policy that is either (a) fixed or (b) drawn i.i.d. from a
stationary distribution — the classic external-sampling assumption.

`opponent_policy: network` breaks this in four compounding ways:

1. **Moving target.** Every iteration updates the network weights, so the
   opponent's policy distribution shifts between iterations. Advantage samples
   stored in the replay buffer were measured under *different* opponent policies
   and cannot be treated as samples from the same distribution. The advantage
   network learns a target that keeps moving underneath it.

2. **Echo chamber.** The traverser and its opponent share the same network, so
   whatever weaknesses the traverser has are invisible to the opponent. States
   that would expose those weaknesses (e.g., a patient Safe Heuristic-style
   opponent that never over-opens) are never generated during traversal.
   Regret signals for responding to such opponents never appear.

3. **No-regret guarantee breaks.** External-sampling MCCFR's unbiased regret
   estimate requires the opponent to sample from a fixed strategy. When the
   opponent is the network-in-training, the estimator is biased in a
   time-varying way. The no-regret property that drives average-strategy
   convergence no longer holds.

4. **Strategy mode collapse.** Self-play between identical agents tends to
   converge to a deterministic-like Nash approximation even when the true Nash
   is mixed. In an imperfect-information game like Lost Cities, that collapsed
   strategy is exploitable by any opponent outside the narrow equilibrium.

## Observed behavior

Two controlled experiments (512x3 and 1024x4 hidden size / layers) both reached
a performance peak early and then diverged:

- **512x3**: peak at iteration 15 (~85% win rate vs. Random), then rapid
  collapse by iteration 30 to below-random performance, stable there through
  iteration 363.
- **1024x4**: larger capacity delayed collapse — plateau held from roughly
  iteration 30 to 95, with a best win rate of 13% against Safe Heuristic at
  iteration 85 — but divergence was ultimately the same.

A directly comparable run with `self_play_league` (512x3 architecture, otherwise
identical hyperparameters) reached a similar early peak, then *continued
improving* through iteration 350 with a 72% win rate vs. Random — a 38
percentage-point gap against the collapsed network run at the same iteration.

The key insight from the comparison: the early peak is similar regardless of
opponent policy, because the initial regret signal is useful for both. The
divergence is entirely post-peak, driven by the stationarity violation.

## Why self_play_league is stable

With `self_play.max_snapshots > 0`, the opponent at each traversal is sampled
from a pool of past checkpoints. Each snapshot is a *fixed* policy at the moment
it was saved. The traversal therefore draws its opponent from a stationary
distribution (the snapshot pool), satisfying the external-sampling assumption.
Diversity across snapshots ensures the traverser encounters a range of opponent
styles, preventing echo-chamber collapse.

The weighted bucket scheme (`current_weight`, `recent_weight`, `older_weight`)
controls how much the pool emphasizes recent vs. historical policies, letting
practitioners tune recency without sacrificing the stationarity guarantee.

## Practical implication

- **Do not use `opponent_policy: network` for extended training runs.** It can
  look promising in the first 10–20 iterations, which makes it easy to
  misinterpret short pilots as success.
- If network-opponent runs are conducted (e.g., to examine early dynamics),
  enable short `save_iteration_interval` and retain checkpoints from the plateau
  phase — divergence is irreversible once started, and final checkpoints are
  useless.
- `self_play_league` with `max_snapshots ≥ 10` is the stable default.
- `opponent_policy: average_strategy` (using the running average-strategy
  network as the opponent) is a theoretically interesting alternative — it more
  closely mirrors the CFR proof — but has not been run at scale in this repo
  as of `ad0be89`.

## References

- Brown, Lerer, Gross, Sandholm. *Deep Counterfactual Regret Minimization.*
  ICML 2019. (Section 4, convergence requirements for the strategy network.)
- Lanctot et al. *Monte Carlo Sampling for Regret Minimization in Extensive
  Games.* NeurIPS 2009. (External-sampling stationarity assumption.)
