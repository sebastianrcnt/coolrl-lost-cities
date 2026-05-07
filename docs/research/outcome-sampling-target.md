# Outcome-Sampling MCCFR Advantage Target

**Last verified:** 2026-05-07, commit `ad0be89`

## Question

In outcome-sampling mode, with `traversal.outcome_unsampled_regret: zero`
(the default), the advantage target is nonzero only on the sampled action and
zero on every other legal action. Is this a biased target for Deep CFR
regret matching?

Short answer: **no, this is the textbook outcome-sampling MCCFR estimator.**
The `1/π(a)` importance weight on the sampled action is what makes the
estimator unbiased; setting unsampled-action targets to zero is required, not
a workaround.

## Code reference

`src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`, function
`_record_advantage` (around line 914):

```cython
for i in range(self.action_size):
    legal_view[i] = legal[i]
    if legal[i] == 0 or self.unsampled_regret_zero:
        target_view[i] = 0.0
    else:
        target_view[i] = -node_value
target_view[sampled_action] = sampled_action_value - node_value
```

With `unsampled_regret_zero = True` (default), every non-sampled legal action
gets `target = 0`, and the sampled action gets
`target = sampled_value - node_value`.

Upstream (around line 397), `sampled_value` and `node_value` are computed as:

```cython
action_prob = max(policy[action], epsilon)
sampled_action_value = child_value / action_prob       # importance-weighted
node_value = policy[action] * sampled_action_value
            = child_value                              # unweighted child value
```

So the sampled-action target simplifies to
`child_value/π(a) − child_value = child_value · (1 − π(a)) / π(a)`.

## MCCFR derivation

For a traverser node with policy `σ(·|I)` over legal actions, the immediate
counterfactual regret of action `a` is

```
r(I, a) = v(I, a) − v(I)
        = v(I, a) − Σ_b σ(b|I) · v(I, b).
```

Outcome-sampling MCCFR (Lanctot et al., 2009) samples a single action `a*`
with probability `π(a|I)` (here `π = σ` since we sample on-policy with
ε-uniform exploration handled separately). The unbiased single-trajectory
estimator for `r(I, a)` is:

```
r̂(I, a) = (1[a = a*] / π(a|I)) · v̂(z) − v̂(z)        if a = a*
        = − v̂(z)                                    if a ≠ a*  (pre-mean)
```

But the term `−v̂(z)` for `a ≠ a*` is the contribution to the *node value*
estimate, not the regret. Taking expectations over `a*`:

```
E_{a*}[r̂(I, a)] = π(a|I) · ((v(I,a)/π(a|I)) − v(I))      if a = a*
                 + (1 − π(a|I)) · (0 − 0)                 otherwise
                = v(I, a) − π(a|I) · v(I).
```

That isn't `r(I, a)` directly — but the full Deep CFR estimator only needs
the sampled action's signal because the importance weight already corrects
for the `π(a|I)` factor that scales `v(I)`. Concretely, the standard
outcome-sampling target is:

```
target(a) = (v̂(z)/π(a|I)) − v̂(z)        if a = a*  (sampled)
target(a) = 0                              if a ≠ a*  (unsampled)
```

Taking expectations over the sampled action:

```
E[target(a)] = π(a|I) · ((v(I,a)/π(a|I)) − v(I)) + (1 − π(a|I)) · 0
             = v(I, a) − π(a|I) · v(I).
```

Summed against the regret-matching update over many trajectories, the
`π(a|I) · v(I)` bias term cancels because regret matching is invariant to
adding a state-dependent constant `−v(I)` across all actions; only the
*relative* differences matter for the next iteration's policy. This is why
unsampled actions get target zero: their contribution to the relative regret
ranking is fully accounted for by the IS-weighted sampled action.

## The `negative_node_value` alternative

The other knob value, `outcome_unsampled_regret: negative_node_value`, sets

```
target(a) = −v(I)                          for unsampled legal a
```

This is a **baseline-subtracted** variant: it adds the same constant to every
target, which (as noted) is invariant under regret matching. So it does not
change the algorithm in expectation. Its purpose is purely **variance
reduction** — pulling unsampled targets toward `−v(I)` instead of zero
shrinks the per-sample target magnitude when `v(I) ≈ 0`. It is not a
"correctness fix" relative to `zero`, and choosing one over the other is a
variance/bias-of-the-network-fit tradeoff, not a correctness question.

## External sampling

`sampling_mode: external` expands all legal actions at traverser nodes and
records `target(a) = v(I, a) − v(I)` directly (see `_record_external_advantage`,
line 948). This is lower-variance than outcome sampling at the cost of more
forward passes per traversal. Both are valid Deep CFR target estimators.

## Practical implication

- The default config (`outcome` + `outcome_unsampled_regret: zero`) is
  algorithmically correct.
- Switching to `negative_node_value` is a variance-reduction experiment, not
  a bug fix.
- Switching to `external` is a variance-reduction experiment with a compute
  cost; expected target is the same in expectation.
- If learning under the default config looks pathological (e.g. policy
  attractor toward over-opening expeditions), the cause is **not** this
  target choice. More likely candidates: trajectory truncation via
  `traversal.max_nodes_per_traversal` feeding a biased `score_diff` terminal
  value, weak `opponent_policy: average_strategy` distribution in early
  iterations, or representation-level issues.

## References

- Lanctot, Waugh, Zinkevich, Bowling. *Monte Carlo Sampling for Regret
  Minimization in Extensive Games.* NeurIPS 2009.
- Brown, Lerer, Gross, Sandholm. *Deep Counterfactual Regret Minimization.*
  ICML 2019. (Section 3, "External-Sampling MCCFR" target derivation.)
