# Strategy Memory Recording Location: Traverser vs. Opponent Nodes

**Last verified:** 2026-05-07, commit `edad3b4`

Source: prompted by static-review feedback claiming our default
`store_strategy_on_traverser_nodes: true` is the reverse of the OpenSpiel
Deep CFR convention.

## Question

Our default config records strategy samples at the *traverser*'s own info
states during a traversal. OpenSpiel's reference Deep CFR records them at
the *opponent*'s info states. Does this difference bias the average-policy
estimate, and does the answer depend on `sampling_mode`?

Short answer: **for the current outcome-sampling default it is empirically
fine, but if we ever flip `sampling_mode: external` we must also flip these
two flags or the strategy memory will diverge from the OpenSpiel
convention.** The bias direction differs between sampling modes; the safe
rule is "use OpenSpiel's convention (opp nodes, opp info state) for
external; either is acceptable for outcome".

## Code reference

Our recording site,
`src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`,
`_record_strategy` (line 877):

```cython
cdef void _record_strategy(self, info_state, legal, policy, player,
                           traverser, iteration, depth, stats):
    if player == traverser:
        if not self.store_strategy_on_traverser_nodes:
            return
    elif not self.store_strategy_on_opponent_nodes:
        return
    ...
    self.strategy_samples.append(TrainingSample(
        info_state=info_state, target=target, legal_mask=legal_mask,
        iteration=iteration, player=player,
    ))
```

Called unconditionally at every decision node (line 323) before the
sampling-mode branch. The `info_state` is computed by `_policy(state,
player, ...)` for the *current acting player*, which is the right thing in
both conventions.

`configs/deep_cfr/default.yaml` sets:

```yaml
store_strategy_on_traverser_nodes: true
store_strategy_on_opponent_nodes: false
```

OpenSpiel reference
(`open_spiel/python/pytorch/deep_cfr.py::_traverse_game_tree`):

```python
elif state.current_player() == player:
    # ... compute regrets, append to ADVANTAGE buffer
    self._append_to_advantage_buffer(player, data)
    return cfv
else:
    other_player = state.current_player()
    _, strategy = self._sample_action_from_advantage(state, other_player)
    # ...
    data = StrategyMemory(
        np.array(state.information_state_tensor(other_player), ...),
        np.array(self._iteration, ...),
        np.array(strategy, dtype=np.float32),
    )
    self._append_to_stategy_buffer(data)
```

OpenSpiel: strategy samples are appended **only at opponent nodes** during
traverser=p's external-sampling tree walk, using the opponent's info state.

## Mechanism: why the convention differs by sampling mode

Average policy theory: π̄_p(I) = Σ_t ρ_p^t(I) · σ_p^t(I) / Σ_t ρ_p^t(I),
where ρ_p^t(I) is player p's contribution to the reach probability of I at
iteration t. The strategy network minimizes (CE/MSE) loss against samples
(I, σ_p^t(I)); the *empirical sample distribution* should be ∝ ρ_p(I) for
the regression to converge to π̄_p.

### External sampling (OpenSpiel default)

During p's traversal:
- At p-nodes, p enumerates all legal actions — so p's branching contributes
  no probability factor. Visit count at p-info-state I scales as ρ_opp(I).
- At opp-nodes, opp samples one action ~ σ_opp. Visit count at opp-info-state
  I scales as ρ_p(I) · ρ_opp(I).

To get π̄_opp samples weighted by ρ_opp, OpenSpiel records at opp-nodes
during p's traversal. The ρ_p factor on top is the iteration-mixing factor
(opp's strategy was fit while p was traversing) — over alternating-player
iterations this averages out as both players take turns being traverser.

If we instead recorded at p-nodes during p's traversal in external mode,
the empirical density would be ∝ ρ_opp(I_p) — which **drops the ρ_p
factor we wanted**, and overweights deep p-side info states reachable
mostly through opp-uniform play. That is the bias the reviewer flagged.

### Outcome sampling (our default)

During p's traversal both players sample on-policy along a single
trajectory:
- p-node info state I: visit prob ∝ ρ_p(I) · ρ_opp(I)
- opp-node info state I: visit prob ∝ ρ_p(I) · ρ_opp(I)

Both conventions give the same sample density up to the mixing factor of
the *other* player's reach. Neither is exactly ρ_p(I) without an
importance-sampling correction. The two are equivalent in expectation
modulo variance.

In practice we use uniform-priority reservoir sampling on the strategy
buffer rather than reach-weighted regression, so the residual ρ_other
factor is absorbed into "training-data distribution we accept" — it is
not a correctness bug, it is the Deep CFR approximation choice for both
conventions.

## Audit: what our default does in each mode

| `sampling_mode` | `store_on_traverser` | `store_on_opp` | OpenSpiel-equivalent? | Bias risk |
|---|---|---|---|---|
| `outcome` (current default) | true | false | n/a — OpenSpiel doesn't ship outcome Deep CFR | low; symmetric |
| `external` (if user flips) | **true** | **false** | **NO — reversed** | **high — drops ρ_p factor on π̄ samples** |
| `external` + flip flags | false | true | yes | matches reference |

## Practical implication

- The current default (outcome + traverser-node strategy) is not a bug for
  the algorithm we are actually running. Keep it.
- Add a config validation that warns/errors when
  `sampling_mode: external` is paired with
  `store_strategy_on_traverser_nodes: true` or
  `store_strategy_on_opponent_nodes: false`. This is the easy guardrail
  that catches the reviewer's scenario.
- When introducing an `open_spiel_like.yaml` preset for parity
  experiments, set:

  ```yaml
  traversal:
    sampling_mode: external
    store_strategy_on_traverser_nodes: false
    store_strategy_on_opponent_nodes: true
  ```

- Do **not** change the outcome-mode default to match OpenSpiel's
  external-mode convention — that would be a cargo-cult fix; the
  underlying reach-distribution argument does not transfer.

## References

- Brown, Lerer, Gross, Sandholm. *Deep Counterfactual Regret
  Minimization.* ICML 2019. (Algorithm 1 — strategy memory M_Π
  collection.)
- OpenSpiel `python/pytorch/deep_cfr.py`,
  `DeepCFRSolver._traverse_game_tree` (commit master @ 2026-05-07).
- `docs/research/outcome-sampling-target.md` — companion note on the
  advantage-target side of outcome sampling.
