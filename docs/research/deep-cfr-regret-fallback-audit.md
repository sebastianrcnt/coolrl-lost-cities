# Deep CFR Regret Matching Fallback and Early Over-Opening

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-regret-fallback-audit-2026-05-07.md`

## Question

In Deep CFR, when all legal actions at a traverser node have non-positive estimated regrets, the standard regret matching algorithm cannot produce a probability distribution by normalizing positive regrets. In such cases, a "fallback" policy must be used. Does the choice of fallback policy—specifically the default uniform distribution—contribute to the "over-opening" pathology (where the agent starts too many expeditions) observed in early Lost Cities training runs?

Short answer: **Yes.** Diagnostic audits show that in early iterations, nearly half of all regret-matching decisions fall back to the default policy because the network predicts negative regrets for all legal actions. In Lost Cities, the large number of legal "open new expedition" actions causes a uniform fallback to select an opening move more frequently than a more informed tie-breaking strategy would.

## Code reference

The regret matching fallback logic is implemented in `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx` within `_policy_from_networks` (around line 477):

```cython
for i in range(self.action_size):
    if legal[i] != 0:
        positive = adv_view[i] if adv_view[i] > 0.0 else 0.0
        positive_sum += positive
self.last_policy_regret_fallback = positive_sum <= self.epsilon
```

If `positive_sum` is below the threshold, the policy is determined by the `all_negative_fallback` configuration. The default `uniform` mode uses `regret_matching_c` from `src/coolrl_lost_cities/games/classic/deep_cfr/cfr_math.pyx` (line 31), which assigns `1.0 / legal_count` to every legal action.

The alternative `argmax_tiebreak` mode (added in the May 2026 audit) instead identifies the legal action(s) with the highest (least negative) advantage:

```cython
if selected < 0 or adv_view[i] > best:
    selected = i
    best = adv_view[i]
    tie_count = 1
elif adv_view[i] == best:
    tie_count += 1
    if _next_u32(&self.rng) % <unsigned int>tie_count == 0:
        selected = i
```

## Analysis

In Lost Cities, every move consists of playing or discarding a card, followed by drawing a card. Opening a new expedition incurs an immediate -20 point penalty. Early in training, before the advantage networks have learned the long-term value of expeditions, they frequently assign negative regrets to all "open new" actions.

If a traverser has 5 cards in hand that could each open a new expedition, and 3 cards that could be discarded or played on existing expeditions, a uniform fallback over these 8 legal actions will select an "open new" move with 62.5% probability. Because the network hasn't yet learned to distinguish between these "bad" (negative regret) moves, it defaults to a high-entropy selection that over-samples the most numerous action type: opening new expeditions.

The audit at iteration 20 revealed:
- Under **uniform** fallback, the fallback rate was **46.6%**, and the agent opened an average of **4.43** colors per game.
- Under **argmax_tiebreak**, the fallback rate dropped to **15.4%** (as the network only needs to find *one* action it "dislikes least"), and the agent opened **4.61** colors in the same iteration, though it showed lower 5-color opening counts against specific opponents.

The "over-opening" is not necessarily a failure of the network to learn, but a byproduct of how high-entropy fallbacks interact with the game's action space geometry.

## Practical implication

- **Uniform fallback is a source of exploration noise** that is biased by the action count of specific move types. In Lost Cities, this noise pushes the traverser toward starting expeditions it cannot afford.
- **Argmax tie-breaking reduces fallback frequency** by forcing the agent to follow the network's relative preferences even when all absolute preferences are negative. This acts as a variance reduction technique for the policy.
- While `argmax_tiebreak` reduces the frequency of "blind" openings, it should be used cautiously; if the network's relative rankings are also noise, it may converge to a different but equally pathological attractor.
- For Lost Cities Deep CFR, `argmax_tiebreak` is recommended for further evaluation as a potential fix for the early-game opening plateau.

## References

- `docs/archive/deep-cfr-regret-fallback-audit-2026-05-07.md`
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal_stats.py` (Implementation of fallback audit metrics).