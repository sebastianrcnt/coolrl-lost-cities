# Deep CFR Cython Traversal Architecture

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-v0-plan.md`

## Question

How can Deep CFR traversal be implemented efficiently for the Lost Cities game-state hot path while correctly handling hidden information and chance sampling?

The core challenge in scaling Deep CFR for Lost Cities is the overhead of state manipulation and information-state encoding. A Python-heavy implementation would bottleneck on object instantiation and legal action masking. The architectural solution is a Cython-native traversal engine that operates directly on the C-level `GameState` API, using mutation and undo (push/pop) instead of state cloning.

## Code reference

The traversal engine is implemented in `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`. It relies on the `GameState` C API defined in `src/coolrl_lost_cities/games/classic/game.pxd`.

### Chance Sampling and Mutation

Instead of cloning the entire game state for each chance node or action, the traverser uses `_push_action_c` and `_pop_action_c` to navigate the tree. For chance sampling (deck order), it uses C-level deck swaps.

`src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx` (around line 311):

```cython
state._push_action_c(fixed_action)
# ... traversal ...
state._pop_action_c()
if is_chance:
    state._swap_deck_cards_c(swapped_deck_index, state.deck_len - 1)
```

This mutation-based approach significantly reduces memory allocation and garbage collection pressure compared to `state.clone()` calls in the inner loop of a Monte Carlo traversal.

### Information-State Encoding

The encoding layer translates the game state into a fixed-size vector for the advantage and strategy networks. It is critical that this encoding does not "leak" hidden information (such as the opponent's hand or future deck contents) which the current player cannot legally observe.

`src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx`, function `_encode_info_state_with_flags_c` (around line 291):

```cython
out[idx] = 1.0 if state.phase_id == 0 else 0.0
# ...
out[idx] = <float>state.deck_len / <float>state.total_cards
# ... encoding player hand, public expeditions, and discard piles ...
```

The encoding includes:
- Current game phase and active player.
- The acting player's private hand.
- Both players' public expeditions and the shared discard piles.
- The remaining deck ratio and current scores.
- Legal action masks for the current state.

## Analysis

### Performance Rationale

By keeping the entire traversal loop—from legal action generation to regret matching—within Cython, the system avoids the "Python tax" on every node transition. The `GameState` C structure provides direct pointer access to deck and hand buffers, allowing the traverser to perform thousands of rollouts per second on a single thread. 

### Chance Handling via Mutation

In Deep CFR, modeling chance nodes by sampling compatible deck orders is standard. However, the efficiency of this sampling is often a bottleneck. The choice to use `_swap_deck_cards_c` followed by a `push_action` allows the traverser to simulate a specific chance outcome (e.g., drawing a specific card) without rebuilding the state. The `pop_action` call restores the deck and state counters, maintaining the integrity of the search tree.

### Non-leaking Encoding

A common failure mode in Deep CFR is "cheating" by including hidden information in the information-state vector. The v0 architecture strictly separates the `GameState` (which knows all) from the `encoding` (which only sees the current player's perspective). This ensures that the trained networks learn a true imperfect-information strategy rather than a policy that relies on privileged state knowledge.

## Practical implication

- **Developer Note:** When modifying the `GameState` or adding new card types, ensure that `_push_action_c` and `_pop_action_c` are updated to correctly restore any new state variables.
- **Performance:** Any new features in the encoding (e.g., "derived playability" flags) should be implemented in `encoding.pyx` using C-level loops to avoid slowing down the traverser.
- **Verification:** Always verify that `encoding.pyx` does not access `state.hand_cards[1 - player]` or other private indices belonging to the opponent.

## References

- Brown, Lerer, Gross, Sandholm. *Deep Counterfactual Regret Minimization.* ICML 2019.
- Zinkevich, Johanson, Bowling, Piccione. *Regret Minimization in Games with Incomplete Information.* NeurIPS 2007. (CFR foundation.)
- `docs/archive/deep-cfr-v0-plan.md`: Original design document for the Cython traversal pipeline.