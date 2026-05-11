# SO-ISMCTS BC Ceiling — 2026-05-11 Autonomous Session

**Last verified:** 2026-05-11, commit `cba6cae` (branch `autonomous/trap-exploration`)

## Short answer

Under our current compute budget (1 GPU, 12 CPU cores, 50 MCTS sims/move,
768x4 MLP), **behavior-cloning the heuristic-balanced bot is the ceiling**.
Across 13 self-play training variants, no run cleared the BC baseline of
21/100 wins vs `heuristic-cautious` in 100-game evaluation. Every variant
either preserved BC (KL anchor, mirror-descent target) or regressed toward
catastrophic forgetting (naive finetune, high-fraction mixed opponent).

The single largest improvement of the session came from PUCT Q-value
normalization at the *search* level, not from any learning change.

## Headline numbers (vs heuristic-cautious, 100 games, n_sims = 16)

| Run               | Setup                                          | W/100 | Notes |
|-------------------|------------------------------------------------|------:|-------|
| BC pretrain       | 5k heuristic-vs-heuristic games, 20 epochs CE+MSE |  21 | baseline |
| C9 naive finetune | BC + plain self-play (no regularizer)          |   0 | catastrophic forgetting |
| C10 KL β=1.0      | BC + self-play + KL(current ‖ BC)              | ~21 | preserved BC, no improvement |
| C11 KL β=0.3      | weaker anchor                                  | ~21 | preserved BC, no improvement |
| C12 mirror desc.  | target = softmax(α log π_mcts + (1−α) log π_BC) |  19 | preserved BC, no improvement |
| C13 mixed=0.5     | 50 % games vs heuristic-balanced, opponent-aware MCTS, no KL | 0 | forgetting (worse than naive) |
| C14 mixed=0.2     | mixed-opponent + opponent-aware + KL β=1.0     |  17 | preserved BC, no improvement |

CIs (Wilson 95 %) overlap across all "preserved BC" rows; the 17–22 band
is statistically indistinguishable from the BC baseline.

## What actually moved the needle: PUCT Q normalization

`mcts.pyx _select_action` previously used the raw score-unit Q:

```
score = q_eff + c_puct * prior * sqrt(N) / (1 + n)
```

With `value_scale = 100` (Lost Cities score units), a single backup could
swing `q_eff` by ±100, while the exploration bonus is ~1–10. A noisy value
at the root permanently buried low-prior actions before they could be
explored.

Fix (`b9fc569`): divide Q by `q_scale` (defaults to 100) before scoring:

```
score = q_eff / q_scale + c_puct * prior * sqrt(N) / (1 + n)
```

Replaying the exact same BC checkpoint with this fix took win rate vs
heuristic-cautious from **5/100 → 21/100** — a 4× improvement from a
~10-line search change, with no retraining. Worth holding onto as the
load-bearing finding of the session.

## Hypotheses we negated

1. **Symmetric self-play eventually escapes the weak fixed point.**
   Random-init + KL-free self-play ran for 100s of iterations across
   C1–C8 without exceeding the noise floor (0–25 wins, all CIs overlap
   each other and zero).

2. **Mixed-opponent self-play (Codex top pick) breaks the weak
   equilibrium.** With opponent-aware MCTS so the search distribution
   reflects the real opponent (per Codex's "pitfall" warning), C13
   regressed to 0/100. The training signal from vs-heuristic games is
   structurally negative — BC cannot beat the heuristic, so every mixed
   sample is a loss, and the gradient labels every BC action as bad.
   C14 cut the fraction to 0.2 and added a strong KL anchor (β = 1.0),
   which preserved BC but did not lift it.

3. **Deeper search compensates for weak learning.** Increasing
   `n_simulations` from 50 → 200 on the BC checkpoint *reduced* wins
   vs `heuristic-balanced` from 28/64 → 14/64 in earlier probing.
   Deeper search amplifies the network's preferences, including its
   weaker ones, without supplying new information.

4. **A different regularizer would let self-play improve on BC.**
   KL anchor (β ∈ {0.3, 1.0}) and mirror-descent target mixing (α
   annealed 0.3 → 0.8) both kept the network glued to BC. Neither
   supplied a positive gradient to walk away from it.

## Why BC is the ceiling — the mechanism

Self-play seeded from a strong heuristic faces a structural trap:

- BC has internalized the heuristic. Two BC copies playing each other
  produce a near-symmetric outcome distribution; the visit counts at
  most nodes give little policy-improvement signal beyond what BC
  already encodes.
- Against the real heuristic, BC loses systematically (the heuristic
  beats its own clone in approx. 79 % of games at our scale). The
  resulting training signal is uniformly negative; learning that
  signal pushes the policy *away* from BC without pointing anywhere
  productive.
- With 50 MCTS sims/move on a 768x4 network, the search cannot
  reliably *find* moves that beat the heuristic. So the only way out
  of the trap — discovering a positive improvement direction —
  is closed by the search-depth budget.

The result is consistent with the standard SO-ISMCTS picture: π_weak
(the symmetric weak fixed point) sits at roughly BC strength, π_Nash
is unreachable at this compute, and every variant we tried collapses
onto π_weak.

## Things left as configurable dials (no behavior change at defaults)

The `autonomous/trap-exploration` branch leaves the following in place
for future runs with more compute:

- `MctsConfig.q_scale` — PUCT Q normalization (defaults to 100, keep).
- `MctsConfig.root_dirichlet_alpha / epsilon` — AlphaZero exploration noise.
- `MctsConfig.opponent_aware_search` — when true, MCTS treats the
  opponent seat as an external bot (skips tree expansion on opponent
  turns, traverser-centered values).
- `TrainingConfig.mixed_opponent_fraction` — 0 disables (pure self-play).
- `TrainingConfig.mixed_opponent_bot` — bot name from
  `coolrl_lost_cities.games.classic.bots.registry`.
- `TrainingConfig.kl_anchor_ckpt` / `kl_anchor_beta` — frozen reference
  network for `KL(current ‖ ref)` regularization.
- `TrainingConfig.md_target_ref_ckpt` / `md_target_alpha_*` — mirror-
  descent policy target with annealed mixing.
- `lost-cities-ismcts pretrain` — heuristic behavior cloning subcommand.
- `lost-cities-ismcts eval --ckpt … --n-sims N --games N --device cpu`
  — standalone evaluator with Wilson CIs (`eval_checkpoint.py`).

## What would be worth trying with more compute

Not implemented here. These are the directions that the mechanism above
*does not rule out*:

- **Deeper search at training time** (n_sims ≫ 200, e.g. 800–1600).
  Enough simulations should eventually surface a heuristic-beating
  action somewhere in the search tree; that's a positive gradient.
- **Population training with frozen snapshots.** Periodically snapshot
  the trainer and route 10–20 % of self-play games against the snapshot
  pool. Combined with opponent-aware search, this gives a stationary
  diverse-opponent gradient without the all-negative-signal problem of
  pure heuristic mixing.
- **Value-weighted replay.** Prioritize high-error samples in the
  buffer so the value head sees the cases where it disagrees with the
  search rollout.
- **Larger / better-shaped networks.** 768x4 MLP may simply lack the
  capacity to represent the conjunctions Lost Cities needs (color ×
  expedition × hand composition). Attention or factored heads could be
  worth probing.

## Code references

- Search-side: `src/coolrl_lost_cities/games/classic/ismcts/mcts.pyx`
  (`_select_action`, `prepare_simulation`, `_expand_with_prior`).
- Python parity: `src/coolrl_lost_cities/games/classic/ismcts/mcts.py`.
- Mixed-opponent / opponent-aware wiring:
  `src/coolrl_lost_cities/games/classic/ismcts/interleaved_self_play.py`.
- Regularization (KL anchor, mirror-descent) and metrics:
  `src/coolrl_lost_cities/games/classic/ismcts/trainer.py`.
- BC pretrain: `src/coolrl_lost_cities/games/classic/ismcts/pretrain.py`.
- Eval CLI with Wilson CIs:
  `src/coolrl_lost_cities/games/classic/ismcts/eval_checkpoint.py`.
- BC checkpoint (load with `--resume-from`):
  `runs/pretrain/heuristic_balanced_5kg_20ep.pt` (5 k games, 20 epochs).

## Related memory

- `opponent-policy-network-divergence.md` — the Deep CFR analogue:
  using the live network as its own opponent breaks stationarity and
  diverges. The SO-ISMCTS picture here is the same family of failure:
  bootstrapping from oneself does not provide a positive learning
  signal.
