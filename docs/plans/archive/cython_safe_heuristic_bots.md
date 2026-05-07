# Plan: Cython Port of the Safe-Heuristic Bot Family

**Status:** Archived. Implemented as `heuristic_cy.pyx` with Python shim and equivalence tests.
**Owner:** Codex
**Background:** See `docs/performance.md` → "Evaluation Breakdown" and "Post-A Optimization Calculus". The safe-heuristic family dominates eval wall-clock through `opponent_act_seconds` (7.57s / 4.63s / 9.22s for `safe_heuristic` / `safe_heuristic_loose` / `safe_heuristic_strict` in the inspected eval row), not GPU forward. As denser eval becomes operationally useful (`eval_every: 5`, `evaluation.games: 1000`), this cost becomes a first-order wall-clock concern.

## Goal

Port `SafeHeuristicBot` (and its loose / strict parameterizations) from pure-Python `bots/heuristic.py` to Cython, consuming game state directly through the existing typed Cython `GameState` interface, so that the per-iteration evaluation cost dominated by `opponent_act_seconds` shrinks substantially without any change to bot decision behavior.

## Non-goals

- Do not change bot decision logic. The Cython port must be byte-identical to the Python implementation under fixed seeds for the full action sequence of every benchmarked seeded game.
- Do not port `RandomBot`, `PassiveDiscardBot`, or `NoisyPolicy`. Their per-iter cost is small (see `docs/performance.md` table: `random` 0.06s, `passive_discard` 0.00s, `noisy_safe` 0.57s).
- Do not change the public bot registry names or CLI surface.
- Do not touch the Cython traversal pipeline, encoding, training, or replay paths. This is a bot-only change.
- Do not redesign `LostCitiesPolicy` or the `PolicyInput` / `Snapshot` interfaces.

## Success criteria

1. **Equivalence (hard).** For a fixed corpus of seeded games (see "Equivalence strategy" below), the Cython implementation produces a byte-identical action id sequence to the Python implementation for each of `safe_heuristic`, `safe_heuristic_loose`, `safe_heuristic_strict`. Deviation in any single action fails CI.
2. **Per-opponent speedup.** On `configs/deep_cfr/default.yaml` evaluation, per-opponent `opponent_act_seconds` for each safe-heuristic variant decreases by **≥3×** versus the pre-port baseline measured on the same hardware. Concretely, target post-port values:
   - `safe_heuristic`: ≤ 2.5s (from 7.57s)
   - `safe_heuristic_loose`: ≤ 1.5s (from 4.63s)
   - `safe_heuristic_strict`: ≤ 3.1s (from 9.22s)
3. **Eval wall-clock.** With the same `evaluation.games` and `evaluation.eval_every`, `eval/<variant>/elapsed_seconds` for each safe-heuristic variant decreases proportionally, and total `evaluation_seconds` for an iteration that runs the full opponent suite decreases meaningfully (the safe-heuristic opponents are the dominant tail per the performance doc).
4. **All existing tests pass**, including every `test_safe_heuristic_*` test in `tests/games/classic/test_bots.py`. New equivalence tests (see below) also pass.
5. **No behavioral regression in training.** A short training run on `default.yaml` with the Cython bots in eval reproduces the same eval-winrate trajectory (within seed noise) as the Python implementation over at least one full eval cadence.

## Independence claim

This work is **fully independent** of:

- the batched traversal inference server plan (`docs/plans/batched_traversal_inference_server.md`),
- the AMP wiring (`run.use_amp`),
- the `torch.compile` experiment branch (`experiments/torch-compile`).

It touches no traversal, training, or networks code. It can ship in parallel with any of those efforts. The eval pipeline already calls `bot.act(state)` per opponent turn; replacing that bot's implementation language is local to the bots package.

## Key files (current)

- `src/coolrl_lost_cities/games/classic/bots/heuristic.py` — pure-Python implementation. ~1160 lines. Fully read and analyzed. Contains:
  - Module-level helpers `play_action`, `discard_action`, `draw_from_discard_action` (used by tests).
  - `SafeHeuristicParams` (frozen dataclass, behavioral knobs).
  - `DerivedHeuristicConfig` (frozen dataclass, derived constants).
  - `derive_heuristic_config(config, params)` — `lru_cache(maxsize=64)` cached.
  - `SafeHeuristicBot(LostCitiesPolicy)` — the policy itself, with `_act_card` / `_act_draw` and ~20 helper methods.
- `src/coolrl_lost_cities/games/classic/bots/base.py` — `legal_from_obs`, `first_legal`. The fallback path (non-`GameState` input) routes through these. The Cython port must preserve this fallback behavior unchanged.
- `src/coolrl_lost_cities/games/classic/bots/registry.py` — `BOT_REGISTRY`, `LOOSE_SAFE_HEURISTIC_PARAMS`, `STRICT_SAFE_HEURISTIC_PARAMS`. Currently constructs `SafeHeuristicBot` from the Python class.
- `src/coolrl_lost_cities/games/classic/game.pxd` — typed `GameState` declaration. The bot will consume this through Python attribute access (sufficient — see "Cython tactics" below) and via direct `cpdef` calls (`legal_card_mask`, `legal_draw_mask`, `score_diff`, `can_play_card`, etc.).
- `src/coolrl_lost_cities/games/classic/policy.py` — `LostCitiesPolicy` ABC and `PolicyInput` typedef.
- `setup.py` — Cython extension list.
- `tests/games/classic/test_bots.py` — existing safe-heuristic tests (`test_safe_heuristic_mirror_match_finishes`, `test_safe_heuristic_opponent_value_ignores_hidden_hand`, `test_safe_heuristic_started_expedition_value_ignores_invalid_lower_followup`, `test_safe_heuristic_draws_playable_discard_instead_of_deck`, `test_safe_heuristic_can_draw_discard_to_deny_opponent_when_losing`, `test_safe_heuristic_classic_self_play_opens_expeditions`, `test_safe_heuristic_avoids_opening_weak_fifth_color`, `test_safe_heuristic_prefers_followup_on_started_expedition`, `test_safe_heuristic_avoids_unopened_discard_draw_after_four_opens`).

## New files

- `src/coolrl_lost_cities/games/classic/bots/heuristic.pyx` — single Cython file containing the ported `SafeHeuristicBot` class plus `cdef` helpers. One file (not split per variant): the variants differ only in `SafeHeuristicParams` values, not in code.
- `src/coolrl_lost_cities/games/classic/bots/heuristic.pxd` — minimal typed declarations for the bot class and its hot helper signatures, so future extensions (or other Cython modules) can import typed entry points. Optional in Step 1; required in Step 3 if Cython call-site overhead from Python attribute lookup dominates measurement.
- `tests/games/classic/test_safe_heuristic_equivalence.py` — equivalence tests (see "Equivalence strategy").

## Files to touch

- `src/coolrl_lost_cities/games/classic/bots/heuristic.py` — **keep as a thin wrapper** that re-exports `SafeHeuristicBot`, `SafeHeuristicParams`, `DerivedHeuristicConfig`, `derive_heuristic_config`, `play_action`, `discard_action`, `draw_from_discard_action`, `PLAY_OR_DISCARD_ACTIONS_PER_SLOT`, `DRAW_FROM_DECK_ACTION` from `heuristic.pyx`. This preserves all existing imports (`tests/games/classic/test_bots.py` imports `from coolrl_lost_cities.games.classic.bots.heuristic import draw_from_discard_action`). Do not delete the file. Do not duplicate logic.
- `src/coolrl_lost_cities/games/classic/bots/registry.py` — no functional change. Imports continue to resolve through the `heuristic.py` shim.
- `setup.py` — add the new extension:
  ```python
  Extension(
      "coolrl_lost_cities.games.classic.bots.heuristic",
      ["src/coolrl_lost_cities/games/classic/bots/heuristic.pyx"],
  ),
  ```

## Equivalence strategy

Behavior drift is the dominant risk. Equivalence tests gate the merge of every step.

### Test design

`tests/games/classic/test_safe_heuristic_equivalence.py` builds a fixed corpus of seeded `LostCitiesConfig` × seed pairs, plays full games to terminal, and asserts the Cython and Python bots emit byte-identical action ids at every turn.

Corpus:

- **Configs**: at minimum, the default `LostCitiesConfig()` from `tests/games/classic/helpers.py`. Add the small-tier config used by `test_safe_heuristic_started_expedition_value_ignores_invalid_lower_followup` if it exists, plus a tier where `n_handshakes == 0` (exercises the `state.config.n_handshakes <= 0` early return in `_best_handshake_play`).
- **Seeds**: `range(0, 50)` per config. Fifty full games per config gives broad coverage of decision branches without making CI slow.
- **Variants**: all three (`SafeHeuristicBot()`, `SafeHeuristicBot(LOOSE_SAFE_HEURISTIC_PARAMS)`, `SafeHeuristicBot(STRICT_SAFE_HEURISTIC_PARAMS)`). Loose and strict shift only the parameter dataclass, but each must be tested independently because their thresholds drive different branches in `_should_open_expedition`, `_best_handshake_play`, and the visible-draw support logic.

### Driver

Use a self-play harness (mirror match) that constructs two clone `GameState` instances per seed: one driven by the Python bot, one by the Cython bot. At each turn, both bots receive the same `GameState`. The test asserts:

```python
assert cython_action == python_action, (turn, seed, variant, state_summary)
```

If either bot diverges mid-game, the states diverge and subsequent comparisons are meaningless — so abort the comparison on first disagreement and report `(seed, variant, turn, action_py, action_cy)`.

### Branch coverage targets

The corpus must hit all decision paths. Each of these branches must be exercised by at least one (variant, seed, turn) tuple:

- `_act_card` → handshake play taken.
- `_act_card` → number play on a started expedition.
- `_act_card` → speculative open (`speculative_open=True`).
- `_act_card` → strong open (`strong_open=True`).
- `_act_card` → exceptional open (`exceptional_open=True`).
- `_act_card` → single-late open (`single_late_open=True`).
- `_act_card` → forced open path (no expeditions started, no normal play).
- `_act_card` → discard path with `unusable_discard_bonus`.
- `_act_card` → discard path with `discard_safety_bonus`.
- `_act_draw` → draw from deck.
- `_act_draw` → draw from discard (handshake top card).
- `_act_draw` → draw from discard (numeric top card).
- `_act_draw` → unopened-color discard penalty triggers (`opened_colors >= 4`).
- `_visible_draw_value` → exceptional-support short-circuit.
- Loose vs strict thresholds: at least one (seed, turn) where loose opens an expedition that strict declines, and one where strict's `late_open_block_threshold` blocks an opening that loose would take.

A coverage assertion at the end of the test sweep records which of the above branches fired (via a side-channel counter inside a debug build of the bot, or by analyzing the action stream); CI fails if any branch was not hit.

### Floating-point determinism

The bot uses Python `float` arithmetic with `max(...)` over `(value, action)` tuples. In Python, ties break by `action` ordering (since the value is the first element of the tuple). The Cython port **must use the same tie-break order**: stable `(value, action)` comparison, where ties prefer the lower-numbered action that came first in the iteration. Implementation: collect `(value, action)` pairs in the same iteration order as Python, then linear-scan for the maximum, replacing only on strict `>`. This matches Python's `max` semantics on a list-of-tuples.

All arithmetic must be `double` (matches Python `float`). Avoid `cdivision` floating-point divergence — there is essentially no division in the hot path, but `0.25 * number_sum` and similar must remain `double`.

## Cython-ization tactics

The hot loops are inside `_best_number_play`, `_best_discard`, `_visible_draw_value`, `_color_commitment`, `_public_color_commitment_for_opponent`, `_bonus_potential`, and `_opening_plan_value`. They iterate over hand cards (≤ ~10) and colors (≤ 6) per call, and the bot is called once per opponent turn (~30 turns/game × N games × N opponents).

### Tactics, in priority order

1. **Type the bot class as `cdef class SafeHeuristicBot`** with `cdef` helpers. Convert all `_<name>` methods to `cdef inline double _<name>(...)` (or `cdef int` for action ids) where possible. Keep `act` as `cpdef` so the Python registry can construct and call it.
2. **Type all hot locals** as `int` / `double` / `Py_ssize_t`. The dominant cost in the Python version is per-card list comprehensions building intermediate `list[Card]` objects; replace with explicit indexed loops over `state.hands[player]` and short fixed-size `cdef double` accumulators. No intermediate Python lists in the hot path.
3. **Cache `state.config` accessors once per `act` call.** `state.config.expedition_penalty`, `bonus_threshold`, `bonus_amount`, `n_colors`, `min_rank`, `max_rank`, `n_handshakes`, `n_ranks`, `hand_size`, `deck_size` — pull all of these into typed locals at the top of `act`.
4. **Cache `derive_heuristic_config` per `act`.** Already cached via `lru_cache`, but each lookup re-hashes the config. Pull the resulting `DerivedHeuristicConfig` into a typed local; access its fields once.
5. **Avoid Python-object operations in inner loops.** `Card` objects expose `color`, `rank`, `is_handshake`, `numeric_value(min_rank)`. In the Cython port, when the Python `Card` object is unavoidable (it is part of the public `GameState.hands` shape), bind its three relevant attributes to typed locals at the top of each loop iteration. Do not call `card.numeric_value(min_rank)` inside conditions repeatedly — compute it once per card per iteration.
6. **Direct typed access where possible.** `GameState.score_diff(player)` is `cpdef int` and `can_play_card` is a Python wrapper around `can_play_encoded_card` (`cpdef bint`). Where the bot calls `state.can_play_card(player, card)` for a `Card` object, the Cython port can encode the card once (`color * cards_per_color + (rank - min_rank)` per game.pyx encoding) and call `state.can_play_encoded_card(player, encoded)` directly. **Verify the encoding formula against `game.pyx` `_encode_card` before relying on it; otherwise keep the Python `can_play_card` call and accept the small overhead.**
7. **`cpdef list legal_card_mask(self)` returns a Python list of `bool`.** Type the binding as `list legal` in the Cython bot, and index it with typed `Py_ssize_t`. If profiling shows mask access dominates, convert callers to use `unified_legal_mask_np()` and read it as a typed numpy view — but only after Step 4 measurements show this is needed.
8. **Inline tiny helpers.** `_num`, `_late_penalty`, `_new_color_open_penalty` are one-liners. Inline them with `cdef inline`.
9. **Compiler directives.** Use the same directives as `setup.py` already applies project-wide: `boundscheck=False`, `wraparound=False`, `cdivision=True`, `initializedcheck=False`. The bot file inherits these from `setup.py`'s `compiler_directives` block — do not need to override per-file.

### What NOT to optimize

- Do not rewrite `derive_heuristic_config` to drop its `lru_cache`. It is called once per `act` and the cache hit is fast.
- Do not replace `GameState.hands[player]` with raw int-array access. The `Card` Python object boundary is the public API, and crossing it is what `_card_obj` already costs in Cython. Keep the boundary; just don't re-cross it inside tight loops.
- Do not memoize across calls. The bot is stateless; each `act` call gets a fresh state.

## Risks and mitigations

- **Behavior drift introduced silently.** Mitigation: equivalence tests (above) run in CI on every PR. They are the gate. Benchmarks are not run until equivalence is green.
- **Floating-point ordering difference between Python and Cython.** Mitigation: explicit `(value, action)` linear-scan max with strict `>` comparison; identical iteration order. Equivalence tests would surface any divergence.
- **`Card` object identity vs equality.** The Python code uses `is not` comparisons (`if other is not card`, `if followup is not card`, `if card is not exclude_card`). Mitigation: in Cython, preserve `is`-comparison semantics by using object-pointer identity (`PyObject*` compare) or by passing the slot index instead of the Card. Slot-index passing is preferred — it sidesteps identity altogether.
- **`max(candidates)[1]` Python tuple semantics.** Mitigation: documented above. Strict `>` linear scan.
- **Cached `derive_heuristic_config` shared between Python and Cython invocations.** Mitigation: the Cython port must import `derive_heuristic_config` from the same module path so the LRU cache is shared (or rebuild a Cython-side cache keyed identically). Easiest: keep `derive_heuristic_config` and `DerivedHeuristicConfig` in the same `.pyx` and re-export from the `.py` shim.
- **Build-system regression.** Adding an extension that fails to compile in CI on a system without the right toolchain. Mitigation: the project already builds three `.pyx` files (`game`, `cfr_math`, `encoding`, `traversal`); the toolchain is established. Add the new extension and run `uv run python -c "import coolrl_lost_cities.games.classic.bots.heuristic"` locally before pushing.
- **Performance regression on small tiers.** The bot may not actually be the bottleneck in micro-config evaluation. Mitigation: benchmarks are run on `default.yaml` (the contract surface), not on smoke configs.

## Implementation steps (each independently mergeable)

### Step 1: scaffolding (no behavior change)

- Add `bots/heuristic.pyx` containing the entire current `bots/heuristic.py` body verbatim, with no `cdef` or typing changes — just renamed.
- Convert `bots/heuristic.py` to a re-export shim (`from .heuristic_impl import *` style) — but to avoid a circular import via the `.pyx` module name, keep both as `heuristic.{py,pyx}` is not workable. Instead, name the new file `heuristic_cy.pyx` (Cython compiles to `heuristic_cy`), and have `heuristic.py` import the public symbols from `heuristic_cy`. Update `setup.py` accordingly.
  - **Decision point:** if Cython supports a `.pyx` shadowing a `.py` in the same package, prefer that for a cleaner import path. Otherwise use the `heuristic_cy` naming. Resolve at Step 1 implementation time and update this plan inline.
- Run all existing tests to confirm zero behavior change.
- Merge.

### Step 2: type the base `SafeHeuristicBot`

- Convert `SafeHeuristicBot` to `cdef class`. Add typed locals to the hot helpers listed under "Cython tactics". Keep the data classes (`SafeHeuristicParams`, `DerivedHeuristicConfig`) as Python frozen dataclasses — they are not hot.
- Run the equivalence test suite (built in Step 4 — but for this step, run the existing `test_safe_heuristic_*` tests as a proxy gate; the full equivalence suite lands in Step 4).
- Merge.

### Step 3: type `cdef inline` helpers and inner loops

- Inline `_num`, `_late_penalty`, `_new_color_open_penalty`. Convert `_color_commitment`, `_public_color_commitment_for_opponent`, `_bonus_potential`, `_opening_plan_value`, `_visible_draw_value`, `_visible_open_support_value`, `_visible_number_can_help_open` to typed `cdef double` / `cdef bint` helpers.
- Replace list comprehensions with indexed loops over `state.hands[player]`, accumulating into typed scalars.
- Run equivalence suite. Merge only if green.

### Step 4: equivalence test suite

- Add `tests/games/classic/test_safe_heuristic_equivalence.py` with the corpus and branch-coverage assertions described under "Equivalence strategy".
- Wire into `pytest -q tests/games/classic/test_safe_heuristic_equivalence.py`. Confirm the test passes against the Step 3 build. If it surfaces drift, fix Step 3 before continuing.
- Merge.
- **Note on ordering vs Step 2/3:** ideally Step 4 lands before Step 2 so equivalence is a green gate the entire time. Recommended order: 1 → 4 (against the verbatim port, must be a no-op pass) → 2 → 3.

### Step 5: variant validation

- Loose and strict differ only by params, but exercise their threshold differences explicitly. Confirm `LOOSE_SAFE_HEURISTIC_PARAMS` and `STRICT_SAFE_HEURISTIC_PARAMS` in `registry.py` flow through the Cython implementation unchanged.
- Confirm `NoisyPolicy(SafeHeuristicBot(), RandomBot(seed))` still works (`NoisyPolicy` lives in `registry.py` and wraps the bot; the wrapper does not need porting).
- Add a small targeted test that constructs all three variants and runs one full game each.
- Merge.

### Step 6: benchmark and documentation

- Benchmark protocol: run `configs/deep_cfr/default.yaml` evaluation (full opponent suite, default `evaluation.games`) once on the pre-port commit and once on the post-port commit, on the same `home` machine, with no other GPU load. Record:
  - `eval/safe_heuristic/elapsed_seconds`, `eval/safe_heuristic/opponent_act_seconds`,
  - `eval/safe_heuristic_loose/elapsed_seconds`, `eval/safe_heuristic_loose/opponent_act_seconds`,
  - `eval/safe_heuristic_strict/elapsed_seconds`, `eval/safe_heuristic_strict/opponent_act_seconds`,
  - `evaluation_seconds` for the iteration.
- Confirm success criteria 2 and 3.
- Add a date-stamped subsection to `docs/performance.md` recording results, methodology, and any branch-coverage gaps surfaced during equivalence testing.
- Merge.

## Out-of-scope follow-ups (do not start)

- Porting `RandomBot`, `PassiveDiscardBot`, or `NoisyPolicy` to Cython. Their measured cost is small. Re-evaluate only if a future profile shows them on the critical path.
- Replacing `Card` Python objects with raw int encoding inside the bot interface. This would require touching `LostCitiesPolicy` and `PolicyInput` more broadly, expanding scope.
- Caching across `act` calls (e.g., remembering opened-color counts). The bot is stateless by design; introducing state risks correctness for marginal speedup.
- TensorRT / `torch.compile` on policy networks during eval. Tracked separately under the inference-server plan and `docs/performance.md` § "Post-A Optimization Calculus".
- Vectorizing eval to compute many game states' bot actions simultaneously. Substantial rework of the eval driver; out of scope here.
