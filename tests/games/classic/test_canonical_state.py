import json
import random
from pathlib import Path

import pytest
from coolrl_lost_cities.games.classic.game import Card, GameState, LostCitiesConfig

import coolrl_lost_cities.games.classic as classic

FIXTURE_DIR = Path(classic.__file__).resolve().parent / "fixtures"


def _small_config() -> LostCitiesConfig:
    return LostCitiesConfig(
        n_colors=2,
        n_ranks=2,
        min_rank=1,
        n_handshakes=0,
        hand_size=1,
        expedition_penalty=0,
        bonus_threshold=99,
        bonus_amount=0,
    )


def test_new_game_from_deck_uses_explicit_internal_deck_order() -> None:
    config = _small_config()
    state = GameState.new_game_from_deck(
        [
            Card(0, 1),
            Card(0, 2),
            Card(1, 1),
            Card(1, 2),
        ],
        config,
    )

    assert state.hands == [[Card(1, 2)], [Card(1, 1)]]
    assert state.deck == [Card(0, 1), Card(0, 2)]
    state.validate_invariants()


def test_snapshot_roundtrip_preserves_json_state() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=5))
    first_action = next(index for index, legal in enumerate(state.unified_legal_mask()) if legal)
    state.apply_unified_action(first_action)
    second_action = next(index for index, legal in enumerate(state.unified_legal_mask()) if legal)
    state.apply_unified_action(second_action)

    payload = json.loads(json.dumps(state.to_snapshot()))
    restored = GameState.from_snapshot(payload)

    assert restored.to_snapshot() == state.to_snapshot()
    restored.validate_invariants()


def test_validate_invariants_detects_card_loss() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=7))
    state.deck.pop()

    with pytest.raises(ValueError, match="card conservation"):
        state.validate_invariants()


def test_validate_invariants_detects_bad_expedition_order() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=8))
    card = state.deck.pop()
    state.expeditions[0][card.color].extend([Card(card.color, 2), Card(card.color, 1)])
    state.deck.extend([Card(card.color, 2), Card(card.color, 1)])

    with pytest.raises(ValueError, match="strictly increasing"):
        state.validate_invariants()


def test_canonical_small_fixture_matches_expected_trace() -> None:
    fixture = json.loads((FIXTURE_DIR / "canonical_small.json").read_text())
    config = LostCitiesConfig(**fixture["config"])
    state = GameState.new_game_from_deck(fixture["initial_deck"], config)

    for step in fixture["steps"]:
        if step["action"] is not None:
            state.apply_unified_action(step["action"])
        assert state.phase == step["phase"]
        assert state.current_player == step["current_player"]
        assert state.turn_count == step["turn_count"]
        assert state.terminal is step["terminal"]
        assert state.score_diff(0) == step["score_diff_player0"]
        assert state.unified_legal_mask() == step["legal_mask"]
        state.validate_invariants()


def test_random_games_preserve_python_core_invariants() -> None:
    config = LostCitiesConfig(
        n_colors=3,
        n_ranks=5,
        min_rank=2,
        n_handshakes=1,
        hand_size=5,
    )
    for seed in range(128):
        state = GameState.new_game(config, seed=seed)
        rng = random.Random(seed ^ 0x5EED)
        steps = 0
        while not state.terminal:
            state.validate_invariants()
            legal = [index for index, is_legal in enumerate(state.unified_legal_mask()) if is_legal]
            state.apply_unified_action(rng.choice(legal))
            steps += 1
            assert steps < 1000
        state.validate_invariants()


def test_same_seed_and_action_sequence_are_deterministic() -> None:
    config = LostCitiesConfig(seed=1234)
    left = GameState.new_game(config)
    right = GameState.new_game(config)

    while True:
        assert left.to_snapshot() == right.to_snapshot()
        if left.terminal:
            break

        action = next(index for index, is_legal in enumerate(left.unified_legal_mask()) if is_legal)
        left.apply_unified_action(action)
        right.apply_unified_action(action)
