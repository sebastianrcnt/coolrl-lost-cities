from __future__ import annotations

import random

import pytest
from coolrl_lost_cities.games.classic.game import Card, GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.engines import FastGameState


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


def test_fast_new_game_from_deck_matches_game_state_snapshot() -> None:
    config = _small_config()
    deck = [
        Card(0, 1),
        Card(0, 2),
        Card(1, 1),
        Card(1, 2),
    ]

    classic = GameState.new_game_from_deck(deck, config)
    fast = FastGameState.new_game_from_deck(deck, config)

    assert fast.to_snapshot() == classic.to_snapshot()
    fast.validate_invariants()


def test_fast_snapshot_roundtrip_preserves_snapshot() -> None:
    config = LostCitiesConfig(seed=11)
    classic = GameState.new_game(config)
    fast = FastGameState.from_snapshot(classic.to_snapshot())

    assert fast.to_snapshot() == classic.to_snapshot()
    restored = FastGameState.from_snapshot(fast.to_snapshot())
    assert restored.to_snapshot() == fast.to_snapshot()


def test_fast_from_snapshot_rejects_oversized_regions_before_write() -> None:
    config = _small_config()
    state = GameState.new_game(config, seed=3)

    deck_snapshot = state.to_snapshot()
    deck_snapshot["deck"] = deck_snapshot["deck"] + [
        {"color": 0, "rank": 1},
        {"color": 0, "rank": 2},
        {"color": 1, "rank": 1},
    ]
    with pytest.raises(ValueError, match="deck snapshot exceeds capacity"):
        FastGameState.from_snapshot(deck_snapshot)

    hand_snapshot = state.to_snapshot()
    hand_snapshot["hands"][0] = hand_snapshot["hands"][0] + [{"color": 0, "rank": 1}]
    with pytest.raises(ValueError, match="hand 0 snapshot exceeds hand_size"):
        FastGameState.from_snapshot(hand_snapshot)

    expedition_snapshot = state.to_snapshot()
    expedition_snapshot["expeditions"][0][0] = [
        {"color": 0, "rank": 1},
        {"color": 0, "rank": 2},
        {"color": 0, "rank": 1},
    ]
    with pytest.raises(ValueError, match="expedition 0/0 snapshot exceeds capacity"):
        FastGameState.from_snapshot(expedition_snapshot)

    discard_snapshot = state.to_snapshot()
    discard_snapshot["discards"][0] = [
        {"color": 0, "rank": 1},
        {"color": 0, "rank": 2},
        {"color": 0, "rank": 1},
    ]
    with pytest.raises(ValueError, match="discard 0 snapshot exceeds capacity"):
        FastGameState.from_snapshot(discard_snapshot)


def test_fast_validate_invariants_rejects_bad_expedition_order() -> None:
    config = _small_config()
    snapshot = FastGameState.new_game(config, seed=4).to_snapshot()
    snapshot["deck"].extend(
        [
            {"color": 0, "rank": 2},
            {"color": 0, "rank": 1},
        ]
    )
    snapshot["expeditions"][0][0] = [
        {"color": 0, "rank": 2},
        {"color": 0, "rank": 1},
    ]

    with pytest.raises(ValueError, match="expedition is not strictly increasing"):
        FastGameState.from_snapshot(snapshot)


def test_fast_random_action_sequence_matches_game_state() -> None:
    config = LostCitiesConfig(
        n_colors=3,
        n_ranks=5,
        min_rank=2,
        n_handshakes=1,
        hand_size=5,
    )
    for seed in range(48):
        classic = GameState.new_game(config, seed=seed)
        fast = FastGameState.new_game(config, seed=seed)
        rng = random.Random(seed ^ 0xF457)
        steps = 0

        while True:
            assert fast.to_snapshot() == classic.to_snapshot()
            assert fast.unified_legal_mask() == classic.unified_legal_mask()
            assert fast.unified_legal_actions() == [
                index for index, is_legal in enumerate(classic.unified_legal_mask()) if is_legal
            ]
            assert fast.score_diff(0) == classic.score_diff(0)
            if classic.terminal:
                break

            legal = [
                index for index, is_legal in enumerate(classic.unified_legal_mask()) if is_legal
            ]
            action = rng.choice(legal)
            classic.apply_unified_action(action)
            fast.apply_unified_action(action)
            steps += 1
            assert steps < 1000


def test_fast_apply_undo_restores_every_legal_action() -> None:
    config = LostCitiesConfig(
        n_colors=3,
        n_ranks=5,
        min_rank=2,
        n_handshakes=1,
        hand_size=5,
    )
    for seed in range(32):
        state = FastGameState.new_game(config, seed=seed)
        rng = random.Random(seed ^ 0xFA57A11)
        steps = 0

        while not state.terminal:
            legal = [index for index, is_legal in enumerate(state.unified_legal_mask()) if is_legal]
            for action in legal:
                candidate = state.clone()
                before = candidate.to_snapshot()
                undo = candidate.apply_unified_action_with_undo(action)
                candidate.undo_action(undo)
                assert candidate.to_snapshot() == before
                candidate.validate_invariants()

            state.apply_unified_action(rng.choice(legal))
            steps += 1
            assert steps < 1000
