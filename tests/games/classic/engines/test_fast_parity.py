from __future__ import annotations

import random

import pytest
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig, build_deck

from coolrl_lost_cities.games.classic.bots import RandomBot
from coolrl_lost_cities.games.classic.engines import FastGameState


def _card(color: int, rank: int) -> dict[str, int]:
    return {"color": color, "rank": rank}


def _snapshot(
    *,
    deck: list[dict[str, int]] | None = None,
    hands: list[list[dict[str, int]]] | None = None,
    expeditions: list[list[list[dict[str, int]]]] | None = None,
    discards: list[list[dict[str, int]]] | None = None,
    current_player: int = 0,
    phase: str = "card",
    pending_discarded_color: int | None = None,
    turn_count: int = 0,
    terminal: bool = False,
) -> dict:
    config = LostCitiesConfig()
    deck = list(deck or [])
    hands = hands or [[], []]
    expeditions = expeditions or [[[] for _ in range(config.n_colors)] for _ in range(2)]
    discards = discards or [[] for _ in range(config.n_colors)]

    used = []
    used.extend(deck)
    for hand in hands:
        used.extend(hand)
    for player_expeditions in expeditions:
        for expedition in player_expeditions:
            used.extend(expedition)
    for discard in discards:
        used.extend(discard)

    remaining = [card.to_snapshot() for card in build_deck(config)]
    for card in used:
        remaining.remove(card)

    return {
        "config": config.to_snapshot(),
        "deck": remaining + deck,
        "hands": hands,
        "expeditions": expeditions,
        "discards": discards,
        "current_player": current_player,
        "phase": phase,
        "pending_discarded_color": pending_discarded_color,
        "turn_count": turn_count,
        "terminal": terminal,
    }


def test_public_game_state_alias_matches_fast_new_game_from_deck_snapshot() -> None:
    config = LostCitiesConfig()
    deck = build_deck(config)

    assert GameState is FastGameState
    left = GameState.new_game_from_deck(deck, config)
    right = FastGameState.new_game_from_deck(deck, config)

    assert right.to_snapshot() == left.to_snapshot()
    right.validate_invariants()


def test_fast_snapshot_roundtrip_preserves_snapshot() -> None:
    config = LostCitiesConfig(seed=11)
    left = GameState.new_game(config)
    right = FastGameState.from_snapshot(left.to_snapshot())

    assert right.to_snapshot() == left.to_snapshot()
    restored = FastGameState.from_snapshot(right.to_snapshot())
    assert restored.to_snapshot() == right.to_snapshot()


def test_fast_from_snapshot_rejects_oversized_regions_before_write() -> None:
    config = LostCitiesConfig()
    state = GameState.new_game(config, seed=3)

    deck_snapshot = state.to_snapshot()
    deck_snapshot["deck"] = [_card(0, 1)] * (config.deck_size + 1)
    with pytest.raises(ValueError, match="deck snapshot exceeds capacity"):
        FastGameState.from_snapshot(deck_snapshot)

    hand_snapshot = state.to_snapshot()
    hand_snapshot["hands"][0] = [_card(0, 1)] * (config.hand_size + 1)
    with pytest.raises(ValueError, match="hand 0 snapshot exceeds hand_size"):
        FastGameState.from_snapshot(hand_snapshot)

    expedition_snapshot = state.to_snapshot()
    expedition_snapshot["expeditions"][0][0] = [_card(0, 1)] * (
        config.n_ranks + config.n_handshakes + 1
    )
    with pytest.raises(ValueError, match="expedition 0/0 snapshot exceeds capacity"):
        FastGameState.from_snapshot(expedition_snapshot)

    discard_snapshot = state.to_snapshot()
    discard_snapshot["discards"][0] = [_card(0, 1)] * (config.n_ranks + config.n_handshakes + 1)
    with pytest.raises(ValueError, match="discard 0 snapshot exceeds capacity"):
        FastGameState.from_snapshot(discard_snapshot)


def test_fast_validate_invariants_rejects_bad_expedition_order() -> None:
    config = LostCitiesConfig()
    snapshot = FastGameState.new_game(config, seed=4).to_snapshot()
    snapshot["deck"].extend(
        [
            _card(0, 2),
            _card(0, 1),
        ]
    )
    snapshot["expeditions"][0][0] = [
        _card(0, 2),
        _card(0, 1),
    ]

    with pytest.raises(ValueError, match="expedition is not strictly increasing"):
        FastGameState.from_snapshot(snapshot)


def test_fast_pending_discard_sequence_is_deterministic() -> None:
    snapshot = _snapshot(
        hands=[
            [_card(0, 1)],
            [_card(1, 1)],
        ],
        deck=[_card(2, 1), _card(3, 1)],
    )
    left = GameState.from_snapshot(snapshot)
    right = FastGameState.from_snapshot(snapshot)

    left.apply_action(1)
    right.apply_action(1)
    assert right.to_snapshot() == left.to_snapshot()
    assert right.legal_draw_mask() == left.legal_draw_mask()
    assert right.legal_draw_mask()[1] is False

    left.apply_action(0)
    right.apply_action(0)
    left.apply_action(1)
    right.apply_action(1)
    left.apply_action(0)
    right.apply_action(0)
    left.apply_action(1)
    right.apply_action(1)

    assert right.to_snapshot() == left.to_snapshot()
    assert right.legal_draw_mask() == left.legal_draw_mask()
    assert right.legal_draw_mask()[1] is True


def test_fast_terminal_edges_are_deterministic() -> None:
    last_draw_snapshot = _snapshot(
        deck=[_card(1, 1)],
        hands=[
            [_card(0, 1)],
            [],
        ],
    )
    remaining_deck = last_draw_snapshot["deck"][:-1]
    last_draw_snapshot["deck"] = [last_draw_snapshot["deck"][-1]]
    for card in remaining_deck:
        last_draw_snapshot["discards"][card["color"]].append(card)
    left = GameState.from_snapshot(last_draw_snapshot)
    right = FastGameState.from_snapshot(last_draw_snapshot)

    left.apply_action(1)
    right.apply_action(1)
    left.apply_action(0)
    right.apply_action(0)

    assert right.to_snapshot() == left.to_snapshot()
    assert right.terminal is True

    defensive_snapshot = {
        "config": LostCitiesConfig().to_snapshot(),
        "deck": [],
        "hands": [[_card(0, 1)], []],
        "expeditions": [[[] for _ in range(5)] for _ in range(2)],
        "discards": [[] for _ in range(5)],
        "current_player": 0,
        "phase": "card",
        "pending_discarded_color": None,
        "turn_count": 0,
        "terminal": False,
    }
    left = GameState.from_snapshot(defensive_snapshot, validate=False)
    right = FastGameState.from_snapshot(defensive_snapshot, validate=False)

    left.apply_action(1)
    right.apply_action(1)

    assert right.to_snapshot() == left.to_snapshot()
    assert right.terminal is True


def test_fast_last_numeric_legality_edges() -> None:
    handshake_snapshot = _snapshot(
        hands=[
            [_card(0, 1)],
            [],
        ],
        expeditions=[
            [[_card(0, 0)], [], [], [], []],
            [[], [], [], [], []],
        ],
    )
    left = GameState.from_snapshot(handshake_snapshot)
    right = FastGameState.from_snapshot(handshake_snapshot)
    assert right.legal_card_mask() == left.legal_card_mask()
    assert right.legal_card_mask()[0] is True

    numeric_snapshot = _snapshot(
        hands=[
            [_card(0, 0), _card(0, 3), _card(0, 5)],
            [],
        ],
        expeditions=[
            [[_card(0, 4)], [], [], [], []],
            [[], [], [], [], []],
        ],
    )
    left = GameState.from_snapshot(numeric_snapshot)
    right = FastGameState.from_snapshot(numeric_snapshot)
    assert right.legal_card_mask() == left.legal_card_mask()
    assert right.legal_card_mask()[0] is False
    assert right.legal_card_mask()[2] is False
    assert right.legal_card_mask()[4] is True


def test_fast_score_cache_and_undo_restore_snapshot() -> None:
    snapshot = _snapshot(
        hands=[
            [_card(0, 7)],
            [],
        ],
        expeditions=[
            [
                [
                    _card(0, 0),
                    _card(0, 0),
                    _card(0, 1),
                    _card(0, 2),
                    _card(0, 3),
                    _card(0, 4),
                    _card(0, 5),
                    _card(0, 6),
                ],
                [],
                [],
                [],
                [],
            ],
            [[], [], [], [], []],
        ],
    )
    left = GameState.from_snapshot(snapshot)
    right = FastGameState.from_snapshot(snapshot)
    before = right.to_snapshot()

    assert right.expedition_score(0, 0) == left.expedition_score(0, 0)
    assert right.total_score(0) == left.total_score(0)

    undo = right.apply_action_with_undo(0)
    left.apply_action(0)
    assert right.to_snapshot() == left.to_snapshot()
    assert right.expedition_score(0, 0) == left.expedition_score(0, 0)
    assert right.total_score(0) == left.total_score(0)

    right.undo_action(undo)
    assert right.to_snapshot() == before
    assert right.total_score(0) == GameState.from_snapshot(before).total_score(0)


def test_fast_discard_draw_push_pop_restores_snapshot() -> None:
    snapshot = _snapshot(
        hands=[[], [_card(1, 1)]],
        discards=[[_card(0, 1)], [], [], [], []],
        phase="draw",
    )
    state = FastGameState.from_snapshot(snapshot)
    before = state.to_snapshot()

    assert state.push_action(1) == 1
    state.validate_invariants()
    assert state.pop_action() == 1
    assert state.to_snapshot() == before


def test_fast_random_action_sequence_is_deterministic() -> None:
    config = LostCitiesConfig()
    for seed in range(48):
        left = GameState.new_game(config, seed=seed)
        right = FastGameState.new_game(config, seed=seed)
        rng = random.Random(seed ^ 0xF457)
        steps = 0

        while True:
            assert right.to_snapshot() == left.to_snapshot()
            assert right.unified_legal_mask() == left.unified_legal_mask()
            assert right.unified_legal_actions() == [
                index for index, is_legal in enumerate(left.unified_legal_mask()) if is_legal
            ]
            assert right.score_diff(0) == left.score_diff(0)
            if left.terminal:
                break

            legal = [index for index, is_legal in enumerate(left.unified_legal_mask()) if is_legal]
            action = rng.choice(legal)
            left.apply_unified_action(action)
            right.apply_unified_action(action)
            steps += 1
            assert steps < 1000


def test_fast_random_bot_self_play_is_deterministic() -> None:
    config = LostCitiesConfig()
    for seed in range(32):
        left = GameState.new_game(config, seed=seed)
        right = FastGameState.new_game(config, seed=seed)
        left_bots = [RandomBot(seed=seed * 2), RandomBot(seed=seed * 2 + 1)]
        right_bots = [RandomBot(seed=seed * 2), RandomBot(seed=seed * 2 + 1)]
        steps = 0

        while True:
            assert right.to_snapshot() == left.to_snapshot()
            if left.terminal:
                break

            player = left.current_player
            assert right.current_player == player
            left_action = left_bots[player].act(left)
            right_action = right_bots[player].act({"legal_mask": right.legal_mask()})
            assert right_action == left_action

            left.apply_action(left_action)
            right.apply_action(right_action)
            steps += 1
            assert steps < 1000

        assert right.total_score(0) == left.total_score(0)
        assert right.total_score(1) == left.total_score(1)
        assert right.score_diff(0) == left.score_diff(0)


def test_fast_apply_undo_restores_every_legal_action() -> None:
    config = LostCitiesConfig()
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


def test_fast_push_pop_action_restores_nested_sequence() -> None:
    config = LostCitiesConfig()
    for seed in range(32):
        state = FastGameState.new_game(config, seed=seed)
        rng = random.Random(seed ^ 0x517ACC)
        before = state.to_snapshot()
        actions: list[int] = []

        for depth in range(20):
            if state.terminal:
                break
            legal = state.unified_legal_actions()
            action = rng.choice(legal)
            actions.append(action)
            assert state.push_unified_action(action) == depth + 1
            state.validate_invariants()

        for action in reversed(actions):
            assert state.pop_action() == state.from_unified_action(action)
            state.validate_invariants()

        assert state.to_snapshot() == before

    with pytest.raises(ValueError, match="undo stack is empty"):
        FastGameState.new_game(config, seed=1).pop_action()
