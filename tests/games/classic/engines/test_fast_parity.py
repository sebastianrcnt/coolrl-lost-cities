from __future__ import annotations

import random

import pytest
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig, build_deck

from coolrl_lost_cities.games.classic.bots import RandomBot
from coolrl_lost_cities.games.classic.engines import FastGameState


def _card(color: int, rank: int) -> dict[str, int]:
    return {"color": color, "rank": rank}


def _classic_snapshot(
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


def test_fast_new_game_from_deck_matches_game_state_snapshot() -> None:
    config = LostCitiesConfig()
    deck = build_deck(config)

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


def test_fast_pending_discard_matches_game_state() -> None:
    snapshot = _classic_snapshot(
        hands=[
            [_card(0, 1)],
            [_card(1, 1)],
        ],
        deck=[_card(2, 1), _card(3, 1)],
    )
    classic = GameState.from_snapshot(snapshot)
    fast = FastGameState.from_snapshot(snapshot)

    classic.apply_action(1)
    fast.apply_action(1)
    assert fast.to_snapshot() == classic.to_snapshot()
    assert fast.legal_draw_mask() == classic.legal_draw_mask()
    assert fast.legal_draw_mask()[1] is False

    classic.apply_action(0)
    fast.apply_action(0)
    classic.apply_action(1)
    fast.apply_action(1)
    classic.apply_action(0)
    fast.apply_action(0)
    classic.apply_action(1)
    fast.apply_action(1)

    assert fast.to_snapshot() == classic.to_snapshot()
    assert fast.legal_draw_mask() == classic.legal_draw_mask()
    assert fast.legal_draw_mask()[1] is True


def test_fast_terminal_edges_match_game_state() -> None:
    last_draw_snapshot = _classic_snapshot(
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
    classic = GameState.from_snapshot(last_draw_snapshot)
    fast = FastGameState.from_snapshot(last_draw_snapshot)

    classic.apply_action(1)
    fast.apply_action(1)
    classic.apply_action(0)
    fast.apply_action(0)

    assert fast.to_snapshot() == classic.to_snapshot()
    assert fast.terminal is True

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
    classic = GameState.from_snapshot(defensive_snapshot, validate=False)
    fast = FastGameState.from_snapshot(defensive_snapshot, validate=False)

    classic.apply_action(1)
    fast.apply_action(1)

    assert fast.to_snapshot() == classic.to_snapshot()
    assert fast.terminal is True


def test_fast_last_numeric_legality_matches_game_state() -> None:
    handshake_snapshot = _classic_snapshot(
        hands=[
            [_card(0, 1)],
            [],
        ],
        expeditions=[
            [[_card(0, 0)], [], [], [], []],
            [[], [], [], [], []],
        ],
    )
    classic = GameState.from_snapshot(handshake_snapshot)
    fast = FastGameState.from_snapshot(handshake_snapshot)
    assert fast.legal_card_mask() == classic.legal_card_mask()
    assert fast.legal_card_mask()[0] is True

    numeric_snapshot = _classic_snapshot(
        hands=[
            [_card(0, 0), _card(0, 3), _card(0, 5)],
            [],
        ],
        expeditions=[
            [[_card(0, 4)], [], [], [], []],
            [[], [], [], [], []],
        ],
    )
    classic = GameState.from_snapshot(numeric_snapshot)
    fast = FastGameState.from_snapshot(numeric_snapshot)
    assert fast.legal_card_mask() == classic.legal_card_mask()
    assert fast.legal_card_mask()[0] is False
    assert fast.legal_card_mask()[2] is False
    assert fast.legal_card_mask()[4] is True


def test_fast_score_cache_and_undo_match_game_state() -> None:
    snapshot = _classic_snapshot(
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
    classic = GameState.from_snapshot(snapshot)
    fast = FastGameState.from_snapshot(snapshot)
    before = fast.to_snapshot()

    assert fast.expedition_score(0, 0) == classic.expedition_score(0, 0)
    assert fast.total_score(0) == classic.total_score(0)

    undo = fast.apply_action_with_undo(0)
    classic.apply_action(0)
    assert fast.to_snapshot() == classic.to_snapshot()
    assert fast.expedition_score(0, 0) == classic.expedition_score(0, 0)
    assert fast.total_score(0) == classic.total_score(0)

    fast.undo_action(undo)
    assert fast.to_snapshot() == before
    assert fast.total_score(0) == GameState.from_snapshot(before).total_score(0)


def test_fast_discard_draw_push_pop_restores_snapshot() -> None:
    snapshot = _classic_snapshot(
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


def test_fast_random_action_sequence_matches_game_state() -> None:
    config = LostCitiesConfig()
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


def test_fast_random_bot_self_play_matches_game_state() -> None:
    config = LostCitiesConfig()
    for seed in range(32):
        classic = GameState.new_game(config, seed=seed)
        fast = FastGameState.new_game(config, seed=seed)
        classic_bots = [RandomBot(seed=seed * 2), RandomBot(seed=seed * 2 + 1)]
        fast_bots = [RandomBot(seed=seed * 2), RandomBot(seed=seed * 2 + 1)]
        steps = 0

        while True:
            assert fast.to_snapshot() == classic.to_snapshot()
            if classic.terminal:
                break

            player = classic.current_player
            assert fast.current_player == player
            classic_action = classic_bots[player].act(classic)
            fast_action = fast_bots[player].act({"legal_mask": fast.legal_mask()})
            assert fast_action == classic_action

            classic.apply_action(classic_action)
            fast.apply_action(fast_action)
            steps += 1
            assert steps < 1000

        assert fast.total_score(0) == classic.total_score(0)
        assert fast.total_score(1) == classic.total_score(1)
        assert fast.score_diff(0) == classic.score_diff(0)


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
