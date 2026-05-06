import pytest
from coolrl_lost_cities.games.classic.game import (
    Card,
    GameState,
    IllegalMoveError,
    LostCitiesConfig,
    build_deck,
)

from tests.games.classic.helpers import make_state


def test_deck_generation_count() -> None:
    config = LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=1, hand_size=5)
    assert len(build_deck(config)) == config.n_colors * (config.n_ranks + config.n_handshakes)


def test_initial_hands_remove_cards_from_deck() -> None:
    config = LostCitiesConfig(seed=7)
    state = GameState.new_game(config)
    assert len(state.hands[0]) == config.hand_size
    assert len(state.hands[1]) == config.hand_size
    assert len(state.deck) == config.deck_size - 2 * config.hand_size


def test_play_must_be_ascending() -> None:
    config = LostCitiesConfig()
    state = make_state(
        config,
        hands=[[Card(0, 2)], []],
        expeditions=[[[Card(0, 4)], [], [], [], []], [[], [], [], [], []]],
    )
    assert state.legal_card_mask()[0] is False


def test_handshake_after_number_forbidden() -> None:
    config = LostCitiesConfig()
    state = make_state(
        config,
        hands=[[Card(1, 0)], []],
        expeditions=[[[], [Card(1, 1)], [], [], []], [[], [], [], [], []]],
    )
    assert state.legal_card_mask()[0] is False


def test_cannot_draw_just_discarded_color() -> None:
    config = LostCitiesConfig()
    state = make_state(config, deck=[Card(0, 1)], hands=[[Card(2, 2)], []])
    state.apply_action(1)
    mask = state.legal_draw_mask()
    assert mask[1 + 2] is False


def test_drawing_just_discarded_color_is_rejected() -> None:
    config = LostCitiesConfig()
    state = make_state(config, deck=[Card(0, 1)], hands=[[Card(2, 2)], []])

    state.apply_action(1)

    with pytest.raises(IllegalMoveError):
        state.apply_action(1 + 2)


def test_discarded_color_can_be_drawn_after_turn_advances() -> None:
    config = LostCitiesConfig()
    state = make_state(
        config,
        deck=[Card(1, 1), Card(1, 2)],
        hands=[[Card(2, 2)], [Card(0, 1)]],
    )
    state.apply_action(1)
    state.apply_action(0)
    assert state.current_player == 1
    state.apply_action(1)
    assert state.phase == "draw"
    assert state.legal_draw_mask()[1 + 2] is True


def test_discarded_card_is_removed_when_drawn_later() -> None:
    config = LostCitiesConfig()
    state = make_state(
        config,
        deck=[Card(1, 1), Card(1, 2)],
        hands=[[Card(2, 2)], [Card(0, 1)]],
    )

    state.apply_action(1)
    assert state.discards[2] == [Card(2, 2)]

    state.apply_action(0)
    assert state.current_player == 1

    state.apply_action(1)
    state.apply_action(1 + 2)

    assert state.discards[2] == []
    assert Card(2, 2) in state.hands[1]


def test_deck_exhaustion_ends_after_last_deck_draw() -> None:
    config = LostCitiesConfig()
    state = make_state(config, deck=[Card(1, 1)], hands=[[Card(0, 1)], []])
    state.apply_action(1)
    state.apply_action(0)
    assert state.terminal is True
    assert len(state.deck) == 0


def test_card_phase_can_end_game_when_no_draw_sources_exist() -> None:
    config = LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=1, hand_size=5)
    state = make_state(config, hands=[[Card(0, 1)], [Card(1, 1)]])
    state.apply_action(1)
    assert state.phase == "draw"
    assert state.terminal is True
