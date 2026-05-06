from coolrl_lost_cities.games.classic.game import Card, GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots import RandomBot
from tests.games.classic.helpers import make_state


def test_legal_mask_has_action_in_nonterminal_phases() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=3))
    while not state.terminal:
        assert any(state.legal_mask())
        action = RandomBot(11).act(state)
        state.apply_action(action)


def test_empty_hand_slots_are_masked() -> None:
    state = make_state(hands=[[Card(0, 1)], []])
    mask = state.legal_card_mask()
    assert mask[0] is True
    assert mask[1] is True
    assert all(value is False for value in mask[2:])


def test_empty_discard_pile_draw_is_illegal() -> None:
    state = make_state(deck=[Card(0, 1)], phase="draw")
    mask = state.legal_draw_mask()
    assert mask[0] is True
    assert all(mask[1 + color] is False for color in range(state.config.n_colors))


def test_unified_legal_mask_has_fixed_shape_across_phases() -> None:
    config = LostCitiesConfig()
    state = GameState.new_game(config, seed=1)
    assert len(state.unified_legal_mask()) == config.action_size

    action = next(index for index, legal in enumerate(state.legal_mask()) if legal)
    state.apply_action(action)

    mask = state.unified_legal_mask()
    assert state.phase == "draw"
    assert len(mask) == config.action_size
    assert all(value is False for value in mask[: config.card_action_size])
    assert any(mask[config.card_action_size :])


def test_random_fuzz_invariants() -> None:
    config = LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=1, hand_size=5)
    bot = RandomBot(99)
    for seed in range(1000):
        state = GameState.new_game(config, seed=seed)
        steps = 0
        while not state.terminal:
            mask = state.legal_mask()
            assert any(mask)
            action = bot.act(state)
            state.apply_action(action)
            steps += 1
            assert steps < 1000
