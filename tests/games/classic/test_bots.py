from coolrl_lost_cities.games.classic.game import Card, GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots import (
    HeuristicBot,
    LostCitiesPolicy,
    RandomBot,
)
from coolrl_lost_cities.games.classic.bots.heuristic import draw_from_discard_action
from coolrl_lost_cities.games.classic.evaluation import play_game_for_evaluation
from tests.games.classic.helpers import make_state


def _expeditions(config: LostCitiesConfig) -> list[list[list[Card]]]:
    return [
        [[] for _ in range(config.n_colors)],
        [[] for _ in range(config.n_colors)],
    ]


def test_builtin_bots_implement_lost_cities_policy() -> None:
    assert isinstance(RandomBot(1), LostCitiesPolicy)
    assert isinstance(HeuristicBot(), LostCitiesPolicy)


def test_heuristic_mirror_match_finishes() -> None:
    state, result = play_game_for_evaluation(
        HeuristicBot(),
        HeuristicBot(),
        LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=1, hand_size=5),
        seed=2000,
        max_steps=200,
    )
    assert state.terminal is True
    assert result.timed_out is False


def test_heuristic_opponent_value_ignores_hidden_hand() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)
    bot = HeuristicBot()
    discard_card = Card(color=0, rank=6)

    expeditions_a = _expeditions(config)
    expeditions_a[1][0] = [Card(color=0, rank=0), Card(color=0, rank=4)]
    state_a = make_state(
        config,
        hands=[[], [Card(color=0, rank=5)]],
        expeditions=expeditions_a,
        discards=[[discard_card], []],
    )

    expeditions_b = _expeditions(config)
    expeditions_b[1][0] = [Card(color=0, rank=0), Card(color=0, rank=4)]
    state_b = make_state(
        config,
        hands=[
            [],
            [Card(color=0, rank=5), Card(color=0, rank=7), Card(color=0, rank=8)],
        ],
        expeditions=expeditions_b,
        discards=[[discard_card], []],
    )

    value_a = bot._card_value_for_opponent(
        state=state_a,
        opponent=1,
        card=discard_card,
        derived=bot._derived(state_a),
    )
    value_b = bot._card_value_for_opponent(
        state=state_b,
        opponent=1,
        card=discard_card,
        derived=bot._derived(state_b),
    )

    assert value_a == value_b


def test_heuristic_started_expedition_value_ignores_invalid_lower_followup() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)
    bot = HeuristicBot()
    high_card = Card(color=0, rank=8)

    base_expeditions = _expeditions(config)
    base_expeditions[0][0] = [Card(color=0, rank=4)]
    base_state = make_state(
        config,
        hands=[[high_card], []],
        expeditions=base_expeditions,
    )

    lower_expeditions = _expeditions(config)
    lower_expeditions[0][0] = [Card(color=0, rank=4)]
    lower_followup_state = make_state(
        config,
        hands=[[Card(color=0, rank=5), high_card], []],
        expeditions=lower_expeditions,
    )

    base_value = bot._started_expedition_play_value(
        state=base_state,
        player=0,
        card=high_card,
        derived=bot._derived(base_state),
        deck_left=config.deck_size,
    )
    lower_followup_value = bot._started_expedition_play_value(
        state=lower_followup_state,
        player=0,
        card=high_card,
        derived=bot._derived(lower_followup_state),
        deck_left=config.deck_size,
    )

    assert lower_followup_value == base_value


def test_heuristic_draws_playable_discard_instead_of_deck() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)
    bot = HeuristicBot()

    expeditions = _expeditions(config)
    expeditions[0][0] = [Card(color=0, rank=4)]
    state = make_state(
        config,
        deck=[Card(color=1, rank=8)],
        expeditions=expeditions,
        discards=[[Card(color=0, rank=6)], []],
        phase="draw",
    )

    assert bot._act_draw(state) == draw_from_discard_action(0)


def test_heuristic_can_draw_discard_to_deny_opponent_when_losing() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=4)
    bot = HeuristicBot()

    expeditions = _expeditions(config)
    expeditions[0][1] = [Card(color=1, rank=8)]
    expeditions[1][0] = [
        Card(color=0, rank=0),
        Card(color=0, rank=5),
        Card(color=0, rank=6),
        Card(color=0, rank=7),
        Card(color=0, rank=8),
    ]
    state = make_state(
        config,
        deck=[Card(color=1, rank=8), Card(color=1, rank=7)],
        hands=[[Card(color=0, rank=0), Card(color=0, rank=7)], []],
        expeditions=expeditions,
        discards=[[Card(color=0, rank=6)], []],
        phase="draw",
    )

    assert state.score_diff(0) < 0
    assert bot._act_draw(state) == draw_from_discard_action(0)


def test_heuristic_classic_self_play_opens_expeditions() -> None:
    state = GameState.new_game(LostCitiesConfig(), seed=1)
    bot = HeuristicBot()
    player0_actions: list[int] = []

    for _ in range(60):
        if state.terminal:
            break
        action = bot.act(state)
        unified = state.to_unified_action(action)
        if state.current_player == 0:
            player0_actions.append(unified)
        state.apply_unified_action(unified)

    play_actions = [
        action
        for action in player0_actions
        if action < state.config.card_action_size and action % 2 == 0
    ]

    assert play_actions
    assert any(state.expeditions[0][color] for color in range(state.config.n_colors))


def test_heuristic_avoids_opening_weak_fifth_color() -> None:
    config = LostCitiesConfig(n_colors=5, n_ranks=8, hand_size=8)
    bot = HeuristicBot()
    expeditions = _expeditions(config)
    expeditions[0][0] = [Card(color=0, rank=4)]
    expeditions[0][1] = [Card(color=1, rank=4)]
    expeditions[0][2] = [Card(color=2, rank=5)]
    expeditions[0][3] = [Card(color=3, rank=6)]
    weak_open = Card(color=4, rank=4)
    state = make_state(
        config,
        hands=[[weak_open, Card(color=4, rank=7), Card(color=0, rank=6)], []],
        expeditions=expeditions,
    )
    state.sort_hand(0)

    assert (
        bot._should_open_expedition(
            state=state,
            player=0,
            color=4,
            opening_card=weak_open,
            derived=bot._derived(state),
            deck_left=config.deck_size,
        )
        is False
    )


def test_heuristic_prefers_followup_on_started_expedition() -> None:
    config = LostCitiesConfig(n_colors=3, n_ranks=8, hand_size=5)
    bot = HeuristicBot()
    expeditions = _expeditions(config)
    expeditions[0][0] = [Card(color=0, rank=4)]
    state = make_state(
        config,
        hands=[
            [Card(color=0, rank=6), Card(color=1, rank=4), Card(color=1, rank=7)],
            [],
        ],
        expeditions=expeditions,
    )
    state.sort_hand(0)

    action = bot._act_card(state)
    chosen = state.hands[0][action // 2]

    assert action % 2 == 0
    assert chosen.color == 0


def test_heuristic_avoids_unopened_discard_draw_after_four_opens() -> None:
    config = LostCitiesConfig(n_colors=5, n_ranks=8, hand_size=8)
    bot = HeuristicBot()
    expeditions = _expeditions(config)
    expeditions[0][0] = [Card(color=0, rank=4)]
    expeditions[0][1] = [Card(color=1, rank=4)]
    expeditions[0][2] = [Card(color=2, rank=5)]
    expeditions[0][3] = [Card(color=3, rank=6)]
    discards = [[] for _ in range(config.n_colors)]
    discards[4] = [Card(color=4, rank=5)]
    state = make_state(
        config,
        deck=[Card(color=0, rank=8), Card(color=1, rank=8)],
        hands=[[Card(color=4, rank=4), Card(color=4, rank=7)], []],
        expeditions=expeditions,
        discards=discards,
        phase="draw",
    )

    assert bot._act_draw(state) == 0
