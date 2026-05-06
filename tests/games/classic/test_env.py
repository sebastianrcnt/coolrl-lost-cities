import numpy as np

from coolrl_lost_cities.games.classic.env import LostCitiesEnv
from coolrl_lost_cities.games.classic.game import Card, GameState, LostCitiesConfig


def test_env_observation_uses_fixed_unified_mask() -> None:
    config = LostCitiesConfig()
    env = LostCitiesEnv(config)

    obs = env.reset()
    assert obs["legal_mask"].shape == (config.action_size,)
    assert env.phase == "card"

    card_action = int(np.nonzero(obs["legal_mask"])[0][0])
    obs, _, _, _ = env.step(card_action)

    assert env.phase == "draw"
    assert obs["legal_mask"].shape == (config.action_size,)
    assert np.all(obs["legal_mask"][:config.card_action_size] == 0)
    assert np.any(obs["legal_mask"][config.card_action_size:])


def test_env_step_accepts_legacy_draw_action_ids() -> None:
    config = LostCitiesConfig()
    env = LostCitiesEnv(config)
    env.state = GameState.empty(config)
    env.state.hands[0] = [Card(0, 1)]
    env.state.hands[1] = [Card(1, 1)]
    env.state.deck = [Card(2, 1), Card(2, 2)]
    env.state.phase = "draw"

    obs, reward, done, _ = env.step(0)

    assert obs["legal_mask"].shape == (config.action_size,)
    assert reward == 0.0
    assert done is False
    assert env.current_player == 1
    assert env.phase == "card"


def test_terminal_reward_is_relative_to_actor() -> None:
    config = LostCitiesConfig(
        n_colors=2,
        n_ranks=1,
        min_rank=1,
        n_handshakes=0,
        hand_size=1,
        expedition_penalty=0,
        bonus_threshold=99,
    )
    env = LostCitiesEnv(config)
    env.state = GameState.empty(config)
    env.state.current_player = 1
    env.state.phase = "draw"
    env.state.deck = [Card(1, 1)]
    env.state.expeditions[1][0] = [Card(0, 1)]

    _, reward, done, _ = env.step(config.card_action_size)

    assert done is True
    assert reward == 1.0
