from __future__ import annotations

from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.policy_gradient import (
    fine_tune_strategy_policy_gradient,
)


def test_policy_gradient_fine_tune_smoke() -> None:
    config = LostCitiesConfig(seed=71)
    state = GameState.new_game(config, seed=71)
    network = DeepCFRMLP(input_dim(state), 2 * config.hand_size + 1 + config.n_colors, 16)

    metrics = fine_tune_strategy_policy_gradient(
        network,
        config,
        episodes=1,
        seed=71,
        max_steps=64,
    )

    assert metrics.episodes == 1
    assert isinstance(metrics.avg_reward, float)
    assert isinstance(metrics.loss, float)
