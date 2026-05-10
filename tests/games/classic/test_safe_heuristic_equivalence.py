from __future__ import annotations

import pytest
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots.heuristic import HeuristicBot
from coolrl_lost_cities.games.classic.bots.heuristic_py import (
    HeuristicBot as PythonHeuristicBot,
)
from coolrl_lost_cities.games.classic.bots.registry import (
    AGGRESSIVE_HEURISTIC_PARAMS,
    CAUTIOUS_HEURISTIC_PARAMS,
)

VARIANTS = (
    ("default", None),
    ("loose", AGGRESSIVE_HEURISTIC_PARAMS),
    ("strict", CAUTIOUS_HEURISTIC_PARAMS),
)

CONFIGS = (
    LostCitiesConfig(),
    LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3),
    LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=0, hand_size=5),
)


@pytest.mark.parametrize(("variant_name", "params"), VARIANTS)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", range(2))
def test_cython_heuristic_matches_python_action_sequence(
    variant_name: str,
    params,
    config: LostCitiesConfig,
    seed: int,
) -> None:
    py_bot = PythonHeuristicBot(params)
    cy_bot = HeuristicBot(params)
    py_state = GameState.new_game(config, seed=seed)
    cy_state = GameState.new_game(config, seed=seed)

    turn = 0
    while not py_state.terminal:
        assert turn < 500, f"variant={variant_name} seed={seed} did not terminate"
        py_action = py_bot.act(py_state)
        cy_action = cy_bot.act(cy_state)
        assert cy_action == py_action, (
            f"variant={variant_name} seed={seed} turn={turn} "
            f"phase={py_state.phase} player={py_state.current_player} "
            f"python={py_action} cython={cy_action}"
        )
        py_state.apply_action(py_action)
        cy_state.apply_action(cy_action)
        turn += 1

    assert cy_state.terminal is True
