from __future__ import annotations

from dataclasses import replace

from ..game import GameState, LostCitiesConfig
from ..interfaces import LostCitiesBot

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Lost Cities bots") from exc


def play_game(
    bot0: LostCitiesBot,
    bot1: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> GameState:
    game_config = replace(config, seed=seed) if seed is not None else config
    state = GameState.new_game(game_config)
    bots = [bot0, bot1]
    for _ in range(max_steps):
        if state.terminal:
            return state
        action = bots[state.current_player].act(state)
        state.apply_action(action)
    raise RuntimeError(f"game exceeded max_steps={max_steps}")


def run_series(
    bot0: LostCitiesBot,
    bot1: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    games: int = 100,
    seed: int = 0,
) -> dict:
    diffs: list[int] = []
    wins0 = 0
    wins1 = 0
    draws = 0
    for index in range(games):
        state = play_game(bot0, bot1, config, seed=seed + index)
        diff = state.score_diff(0)
        diffs.append(diff)
        if diff > 0:
            wins0 += 1
        elif diff < 0:
            wins1 += 1
        else:
            draws += 1
    return {
        "games": games,
        "avg_diff": float(np.mean(diffs)) if diffs else 0.0,
        "wins0": wins0,
        "wins1": wins1,
        "draws": draws,
    }
