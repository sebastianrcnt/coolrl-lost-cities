from __future__ import annotations

from .bots import (
    LostCitiesBot,
    available_bot_names,
    build_bot,
    play_game,
    run_series,
)
from .env import LostCitiesEnv
from .evaluation import (
    GameResult,
    MatchResult,
    evaluate_bot,
    make_bot_factory,
    play_game_for_evaluation,
    play_match,
)
from .game import (
    GameState,
    IllegalMoveError,
    LostCitiesConfig,
    classic_config,
)
from .interfaces import Snapshot

__all__ = [
    "GameState",
    "GameResult",
    "IllegalMoveError",
    "LostCitiesBot",
    "LostCitiesConfig",
    "LostCitiesEnv",
    "MatchResult",
    "Snapshot",
    "available_bot_names",
    "build_bot",
    "classic_config",
    "evaluate_bot",
    "make_bot_factory",
    "play_game",
    "play_game_for_evaluation",
    "play_match",
    "run_series",
]


def main() -> None:
    config = classic_config()
    state = GameState.new_game(config)
    print(f"Lost Cities classic: {config.deck_size} cards, player {state.current_player} to act")
