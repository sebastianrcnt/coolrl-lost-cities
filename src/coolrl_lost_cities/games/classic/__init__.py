from __future__ import annotations

from .backends import build_backend
from .bots import (
    LostCitiesBot,
    available_bot_names,
    build_bot,
    play_game,
    run_series,
)
from .env import LostCitiesEnv
from .game import (
    GameState,
    IllegalMoveError,
    LostCitiesConfig,
    classic_config,
)
from .interfaces import BackendName, LostCitiesBackend, Snapshot

__all__ = [
    "BackendName",
    "GameState",
    "IllegalMoveError",
    "LostCitiesBackend",
    "LostCitiesBot",
    "LostCitiesConfig",
    "LostCitiesEnv",
    "Snapshot",
    "available_bot_names",
    "build_backend",
    "build_bot",
    "classic_config",
    "play_game",
    "run_series",
]


def main() -> None:
    config = classic_config()
    state = GameState.new_game(config)
    print(f"Lost Cities classic: {config.deck_size} cards, player {state.current_player} to act")
