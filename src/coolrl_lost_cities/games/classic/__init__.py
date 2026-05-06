from __future__ import annotations

from .game import Card, GameState, IllegalMoveError, LostCitiesConfig, classic_config

__all__ = [
    "Card",
    "GameState",
    "IllegalMoveError",
    "LostCitiesConfig",
    "classic_config",
]


def main() -> None:
    config = classic_config()
    state = GameState.new_game(config)
    print(f"Lost Cities classic: {config.deck_size} cards, player {state.current_player} to act")
