from __future__ import annotations

from .bots import (
    LostCitiesPolicy,
    available_bot_names,
    build_bot,
)
from .env import LostCitiesEnv
from .evaluation import (
    GameResult,
    MatchEvalRecord,
    MatchResult,
    TimingResult,
    evaluate_policy,
    make_policy_factory,
    play_game_for_evaluation,
    play_match,
)
from .game import (
    GameState,
    IllegalMoveError,
    LostCitiesConfig,
    classic_config,
)
from .snapshots import Snapshot

__all__ = [
    "GameState",
    "GameResult",
    "IllegalMoveError",
    "LostCitiesPolicy",
    "LostCitiesConfig",
    "LostCitiesEnv",
    "MatchEvalRecord",
    "MatchResult",
    "Snapshot",
    "TimingResult",
    "available_bot_names",
    "build_bot",
    "classic_config",
    "evaluate_policy",
    "make_policy_factory",
    "play_game_for_evaluation",
    "play_match",
]


def main() -> None:
    config = classic_config()
    state = GameState.new_game(config)
    print(f"Lost Cities classic: {config.deck_size} cards, player {state.current_player} to act")
