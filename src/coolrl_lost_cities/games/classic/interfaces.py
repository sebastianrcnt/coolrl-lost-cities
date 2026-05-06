from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

from .game import GameState
from .snapshots import Snapshot

BotInput: TypeAlias = dict | GameState | Snapshot


@runtime_checkable
class LostCitiesBot(Protocol):
    def act(self, obs_or_state: BotInput) -> int:
        """Choose an action id from the current state or observation."""
