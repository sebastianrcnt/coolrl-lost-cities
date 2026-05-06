from __future__ import annotations

import logging

from ..game import GameState, LostCitiesConfig
from ..interfaces import BackendName, Snapshot
from .common import snapshot_from_state, snapshot_summary

LOGGER = logging.getLogger("coolrl_lost_cities.games.classic.backends.python")


class PythonLostCitiesBackend:
    name: BackendName = "python"

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        self.config = config
        self.seed = seed
        self.state = GameState.new_game(config, seed=seed)
        self.history: list[GameState] = []
        LOGGER.debug("파이썬 백엔드 초기화: %s", snapshot_summary(self.snapshot()))

    def snapshot(self) -> Snapshot:
        return snapshot_from_state(self.state)

    def apply(self, action_id: int) -> None:
        before = self.snapshot()
        self.history.append(self.state.clone())
        self.state.apply_unified_action(action_id)
        LOGGER.debug(
            "파이썬 액션 적용: 액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            action_id,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.history),
        )

    def can_undo(self) -> bool:
        return bool(self.history)

    def undo(self) -> bool:
        if not self.history:
            LOGGER.debug("파이썬 되돌리기 무시: 기록이 비어 있음")
            return False
        before = self.snapshot()
        self.state = self.history.pop()
        LOGGER.debug(
            "파이썬 되돌리기: 이전={%s} 이후={%s} 되돌리기깊이=%s",
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.history),
        )
        return True
