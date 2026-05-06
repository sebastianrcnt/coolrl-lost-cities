from __future__ import annotations

from .game import GameState, IllegalMoveError, LostCitiesConfig

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for LostCitiesEnv") from exc


class LostCitiesEnv:
    def __init__(self, config: LostCitiesConfig | None = None):
        self.config = config or LostCitiesConfig()
        self.state: GameState | None = None

    def reset(self) -> dict:
        self.state = GameState.new_game(self.config)
        return self._obs()

    def step(self, action_id: int) -> tuple[dict, float, bool, dict]:
        if self.state is None:
            self.reset()
        assert self.state is not None
        actor = self.state.current_player
        self.state.apply_unified_action(self._normalize_action_id(action_id))
        done = self.state.terminal
        reward = float(self.state.score_diff(actor)) if done else 0.0
        return self._obs(), reward, done, {}

    def legal_actions(self) -> np.ndarray:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return np.asarray(self.state.unified_legal_mask(), dtype=bool)

    @property
    def current_player(self) -> int:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return self.state.current_player

    @property
    def phase(self) -> str:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return self.state.phase

    def _obs(self) -> dict:
        assert self.state is not None
        return {
            "spatial": np.zeros((0,), dtype=np.float32),
            "scalar": np.zeros((0,), dtype=np.float32),
            "legal_mask": np.asarray(self.state.unified_legal_mask(), dtype=bool),
            "phase": 0 if self.state.phase == "card" else 1,
            "player": self.state.current_player,
        }

    def _normalize_action_id(self, action_id: int) -> int:
        assert self.state is not None
        unified_mask = self.state.unified_legal_mask()
        if 0 <= action_id < len(unified_mask) and unified_mask[action_id]:
            return action_id

        local_mask = self.state.legal_mask()
        if 0 <= action_id < len(local_mask) and local_mask[action_id]:
            return self.state.to_unified_action(action_id)

        raise IllegalMoveError(
            f"illegal action {action_id} in phase {self.state.phase} "
            f"for player {self.state.current_player}"
        )
