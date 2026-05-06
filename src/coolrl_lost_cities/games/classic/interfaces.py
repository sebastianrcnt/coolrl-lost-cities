from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from .game import Card, GameState, LostCitiesConfig, score_expedition

BackendName = Literal["python", "rust"]


@dataclass
class Snapshot:
    config: LostCitiesConfig
    deck: list[Card]
    hands: list[list[Card]]
    expeditions: list[list[list[Card]]]
    discards: list[list[Card]]
    current_player: int
    phase: str
    pending_discarded_color: int | None
    turn_count: int
    terminal: bool
    legal_mask: list[bool]

    @property
    def card_action_size(self) -> int:
        return self.config.card_action_size

    @property
    def draw_action_size(self) -> int:
        return self.config.draw_action_size

    def expedition_score(self, player: int, color: int) -> int:
        return score_expedition(self.expeditions[player][color], self.config)

    def total_score(self, player: int) -> int:
        return sum(
            self.expedition_score(player, color)
            for color in range(self.config.n_colors)
        )

    def score_diff(self, player: int = 0) -> int:
        return self.total_score(player) - self.total_score(1 - player)


BotInput: TypeAlias = dict | GameState | Snapshot


@runtime_checkable
class LostCitiesBot(Protocol):
    def act(self, obs_or_state: BotInput) -> int:
        """Choose an action id from the current state or observation."""


@runtime_checkable
class LostCitiesBackend(Protocol):
    name: BackendName
    config: LostCitiesConfig
    seed: int | None

    def snapshot(self) -> Snapshot: ...

    def apply(self, action_id: int) -> None: ...

    def can_undo(self) -> bool: ...

    def undo(self) -> bool: ...
