from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .game import LostCitiesConfig


@dataclass(frozen=True, order=True)
class ReferenceLostCitiesCard:
    color: int
    rank: int


class ReferenceLostCitiesState:
    """Pure Python reference implementation placeholder."""

    def __init__(self, config: LostCitiesConfig | None = None, **_: Any) -> None:
        self.config = config or LostCitiesConfig()

    @classmethod
    def new_game(
        cls,
        config: LostCitiesConfig | None = None,
        *,
        seed: int | None = None,
    ) -> ReferenceLostCitiesState:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    @classmethod
    def new_game_from_deck(
        cls,
        deck: list[ReferenceLostCitiesCard],
        config: LostCitiesConfig | None = None,
    ) -> ReferenceLostCitiesState:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    @classmethod
    def empty(cls, config: LostCitiesConfig | None = None) -> ReferenceLostCitiesState:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def to_snapshot(self) -> dict[str, Any]:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        *,
        validate: bool = True,
    ) -> ReferenceLostCitiesState:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def legal_mask(self) -> list[bool]:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def unified_legal_mask(self) -> list[bool]:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def apply_action(self, action_id: int) -> None:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def apply_unified_action(self, action_id: int) -> None:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def total_score(self, player: int) -> int:
        raise NotImplementedError("pure Python reference engine is not implemented yet")

    def score_diff(self, player: int = 0) -> int:
        raise NotImplementedError("pure Python reference engine is not implemented yet")
