from __future__ import annotations

from coolrl_lost_cities.games.classic.game import Card, GameState, LostCitiesConfig


def make_state(
    config: LostCitiesConfig | None = None,
    *,
    deck: list[Card] | None = None,
    hands: list[list[Card]] | None = None,
    expeditions: list[list[list[Card]]] | None = None,
    discards: list[list[Card]] | None = None,
    current_player: int = 0,
    phase: str = "card",
    pending_discarded_color: int | None = None,
    turn_count: int = 0,
    terminal: bool = False,
    validate: bool = False,
) -> GameState:
    config = config or LostCitiesConfig()
    return GameState.from_snapshot(
        {
            "config": config.to_snapshot(),
            "deck": deck or [],
            "hands": hands or [[], []],
            "expeditions": expeditions
            or [
                [[] for _ in range(config.n_colors)],
                [[] for _ in range(config.n_colors)],
            ],
            "discards": discards or [[] for _ in range(config.n_colors)],
            "current_player": current_player,
            "phase": phase,
            "pending_discarded_color": pending_discarded_color,
            "turn_count": turn_count,
            "terminal": terminal,
        },
        validate=validate,
    )
