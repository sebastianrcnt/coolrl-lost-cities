from __future__ import annotations

from typing import Any

from ..game import Card, GameState, LostCitiesConfig
from ..interfaces import Snapshot


def snapshot_summary(snapshot: Snapshot) -> str:
    scores = [snapshot.total_score(0), snapshot.total_score(1)]
    hand_sizes = [len(hand) for hand in snapshot.hands]
    discard_sizes = [len(discard) for discard in snapshot.discards]
    phase = "카드" if snapshot.phase == "card" else "뽑기"
    return (
        f"플레이어={snapshot.current_player} 단계={phase} "
        f"턴={snapshot.turn_count} 종료={snapshot.terminal} "
        f"덱={len(snapshot.deck)} 손패수={hand_sizes} 점수={scores} "
        f"직전버린색={snapshot.pending_discarded_color} "
        f"버린더미수={discard_sizes}"
    )


def snapshot_from_state(state: GameState) -> Snapshot:
    return Snapshot(
        config=state.config,
        deck=list(state.deck),
        hands=[list(hand) for hand in state.hands],
        expeditions=[
            [list(expedition) for expedition in player_expeditions]
            for player_expeditions in state.expeditions
        ],
        discards=[list(discard) for discard in state.discards],
        current_player=state.current_player,
        phase=state.phase,
        pending_discarded_color=state.pending_discarded_color,
        turn_count=state.turn_count,
        terminal=state.terminal,
        legal_mask=state.unified_legal_mask(),
    )


def snapshot_from_trace(config_data: dict[str, Any], step: dict[str, Any]) -> Snapshot:
    config = LostCitiesConfig(**config_data)
    return Snapshot(
        config=config,
        deck=cards_from_json(step["deck"]),
        hands=[cards_from_json(hand) for hand in step["hands"]],
        expeditions=[
            [cards_from_json(expedition) for expedition in player_expeditions]
            for player_expeditions in step["expeditions"]
        ],
        discards=[cards_from_json(discard) for discard in step["discards"]],
        current_player=int(step["current_player"]),
        phase=str(step["phase"]),
        pending_discarded_color=step.get("pending_discarded_color"),
        turn_count=int(step["turn_count"]),
        terminal=bool(step["terminal"]),
        legal_mask=list(step["legal_mask"]),
    )


def cards_from_json(cards: list[dict[str, int]]) -> list[Card]:
    return [Card.from_snapshot(card) for card in cards]
