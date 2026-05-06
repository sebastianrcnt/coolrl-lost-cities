from __future__ import annotations

from dataclasses import dataclass

from .game import Card, GameState, LostCitiesConfig, score_expedition


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
        return sum(self.expedition_score(player, color) for color in range(self.config.n_colors))

    def score_diff(self, player: int = 0) -> int:
        return self.total_score(player) - self.total_score(1 - player)


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
