from __future__ import annotations

import json
from collections import Counter

from coolrl_lost_cities.games.classic.game import Card, GameState, build_deck


def _card_tuple(card: Card) -> tuple[int, int]:
    return int(card.color), int(card.rank)


def _sorted_cards(cards: list[Card]) -> list[tuple[int, int]]:
    return sorted(_card_tuple(card) for card in cards)


def canonical_info_set_key(state: GameState, player: int) -> bytes:
    """Deterministic key for observable information from ``player``'s POV."""
    p = int(player)
    payload = {
        "config": state.config.to_snapshot(),
        "player": p,
        "current_player": int(state.current_player),
        "phase": state.phase,
        "pending_discarded_color": (
            None if state.pending_discarded_color < 0 else int(state.pending_discarded_color)
        ),
        "turn_count": int(state.turn_count),
        "terminal": bool(state.terminal),
        "deck_size": len(state.deck),
        "hand": _sorted_cards(state.hands[p]),
        "hand_size_opp": len(state.hands[1 - p]),
        "expeditions": [
            [[_card_tuple(card) for card in expedition] for expedition in player_expeditions]
            for player_expeditions in state.expeditions
        ],
        "discards": [[_card_tuple(card) for card in discard] for discard in state.discards],
        "legal_mask": list(map(bool, state.unified_legal_mask())),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def visible_cards(state: GameState, player: int) -> list[Card]:
    cards: list[Card] = []
    cards.extend(state.hands[int(player)])
    for player_expeditions in state.expeditions:
        for expedition in player_expeditions:
            cards.extend(expedition)
    for discard in state.discards:
        cards.extend(discard)
    return cards


def unseen_cards(state: GameState, player: int) -> list[Card]:
    remaining = Counter(_card_tuple(card) for card in build_deck(state.config))
    for card in visible_cards(state, player):
        remaining[_card_tuple(card)] -= 1
    cards: list[Card] = []
    for (color, rank), count in remaining.items():
        cards.extend(Card(color, rank) for _ in range(count))
    return cards
