from __future__ import annotations

import struct
from collections import Counter

from coolrl_lost_cities.games.classic.game import Card, GameState, build_deck


def _card_tuple(card: Card) -> tuple[int, int]:
    return int(card.color), int(card.rank)


def _sorted_cards(cards: list[Card]) -> list[tuple[int, int]]:
    return sorted(_card_tuple(card) for card in cards)


# Phase encoding: "card" -> 0, "draw" -> 1, anything else -> 2.
_PHASE_TO_INT = {"card": 0, "draw": 1}


def canonical_info_set_key(state: GameState, player: int) -> bytes:
    """Deterministic key for observable information from ``player``'s POV.

    Packed binary representation (big-endian) covering the same fields as
    the previous JSON encoding. Faster to compute and produces a more
    compact key while remaining a stable, hashable ``bytes`` value.
    """
    p = int(player)
    cfg = state.config
    parts: list[bytes] = []
    # Header: rule constants that pin down the action/observation shape.
    parts.append(
        struct.pack(
            ">BBBBBhhBBBBB",
            int(cfg.n_colors) & 0xFF,
            int(cfg.n_ranks) & 0xFF,
            int(cfg.min_rank) & 0xFF,
            int(cfg.n_handshakes) & 0xFF,
            int(cfg.hand_size) & 0xFF,
            int(cfg.expedition_penalty),
            int(cfg.bonus_amount),
            int(cfg.bonus_threshold) & 0xFF,
            p & 0xFF,
            int(state.current_player) & 0xFF,
            _PHASE_TO_INT.get(state.phase, 2) & 0xFF,
            (1 if state.terminal else 0) & 0xFF,
        )
    )
    # Variable scalars.
    pending_color = -1 if state.pending_discarded_color < 0 else int(state.pending_discarded_color)
    parts.append(
        struct.pack(
            ">bHHH",
            pending_color,
            int(state.turn_count) & 0xFFFF,
            len(state.deck) & 0xFFFF,
            len(state.hands[1 - p]) & 0xFFFF,
        )
    )
    # Sorted hand for the POV player. Cards encoded as (color, rank).
    hand = state.hands[p]
    parts.append(struct.pack(">H", len(hand)))
    if hand:
        sorted_pairs = sorted((int(c.color), int(c.rank)) for c in hand)
        parts.append(b"".join(struct.pack(">BB", c, r) for c, r in sorted_pairs))
    # Expeditions per player/color (ordered, since order matters for legality).
    expeditions = state.expeditions
    for player_expeditions in expeditions:
        for expedition in player_expeditions:
            parts.append(struct.pack(">H", len(expedition)))
            if expedition:
                parts.append(
                    b"".join(struct.pack(">BB", int(c.color), int(c.rank)) for c in expedition)
                )
    # Discards per color.
    for discard in state.discards:
        parts.append(struct.pack(">H", len(discard)))
        if discard:
            parts.append(b"".join(struct.pack(">BB", int(c.color), int(c.rank)) for c in discard))
    # Legal mask (packed as raw bytes from the underlying list).
    mask = state.unified_legal_mask()
    parts.append(bytes(1 if bool(b) else 0 for b in mask))
    return b"".join(parts)


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
