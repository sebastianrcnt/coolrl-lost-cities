from __future__ import annotations

import random

from coolrl_lost_cities.games.classic.game import GameState

from .info_set import unseen_cards


def sample_determinization(state: GameState, player: int, rng: random.Random) -> GameState:
    """Sample a concrete state uniformly from ``player``'s current information set."""
    if hasattr(state, "determinize_for_player"):
        return state.determinize_for_player(int(player), rng)
    p = int(player)
    opponent = 1 - p
    snapshot = state.to_snapshot()
    unseen = unseen_cards(state, p)
    rng.shuffle(unseen)
    opponent_hand_size = len(state.hands[opponent])
    deck_size = len(state.deck)
    if len(unseen) != opponent_hand_size + deck_size:
        raise ValueError(
            "information set card count mismatch: "
            f"unseen={len(unseen)} opponent_hand={opponent_hand_size} deck={deck_size}"
        )
    snapshot["hands"][opponent] = [card.to_snapshot() for card in unseen[:opponent_hand_size]]
    snapshot["deck"] = [card.to_snapshot() for card in unseen[opponent_hand_size:]]
    det = GameState.from_snapshot(snapshot)
    det.validate_invariants()
    return det
