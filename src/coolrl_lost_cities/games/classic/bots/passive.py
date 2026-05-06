from __future__ import annotations

from ..game import GameState
from ..interfaces import BotInput, Snapshot
from .base import first_legal, legal_from_obs


class PassiveDiscardBot:
    """Baseline that avoids opening expeditions whenever discarding is legal."""

    def act(self, obs_or_state: BotInput) -> int:
        if isinstance(obs_or_state, GameState):
            return self._act_phase_local(
                obs_or_state.phase,
                obs_or_state.legal_mask(),
                obs_or_state.card_action_size,
            )
        if isinstance(obs_or_state, Snapshot):
            return self._act_unified(
                obs_or_state.phase,
                obs_or_state.legal_mask,
                obs_or_state.card_action_size,
            )
        legal = legal_from_obs(obs_or_state)
        return first_legal(legal)

    def _act_phase_local(
        self,
        phase: str,
        legal: list[bool],
        card_action_size: int,
    ) -> int:
        if phase == "card":
            for action in range(1, min(card_action_size, len(legal)), 2):
                if legal[action]:
                    return action
        if phase == "draw" and legal and legal[0]:
            return 0
        return first_legal(legal)

    def _act_unified(
        self,
        phase: str,
        legal: list[bool],
        card_action_size: int,
    ) -> int:
        if phase == "card":
            for action in range(1, min(card_action_size, len(legal)), 2):
                if legal[action]:
                    return action
        deck_action = card_action_size
        if phase == "draw" and deck_action < len(legal) and legal[deck_action]:
            return deck_action
        return first_legal(legal)
