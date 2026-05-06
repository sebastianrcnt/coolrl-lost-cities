from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

from ..game import Card, GameState, LostCitiesConfig
from ..interfaces import BotInput, LostCitiesBot
from .base import first_legal, legal_from_obs

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Lost Cities bots") from exc


PLAY_OR_DISCARD_ACTIONS_PER_SLOT = 2
DRAW_FROM_DECK_ACTION = 0


def play_action(slot: int) -> int:
    return PLAY_OR_DISCARD_ACTIONS_PER_SLOT * slot


def discard_action(slot: int) -> int:
    return PLAY_OR_DISCARD_ACTIONS_PER_SLOT * slot + 1


def draw_from_discard_action(color: int) -> int:
    return 1 + color


LOGGER = logging.getLogger("coolrl_lost_cities.games.classic.bots.safe_heuristic")


@dataclass(frozen=True)
class SafeHeuristicParams:
    # Expedition opening.
    open_target_ratio: float = 0.50
    open_min_card_ratio: float = 0.40

    # Handshake / investment behavior.
    handshake_target_multiplier: float = 1.15
    handshake_min_card_ratio: float = 0.34

    # Game phase thresholds.
    late_deck_ratio: float = 0.20
    mid_deck_ratio: float = 0.35

    # Evaluation weights.
    commitment_weight: float = 1.00
    gift_penalty_weight: float = 1.00
    discard_safety_bonus: float = 6.00
    unusable_discard_bonus: float = 20.00

    # Draw preferences.
    deck_draw_early_value: float = 2.00
    deck_draw_mid_value: float = 1.00
    deck_draw_late_value: float = -1.00
    deny_opponent_weight: float = 0.40
    winning_deck_bonus: float = 0.75
    losing_deck_penalty: float = 1.25
    losing_visible_draw_bonus: float = 1.50
    speculative_visible_draw_bonus: float = 1.50
    dead_visible_draw_penalty: float = 2.00
    unopened_draw_penalty_three_open: float = 10.00
    unopened_draw_penalty_four_open: float = 20.00
    strong_deny_threshold: float = 10.00

    # General behavior.
    late_open_block_ratio: float = 0.20
    low_card_sequence_bonus: float = 5.00
    started_expedition_play_bonus: float = 4.00
    started_expedition_followup_bonus: float = 3.00


@dataclass(frozen=True)
class DerivedHeuristicConfig:
    middle_rank: int
    max_color_sum: int
    break_even_sum: int
    open_target_sum: float
    min_open_cards: int
    min_handshake_numeric_cards: int
    late_deck_threshold: int
    mid_deck_threshold: int
    late_open_block_threshold: int
    bonus_possible: bool
    max_expedition_cards: int


@lru_cache(maxsize=64)
def derive_heuristic_config(
    config: LostCitiesConfig,
    params: SafeHeuristicParams,
) -> DerivedHeuristicConfig:
    max_color_sum = sum(
        config.min_rank + rank - 1
        for rank in range(1, config.n_ranks + 1)
    )
    break_even_sum = -config.expedition_penalty

    # In small tiers, break-even can be impossible. Do not set impossible targets.
    open_target_sum = min(
        0.8 * float(break_even_sum),
        params.open_target_ratio * float(max_color_sum),
    )

    min_open_cards = max(
        1,
        min(
            config.hand_size,
            round(config.hand_size * params.open_min_card_ratio),
        ),
    )

    min_handshake_numeric_cards = max(
        1,
        min(
            config.hand_size,
            round(config.hand_size * params.handshake_min_card_ratio),
        ),
    )

    late_deck_threshold = max(1, round(config.deck_size * params.late_deck_ratio))
    mid_deck_threshold = max(
        late_deck_threshold + 1,
        round(config.deck_size * params.mid_deck_ratio),
    )
    late_open_block_threshold = max(
        1,
        round(config.deck_size * params.late_open_block_ratio),
    )

    max_expedition_cards = config.n_handshakes + config.n_ranks

    return DerivedHeuristicConfig(
        middle_rank=(config.n_ranks + 1) // 2,
        max_color_sum=max_color_sum,
        break_even_sum=break_even_sum,
        open_target_sum=open_target_sum,
        min_open_cards=min_open_cards,
        min_handshake_numeric_cards=min_handshake_numeric_cards,
        late_deck_threshold=late_deck_threshold,
        mid_deck_threshold=mid_deck_threshold,
        late_open_block_threshold=late_open_block_threshold,
        bonus_possible=max_expedition_cards >= config.bonus_threshold,
        max_expedition_cards=max_expedition_cards,
    )


class SafeHeuristicBot(LostCitiesBot):
    def __init__(self, params: SafeHeuristicParams | None = None):
        self.params = params or SafeHeuristicParams()

    def act(self, obs_or_state: BotInput) -> int:
        if not isinstance(obs_or_state, GameState):
            LOGGER.debug(
                "SafeHeuristicBot fallback to first legal: input_type=%s",
                type(obs_or_state).__name__,
            )
            return first_legal(legal_from_obs(obs_or_state))

        LOGGER.debug(
            "SafeHeuristicBot heuristic path: player=%s phase=%s turn=%s",
            obs_or_state.current_player,
            obs_or_state.phase,
            obs_or_state.turn_count,
        )
        if obs_or_state.phase == "card":
            return self._act_card(obs_or_state)

        return self._act_draw(obs_or_state)

    def _act_card(self, state: GameState) -> int:
        player = state.current_player
        hand = state.hands[player]
        legal = state.legal_card_mask()
        derived = self._derived(state)
        deck_left = len(state.deck)

        handshake_action = self._best_handshake_play(
            state=state,
            player=player,
            hand=hand,
            legal=legal,
            derived=derived,
            deck_left=deck_left,
        )
        if handshake_action is not None:
            return handshake_action

        play_action_id = self._best_number_play(
            state=state,
            player=player,
            hand=hand,
            legal=legal,
            derived=derived,
            deck_left=deck_left,
        )
        if play_action_id is not None:
            return play_action_id

        if all(not expedition for expedition in state.expeditions[player]):
            forced_open_action = self._best_forced_open(
                state=state,
                player=player,
                hand=hand,
                legal=legal,
                derived=derived,
                deck_left=deck_left,
            )
            if forced_open_action is not None:
                return forced_open_action

        discard_action_id = self._best_discard(
            state=state,
            player=player,
            hand=hand,
            legal=legal,
            derived=derived,
        )
        if discard_action_id is not None:
            return discard_action_id

        return first_legal(legal)

    def _best_handshake_play(
        self,
        *,
        state: GameState,
        player: int,
        hand: list[Card],
        legal: list[bool] | np.ndarray,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> int | None:
        if state.config.n_handshakes <= 0:
            return None

        candidates: list[tuple[float, int]] = []

        for slot, card in enumerate(hand):
            action = play_action(slot)
            if not legal[action] or not card.is_handshake:
                continue

            color = card.color
            expedition = state.expeditions[player][color]

            # Investment cards should only be played before any number.
            if any(not played.is_handshake for played in expedition):
                continue

            playable_numbers = [
                other
                for other_slot, other in enumerate(hand)
                if (
                    other_slot != slot
                    and other.color == color
                    and not other.is_handshake
                    and state.can_play_card(player, other)
                )
            ]

            number_count = len(playable_numbers)
            number_sum = sum(self._num(state, other) for other in playable_numbers)

            if number_count < derived.min_handshake_numeric_cards:
                continue

            # Handshakes need a stronger support than normal opening.
            required_sum = (
                derived.open_target_sum * self.params.handshake_target_multiplier
            )
            if number_sum < required_sum:
                continue

            if deck_left <= derived.late_open_block_threshold:
                continue

            value = 0.0
            value += number_sum
            value += 2.0 * number_count
            value += self._bonus_potential(
                state=state,
                player=player,
                color=color,
                extra_cards=0,
                derived=derived,
                committed_cards=1,
                exclude_card=card,
            )
            value -= self._late_penalty(derived, deck_left)

            candidates.append((value, action))

        if not candidates:
            return None

        return max(candidates)[1]

    def _best_number_play(
        self,
        *,
        state: GameState,
        player: int,
        hand: list[Card],
        legal: list[bool] | np.ndarray,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> int | None:
        candidates: list[tuple[float, int]] = []

        for slot, card in enumerate(hand):
            action = play_action(slot)
            if not legal[action] or card.is_handshake:
                continue

            color = card.color
            expedition_started = len(state.expeditions[player][color]) > 0

            if expedition_started:
                value = self._started_expedition_play_value(
                    state=state,
                    player=player,
                    card=card,
                    derived=derived,
                    deck_left=deck_left,
                )
                candidates.append((value, action))
                continue

            if self._should_open_expedition(
                state=state,
                player=player,
                color=color,
                opening_card=card,
                derived=derived,
                deck_left=deck_left,
            ):
                value = self._open_expedition_value(
                    state=state,
                    player=player,
                    color=color,
                    opening_card=card,
                    derived=derived,
                    deck_left=deck_left,
                )
                candidates.append((value, action))

        if not candidates:
            return None

        return max(candidates)[1]

    def _started_expedition_play_value(
        self,
        *,
        state: GameState,
        player: int,
        card: Card,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> float:
        color = card.color
        expedition = state.expeditions[player][color]
        numeric_value = self._num(state, card)

        current_sum = sum(
            self._num(state, played)
            for played in expedition
            if not played.is_handshake
        )

        followups = [
            followup
            for followup in state.hands[player]
            if (
                followup is not card
                and followup.color == color
                and not followup.is_handshake
                and followup.rank > card.rank
            )
        ]

        projected_sum = current_sum + numeric_value + sum(
            self._num(state, followup) for followup in followups
        )

        value = 0.0
        value += self.params.started_expedition_play_bonus
        value += self.params.started_expedition_followup_bonus

        # Early/mid game: preserve sequencing by playing lower legal cards first.
        value += float(state.config.max_rank + 1 - numeric_value)

        # Late game: cash out larger cards more aggressively.
        if deck_left <= derived.late_deck_threshold:
            value += 2.0 * numeric_value
        elif deck_left <= derived.mid_deck_threshold:
            value += 0.8 * numeric_value

        # Avoid extending hopeless expeditions unless the game is late.
        if projected_sum < derived.open_target_sum:
            value -= 6.0

        # If investments are already committed, numbers become more urgent.
        handshakes = sum(1 for played in expedition if played.is_handshake)
        value += 3.0 * handshakes

        value += self._bonus_potential(
            state=state,
            player=player,
            color=color,
            extra_cards=0,
            derived=derived,
            committed_cards=1,
            exclude_card=card,
        )

        return value

    def _should_open_expedition(
        self,
        *,
        state: GameState,
        player: int,
        color: int,
        opening_card: Card,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> bool:
        if deck_left <= derived.late_open_block_threshold:
            return False

        return self._opening_plan_value(
            state=state,
            player=player,
            color=color,
            opening_card=opening_card,
            derived=derived,
            deck_left=deck_left,
        ) > 0.0

    def _opening_plan_value(
        self,
        *,
        state: GameState,
        player: int,
        color: int,
        opening_card: Card,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> float:
        numbers = [
            card
            for card in state.hands[player]
            if (
                card.color == color
                and not card.is_handshake
                and card.rank >= opening_card.rank
            )
        ]
        handshakes = [
            card
            for card in state.hands[player]
            if card.color == color and card.is_handshake
        ]
        opened_colors = sum(1 for expedition in state.expeditions[player] if expedition)
        number_sum = sum(self._num(state, card) for card in numbers)
        high_cards = [card for card in numbers if card.rank >= derived.middle_rank]
        high_count = len(high_cards)
        opening_value = self._num(state, opening_card)
        new_color_penalty = self._new_color_open_penalty(opened_colors)

        strong_open = (
            len(numbers) >= derived.min_open_cards
            and number_sum >= derived.open_target_sum
            and (high_cards or number_sum >= 0.85 * derived.max_color_sum)
        )
        speculative_open = (
            opened_colors <= 2
            and
            len(numbers) >= 2
            and number_sum >= 0.65 * derived.open_target_sum
            and bool(high_cards)
        )
        single_late_open = (
            deck_left <= derived.mid_deck_threshold
            and len(numbers) >= 1
            and opening_value >= 8
        )
        exceptional_open = (
            len(numbers) >= derived.min_open_cards + 1
            and number_sum >= max(float(derived.break_even_sum), derived.open_target_sum * 1.4)
            and high_count >= 2
            and deck_left > derived.mid_deck_threshold
        )

        if opened_colors == 3:
            speculative_open = False
        if opened_colors >= 4:
            strong_open = False
            speculative_open = False
            single_late_open = False

        if strong_open:
            return (
                6.0
                + 0.25 * number_sum
                + 0.8 * len(numbers)
                + 0.5 * len(handshakes)
                - new_color_penalty
            )
        if speculative_open:
            return (
                3.0
                + 0.18 * number_sum
                + 0.7 * len(numbers)
                + 0.4 * len(handshakes)
                - new_color_penalty
            )
        if opened_colors == 3:
            return 0.0
        if exceptional_open:
            return (
                10.0
                + 0.3 * number_sum
                + 1.0 * len(numbers)
                + 0.7 * high_count
                - new_color_penalty
            )
        if single_late_open:
            return 1.5 + 0.2 * opening_value - new_color_penalty

        return 0.0

    def _open_expedition_value(
        self,
        *,
        state: GameState,
        player: int,
        color: int,
        opening_card: Card,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> float:
        numbers = [
            card
            for card in state.hands[player]
            if (
                card.color == color
                and not card.is_handshake
                and card.rank >= opening_card.rank
            )
        ]
        handshakes = [
            card
            for card in state.hands[player]
            if card.color == color and card.is_handshake
        ]

        number_sum = sum(self._num(state, card) for card in numbers)

        value = 0.0
        value += number_sum
        value += 2.0 * len(numbers)
        value += 1.5 * len(handshakes)
        value += self._opening_plan_value(
            state=state,
            player=player,
            color=color,
            opening_card=opening_card,
            derived=derived,
            deck_left=deck_left,
        )

        # Opening an expedition accepts the penalty.
        value += state.config.expedition_penalty

        # Low opening cards preserve sequencing.
        value += max(0.0, float(derived.middle_rank - opening_card.rank))

        value += self._bonus_potential(
            state=state,
            player=player,
            color=color,
            extra_cards=0,
            derived=derived,
            committed_cards=1,
            exclude_card=opening_card,
        )

        value -= self._late_penalty(derived, deck_left)

        return value

    def _best_forced_open(
        self,
        *,
        state: GameState,
        player: int,
        hand: list[Card],
        legal: list[bool] | np.ndarray,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> int | None:
        candidates: list[tuple[float, int]] = []

        for slot, card in enumerate(hand):
            action = play_action(slot)
            if not legal[action] or card.is_handshake:
                continue
            if state.expeditions[player][card.color]:
                continue

            opening_value = self._opening_plan_value(
                state=state,
                player=player,
                color=card.color,
                opening_card=card,
                derived=derived,
                deck_left=deck_left,
            )
            color_numbers = [
                other
                for other in hand
                if other.color == card.color and not other.is_handshake
            ]
            number_sum = sum(self._num(state, other) for other in color_numbers)

            if (
                opening_value <= 0.0
                and len(color_numbers) < 2
                and number_sum < 0.5 * derived.open_target_sum
                and deck_left > derived.mid_deck_threshold
            ):
                continue

            forced_value = opening_value
            forced_value += 0.2 * number_sum
            forced_value += float(state.config.max_rank + 1 - self._num(state, card))
            candidates.append((forced_value, action))

        if not candidates:
            return None

        return max(candidates)[1]

    def _best_discard(
        self,
        *,
        state: GameState,
        player: int,
        hand: list[Card],
        legal: list[bool] | np.ndarray,
        derived: DerivedHeuristicConfig,
    ) -> int | None:
        candidates: list[tuple[float, int]] = []
        opponent = 1 - player

        for slot, card in enumerate(hand):
            action = discard_action(slot)
            if not legal[action]:
                continue

            my_value = self._card_value_for_me(
                state=state,
                player=player,
                card=card,
                derived=derived,
            )
            opponent_value = self._card_value_for_opponent(
                state=state,
                opponent=opponent,
                card=card,
                derived=derived,
            )

            # Higher means better to discard.
            score = 0.0
            score -= my_value
            score -= self.params.gift_penalty_weight * opponent_value

            if not state.can_play_card(player, card):
                score += self.params.unusable_discard_bonus

            if not state.can_play_card(opponent, card):
                score += self.params.discard_safety_bonus

            # Handshakes are swingy. Prefer not discarding them unless they are dead.
            if card.is_handshake and state.can_play_card(player, card):
                score -= 4.0

            candidates.append((score, action))

        if not candidates:
            return None

        return max(candidates)[1]

    def _act_draw(self, state: GameState) -> int:
        legal = state.legal_draw_mask()
        player = state.current_player
        derived = self._derived(state)

        candidates: list[tuple[float, int, int]] = []

        if legal[DRAW_FROM_DECK_ACTION]:
            candidates.append(
                (
                    self._deck_draw_value(state, derived),
                    1,
                    DRAW_FROM_DECK_ACTION,
                )
            )

        for color in range(state.config.n_colors):
            action = draw_from_discard_action(color)
            if not legal[action] or not state.discards[color]:
                continue

            card = state.discards[color][-1]
            value = self._visible_draw_value(
                state=state,
                player=player,
                card=card,
                derived=derived,
            )
            candidates.append((value, 0, action))

        if candidates:
            return max(candidates)[2]

        return first_legal(legal)

    def _visible_draw_value(
        self,
        *,
        state: GameState,
        player: int,
        card: Card,
        derived: DerivedHeuristicConfig,
    ) -> float:
        color = card.color
        opponent = 1 - player
        opened_colors = sum(1 for expedition in state.expeditions[player] if expedition)
        is_unopened_color = not state.expeditions[player][color]
        commitment = self._color_commitment(
            state=state,
            player=player,
            color=color,
            derived=derived,
        )
        opponent_value = self._card_value_for_opponent(
            state=state,
            opponent=opponent,
            card=card,
            derived=derived,
        )
        score_diff = state.score_diff(player)

        value = self.params.deny_opponent_weight * opponent_value
        if score_diff <= 0:
            value += self.params.losing_visible_draw_bonus

        exceptional_support = False
        if is_unopened_color:
            if opened_colors >= 4:
                value -= self.params.unopened_draw_penalty_four_open
            elif opened_colors >= 3:
                value -= self.params.unopened_draw_penalty_three_open

        if card.is_handshake:
            if state.has_numeric(player, color):
                return value - self.params.dead_visible_draw_penalty
            if not state.expeditions[player][color]:
                playable_numbers = [
                    other
                    for other in state.hands[player]
                    if (
                        other.color == color
                        and not other.is_handshake
                        and state.can_play_card(player, other)
                    )
                ]
                number_sum = sum(self._num(state, other) for other in playable_numbers)
                required_sum = (
                    derived.open_target_sum * self.params.handshake_target_multiplier
                )
                if (
                    len(playable_numbers) < derived.min_handshake_numeric_cards
                    or number_sum < required_sum
                ):
                    support = self._visible_open_support_value(
                        state=state,
                        player=player,
                        card=card,
                        derived=derived,
                    )
                    exceptional_support = support >= 6.0
                    if (
                        is_unopened_color
                        and opened_colors >= 4
                        and not exceptional_support
                        and opponent_value < self.params.strong_deny_threshold
                        and score_diff > -15
                    ):
                        return -8.0
                    return value + support - 0.5
            return value + 6.0 + commitment

        immediate_playable = state.can_play_card(player, card)
        if immediate_playable:
            value += float(self._num(state, card))
            value += 0.7 * commitment

            if state.expeditions[player][color]:
                value += 5.0
            else:
                support = self._visible_open_support_value(
                    state=state,
                    player=player,
                    card=card,
                    derived=derived,
                )
                exceptional_support = support >= 6.0
                value += support
        else:
            value -= self.params.dead_visible_draw_penalty
            if not state.expeditions[player][color]:
                support = self._visible_open_support_value(
                    state=state,
                    player=player,
                    card=card,
                    derived=derived,
                )
                exceptional_support = support >= 6.0
                value += support

        if (
            is_unopened_color
            and opened_colors >= 4
            and not exceptional_support
            and opponent_value < self.params.strong_deny_threshold
            and score_diff > -15
        ):
            return -8.0

        value += self._bonus_potential(
            state=state,
            player=player,
            color=color,
            extra_cards=1,
            derived=derived,
        )

        return value

    def _visible_open_support_value(
        self,
        *,
        state: GameState,
        player: int,
        card: Card,
        derived: DerivedHeuristicConfig,
    ) -> float:
        color = card.color
        opened_colors = sum(1 for expedition in state.expeditions[player] if expedition)
        same_color_numbers = [
            other
            for other in state.hands[player]
            if other.color == color and not other.is_handshake
        ]
        same_color_handshakes = [
            other
            for other in state.hands[player]
            if other.color == color and other.is_handshake
        ]
        future_numbers = [
            other
            for other in same_color_numbers
            if other is not card and other.rank >= card.rank
        ]

        value = 0.0
        value += 0.8 * len(future_numbers)
        value += 1.0 * len(same_color_handshakes)

        if card.rank <= derived.middle_rank:
            value += self.params.speculative_visible_draw_bonus

        if self._visible_number_can_help_open(
            state=state,
            player=player,
            card=card,
            derived=derived,
        ):
            value += 4.0
        elif opened_colors <= 2 and (future_numbers or same_color_handshakes):
            value += self.params.speculative_visible_draw_bonus

        if opened_colors <= 2:
            value += 0.25 * self._opening_plan_value(
                state=state,
                player=player,
                color=color,
                opening_card=card,
                derived=derived,
                deck_left=len(state.deck),
            )
        elif opened_colors == 3:
            value += 0.1 * max(
                0.0,
                self._opening_plan_value(
                    state=state,
                    player=player,
                    color=color,
                    opening_card=card,
                    derived=derived,
                    deck_left=len(state.deck),
                ),
            )

        return value

    def _visible_number_can_help_open(
        self,
        *,
        state: GameState,
        player: int,
        card: Card,
        derived: DerivedHeuristicConfig,
    ) -> bool:
        numbers = [
            other
            for other in state.hands[player]
            if (
                other.color == card.color
                and not other.is_handshake
                and other.rank >= card.rank
            )
        ]
        numbers.append(card)

        if len(numbers) < derived.min_open_cards:
            return False

        number_sum = sum(self._num(state, other) for other in numbers)
        if number_sum < derived.open_target_sum:
            return False

        return any(other.rank >= derived.middle_rank for other in numbers)

    def _deck_draw_value(
        self,
        state: GameState,
        derived: DerivedHeuristicConfig,
    ) -> float:
        deck_left = len(state.deck)
        score_diff = state.score_diff(state.current_player)

        if deck_left > derived.mid_deck_threshold:
            value = self.params.deck_draw_early_value
        elif deck_left > derived.late_deck_threshold:
            value = self.params.deck_draw_mid_value
        else:
            value = self.params.deck_draw_late_value

        if score_diff > 0:
            value += self.params.winning_deck_bonus
        else:
            value -= self.params.losing_deck_penalty

        return value

    def _card_value_for_me(
        self,
        *,
        state: GameState,
        player: int,
        card: Card,
        derived: DerivedHeuristicConfig,
    ) -> float:
        if not state.can_play_card(player, card):
            return 0.0

        commitment = self._color_commitment(
            state=state,
            player=player,
            color=card.color,
            derived=derived,
        )

        if card.is_handshake:
            return 7.0 + 1.2 * commitment

        numeric_value = self._num(state, card)

        value = 0.0
        value += 0.8 * numeric_value
        value += self.params.commitment_weight * commitment
        if state.expeditions[player][card.color]:
            value += self.params.started_expedition_play_bonus
            value += self.params.started_expedition_followup_bonus

        # Low playable cards are valuable when we are committed to that color.
        if commitment >= 6.0 and card.rank <= derived.middle_rank:
            value += self.params.low_card_sequence_bonus

        return value

    def _new_color_open_penalty(self, opened_colors: int) -> float:
        if opened_colors <= 1:
            return 0.0
        if opened_colors == 2:
            return 6.0
        if opened_colors == 3:
            return 14.0
        return 28.0

    def _card_value_for_opponent(
        self,
        *,
        state: GameState,
        opponent: int,
        card: Card,
        derived: DerivedHeuristicConfig,
    ) -> float:
        if not state.can_play_card(opponent, card):
            return 0.0

        interest = self._public_color_commitment_for_opponent(
            state=state,
            opponent=opponent,
            color=card.color,
            derived=derived,
        )

        if card.is_handshake:
            return 8.0 + 1.5 * interest

        numeric_value = self._num(state, card)
        return numeric_value * (0.4 + 0.25 * interest)

    def _color_commitment(
        self,
        *,
        state: GameState,
        player: int,
        color: int,
        derived: DerivedHeuristicConfig,
    ) -> float:
        expedition = state.expeditions[player][color]
        hand = state.hands[player]

        value = 0.0

        if expedition:
            value += 5.0

        for card in expedition:
            if card.is_handshake:
                value += 2.0
            else:
                value += 0.25 * self._num(state, card)

        playable_cards = [
            card
            for card in hand
            if card.color == color and state.can_play_card(player, card)
        ]

        playable_numbers = [card for card in playable_cards if not card.is_handshake]
        playable_handshakes = [card for card in playable_cards if card.is_handshake]

        value += 1.2 * len(playable_numbers)
        value += 1.5 * len(playable_handshakes)
        value += 0.15 * sum(self._num(state, card) for card in playable_numbers)

        # Bonus chance means the color is strategically more interesting.
        value += 0.05 * self._bonus_potential(
            state=state,
            player=player,
            color=color,
            extra_cards=0,
            derived=derived,
        )

        return value

    def _public_color_commitment_for_opponent(
        self,
        *,
        state: GameState,
        opponent: int,
        color: int,
        derived: DerivedHeuristicConfig,
    ) -> float:
        expedition = state.expeditions[opponent][color]
        discard = state.discards[color]

        value = 0.0

        if expedition:
            value += 5.0

        handshake_count = sum(1 for card in expedition if card.is_handshake)
        value += 2.0 * handshake_count

        numeric_cards = [card for card in expedition if not card.is_handshake]
        value += 0.25 * sum(self._num(state, card) for card in numeric_cards)

        if numeric_cards:
            value += 0.4 * self._num(state, numeric_cards[-1])

        if discard:
            top_card = discard[-1]
            if state.can_play_card(opponent, top_card):
                if top_card.is_handshake:
                    value += 1.5
                else:
                    value += 1.0 + 0.1 * self._num(state, top_card)

        if derived.bonus_possible:
            expedition_len = len(expedition)
            if expedition_len + 1 >= state.config.bonus_threshold:
                value += 0.2 * float(state.config.bonus_amount)

        return value

    def _playable_followup_numbers(
        self,
        state: GameState,
        player: int,
        color: int,
    ) -> list[Card]:
        return [
            card
            for card in state.hands[player]
            if (
                card.color == color
                and not card.is_handshake
                and state.can_play_card(player, card)
            )
        ]

    def _bonus_potential(
        self,
        *,
        state: GameState,
        player: int,
        color: int,
        extra_cards: int,
        derived: DerivedHeuristicConfig,
        committed_cards: int = 0,
        exclude_card: Card | None = None,
    ) -> float:
        if not derived.bonus_possible:
            return 0.0

        expedition_len = len(state.expeditions[player][color]) + committed_cards
        need = state.config.bonus_threshold - expedition_len

        if need <= 0:
            return float(state.config.bonus_amount)

        playable_count = sum(
            1
            for card in state.hands[player]
            if (
                card is not exclude_card
                and card.color == color
                and state.can_play_card(player, card)
            )
        )

        if playable_count + extra_cards >= need:
            return 0.4 * float(state.config.bonus_amount)

        return 0.0

    def _late_penalty(
        self,
        derived: DerivedHeuristicConfig,
        deck_left: int,
    ) -> float:
        if deck_left <= derived.late_deck_threshold:
            return 15.0
        if deck_left <= derived.mid_deck_threshold:
            return 8.0
        return 0.0

    def _num(self, state: GameState, card: Card) -> int:
        return card.numeric_value(state.config.min_rank)

    def _derived(self, state: GameState) -> DerivedHeuristicConfig:
        return derive_heuristic_config(state.config, self.params)
