# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Deprecated compatibility rules engine for Lost Cities classic.

This module remains the public engine while the replacement fast engine is
developed under ``coolrl_lost_cities.games.classic.engines``. New traversal,
simulation, and training work should target the fast engine once it exists.
"""

from collections import Counter
from dataclasses import dataclass, fields
import random
from typing import Any, Literal

import numpy as np

cimport cython


Phase = Literal["card", "draw"]
DEPRECATED_ENGINE = True


class IllegalMoveError(ValueError):
    """Raised when an action id is not legal for the current state."""


@cython.freelist(256)
cdef class Card:
    cdef readonly int color
    cdef readonly int rank

    def __cinit__(self, color, rank):
        self.color = int(color)
        self.rank = int(rank)

    @property
    def is_handshake(self):
        return self.rank == 0

    cpdef int numeric_value(self, int min_rank):
        if self.rank == 0:
            return 0
        return min_rank + self.rank - 1

    def label(self, int min_rank):
        if self.rank == 0:
            return f"[{self.color}]H"
        return f"[{self.color}]{self.numeric_value(min_rank)}"

    def to_snapshot(self):
        return {"color": self.color, "rank": self.rank}

    @classmethod
    def from_snapshot(cls, data):
        if isinstance(data, Card):
            return data
        if isinstance(data, dict):
            return cls(int(data["color"]), int(data["rank"]))
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return cls(int(data[0]), int(data[1]))
        raise ValueError(f"invalid card snapshot: {data!r}")

    def __hash__(self):
        return (self.color << 8) | self.rank

    def __richcmp__(self, other, int op):
        if not isinstance(other, Card):
            return NotImplemented
        cdef Card o = <Card>other
        cdef bint eq = self.color == o.color and self.rank == o.rank
        if op == 2:  # ==
            return eq
        if op == 3:  # !=
            return not eq
        cdef bint lt
        if self.color != o.color:
            lt = self.color < o.color
        else:
            lt = self.rank < o.rank
        if op == 0:  # <
            return lt
        if op == 1:  # <=
            return lt or eq
        if op == 4:  # >
            return not lt and not eq
        if op == 5:  # >=
            return not lt
        return NotImplemented

    def __repr__(self):
        return f"Card(color={self.color}, rank={self.rank})"

    def __reduce__(self):
        return (Card, (self.color, self.rank))


@dataclass(frozen=True)
class LostCitiesConfig:
    n_colors: int = 5
    n_ranks: int = 9
    min_rank: int = 2
    n_handshakes: int = 3
    hand_size: int = 8
    expedition_penalty: int = -20
    bonus_threshold: int = 8
    bonus_amount: int = 20
    seed: int | None = None

    @property
    def deck_size(self) -> int:
        return self.n_colors * (self.n_ranks + self.n_handshakes)

    @property
    def max_rank(self) -> int:
        return self.min_rank + self.n_ranks - 1

    @property
    def card_action_size(self) -> int:
        return 2 * self.hand_size

    @property
    def draw_action_size(self) -> int:
        return 1 + self.n_colors

    @property
    def action_size(self) -> int:
        return self.card_action_size + self.draw_action_size

    def validate(self) -> None:
        if self.n_colors <= 0:
            raise ValueError("n_colors must be positive")
        if self.n_ranks <= 0:
            raise ValueError("n_ranks must be positive")
        if self.min_rank <= 0:
            raise ValueError("min_rank must be positive")
        if self.n_handshakes < 0:
            raise ValueError("n_handshakes cannot be negative")
        if self.hand_size <= 0:
            raise ValueError("hand_size must be positive")
        if self.deck_size < 2 * self.hand_size:
            raise ValueError("deck must contain at least both initial hands")
        if self.bonus_threshold <= 0:
            raise ValueError("bonus_threshold must be positive")

    def to_snapshot(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def classic_config(*, seed=None):
    return LostCitiesConfig(seed=seed)


def config_from_mapping(data):
    allowed = LostCitiesConfig.__dataclass_fields__.keys()
    kwargs = {key: value for key, value in data.items() if key in allowed}
    config = LostCitiesConfig(**kwargs)
    config.validate()
    return config


def config_to_mapping(config):
    return config.to_snapshot()


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load Lost Cities YAML configs") from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in config file: {path}")
    return config_from_mapping(data)


def build_deck(config):
    config.validate()
    cdef list deck = []
    cdef int color, rank
    cdef int n_colors = config.n_colors
    cdef int n_handshakes = config.n_handshakes
    cdef int n_ranks = config.n_ranks
    for color in range(n_colors):
        for _ in range(n_handshakes):
            deck.append(Card(color, 0))
        for rank in range(1, n_ranks + 1):
            deck.append(Card(color, rank))
    return deck


def _card_counter(cards):
    return Counter(cards)


def _cards_from_snapshot(data):
    if not isinstance(data, list):
        raise ValueError(f"expected card list snapshot, got {type(data).__name__}")
    return [Card.from_snapshot(card) for card in data]


def _cards_to_snapshot(cards):
    return [card.to_snapshot() for card in cards]


cdef class GameState:
    cdef public object config
    cdef public list deck
    cdef public list hands
    cdef public list expeditions
    cdef public list discards
    cdef public int current_player
    cdef public str phase
    cdef public object pending_discarded_color
    cdef public int turn_count
    cdef public bint terminal

    def __init__(
        self,
        config,
        deck=None,
        hands=None,
        expeditions=None,
        discards=None,
        int current_player=0,
        phase="card",
        pending_discarded_color=None,
        int turn_count=0,
        bint terminal=False,
    ):
        self.config = config
        self.deck = list(deck) if deck is not None else []
        self.hands = hands if hands is not None else [[], []]
        self.expeditions = expeditions if expeditions is not None else [
            [[] for _ in range(config.n_colors)],
            [[] for _ in range(config.n_colors)],
        ]
        self.discards = discards if discards is not None else [
            [] for _ in range(config.n_colors)
        ]
        self.current_player = current_player
        self.phase = phase
        self.pending_discarded_color = pending_discarded_color
        self.turn_count = turn_count
        self.terminal = terminal

    @classmethod
    def new_game(cls, config=None, *, seed=None):
        config = config or LostCitiesConfig()
        config.validate()
        rng = random.Random(config.seed if seed is None else seed)
        deck = build_deck(config)
        rng.shuffle(deck)
        return cls.new_game_from_deck(deck, config)

    @classmethod
    def new_game_from_deck(cls, deck, config=None):
        config = config or LostCitiesConfig()
        config.validate()
        cards = [Card.from_snapshot(card) for card in deck]
        if _card_counter(cards) != _card_counter(build_deck(config)):
            raise ValueError("deck must contain exactly the cards defined by config")

        state = cls.empty(config)
        state.deck = list(cards)
        cdef int player
        for _ in range(config.hand_size):
            for player in range(2):
                state.hands[player].append(state.deck.pop())
        state.validate_invariants()
        return state

    @classmethod
    def empty(cls, config=None):
        config = config or LostCitiesConfig()
        config.validate()
        return cls(
            config=config,
            deck=[],
            hands=[[], []],
            expeditions=[
                [[] for _ in range(config.n_colors)],
                [[] for _ in range(config.n_colors)],
            ],
            discards=[[] for _ in range(config.n_colors)],
        )

    @classmethod
    def from_snapshot(cls, snapshot, *, validate=True):
        config = config_from_mapping(snapshot["config"])
        phase = snapshot.get("phase", "card")
        if phase not in ("card", "draw"):
            raise ValueError(f"invalid phase: {phase!r}")

        state = cls(
            config=config,
            deck=_cards_from_snapshot(snapshot["deck"]),
            hands=[
                _cards_from_snapshot(snapshot["hands"][0]),
                _cards_from_snapshot(snapshot["hands"][1]),
            ],
            expeditions=[
                [
                    _cards_from_snapshot(color_cards)
                    for color_cards in snapshot["expeditions"][0]
                ],
                [
                    _cards_from_snapshot(color_cards)
                    for color_cards in snapshot["expeditions"][1]
                ],
            ],
            discards=[
                _cards_from_snapshot(color_cards)
                for color_cards in snapshot["discards"]
            ],
            current_player=int(snapshot.get("current_player", 0)),
            phase=phase,
            pending_discarded_color=snapshot.get("pending_discarded_color"),
            turn_count=int(snapshot.get("turn_count", 0)),
            terminal=bool(snapshot.get("terminal", False)),
        )
        if state.pending_discarded_color is not None:
            state.pending_discarded_color = int(state.pending_discarded_color)
        if validate:
            state.validate_invariants()
        return state

    def to_snapshot(self):
        return {
            "config": self.config.to_snapshot(),
            "deck": _cards_to_snapshot(self.deck),
            "hands": [_cards_to_snapshot(hand) for hand in self.hands],
            "expeditions": [
                [_cards_to_snapshot(expedition) for expedition in player_expeditions]
                for player_expeditions in self.expeditions
            ],
            "discards": [_cards_to_snapshot(discard) for discard in self.discards],
            "current_player": self.current_player,
            "phase": self.phase,
            "pending_discarded_color": self.pending_discarded_color,
            "turn_count": self.turn_count,
            "terminal": self.terminal,
        }

    cpdef GameState clone(self):
        cdef GameState other = GameState.__new__(GameState)
        other.config = self.config
        other.deck = list(self.deck)
        other.hands = [list(self.hands[0]), list(self.hands[1])]
        other.expeditions = [
            [list(exp) for exp in self.expeditions[0]],
            [list(exp) for exp in self.expeditions[1]],
        ]
        other.discards = [list(pile) for pile in self.discards]
        other.current_player = self.current_player
        other.phase = self.phase
        other.pending_discarded_color = self.pending_discarded_color
        other.turn_count = self.turn_count
        other.terminal = self.terminal
        return other

    @property
    def card_action_size(self):
        return self.config.card_action_size

    @property
    def draw_action_size(self):
        return self.config.draw_action_size

    @property
    def action_size(self):
        return self.config.action_size

    def sort_hands(self):
        cdef int player
        for player in range(2):
            self.sort_hand(player)

    def sort_hand(self, player=None):
        cdef int p = self.current_player if player is None else int(player)
        self.hands[p].sort(key=_card_sort_key)

    def hand_slots(self, player=None):
        cdef int p = self.current_player if player is None else int(player)
        cdef list hand = self.hands[p]
        cdef int hand_size = self.config.hand_size
        cdef int n = len(hand)
        cdef int i
        cdef list out = []
        for i in range(hand_size):
            if i < n:
                out.append(hand[i])
            else:
                out.append(None)
        return out

    cpdef int last_numeric_rank(self, int player, int color):
        cdef list expedition = self.expeditions[player][color]
        cdef int best = 0
        cdef int n = len(expedition)
        cdef int i
        cdef Card card
        for i in range(n):
            card = <Card>expedition[i]
            if card.rank == 0:
                continue
            if card.rank > best:
                best = card.rank
        return best

    def has_numeric(self, int player, int color):
        return self.last_numeric_rank(player, color) > 0

    cpdef bint can_play_card(self, int player, Card card):
        cdef int n_colors = self.config.n_colors
        cdef int n_ranks = self.config.n_ranks
        if card.color < 0 or card.color >= n_colors:
            return False
        if card.rank < 0 or card.rank > n_ranks:
            return False
        cdef int last_numeric = self.last_numeric_rank(player, card.color)
        if card.rank == 0:
            return last_numeric == 0
        return card.rank > last_numeric

    cpdef list legal_card_mask(self):
        cdef int size = self.card_action_size
        cdef list mask = [False] * size
        if self.terminal:
            return mask
        cdef list hand = self.hands[self.current_player]
        cdef int hand_size = self.config.hand_size
        cdef int n = len(hand)
        cdef int slot
        cdef Card card
        for slot in range(hand_size):
            if slot >= n:
                continue
            card = <Card>hand[slot]
            mask[2 * slot] = self.can_play_card(self.current_player, card)
            mask[2 * slot + 1] = True
        return mask

    cpdef list legal_draw_mask(self):
        cdef int size = self.draw_action_size
        cdef list mask = [False] * size
        if self.terminal:
            return mask
        mask[0] = len(self.deck) > 0
        cdef int n_colors = self.config.n_colors
        cdef int color
        cdef object pending = self.pending_discarded_color
        for color in range(n_colors):
            mask[1 + color] = (
                len(self.discards[color]) > 0
                and (pending is None or color != pending)
            )
        return mask

    cpdef list legal_mask(self):
        if self.phase == "card":
            return self.legal_card_mask()
        return self.legal_draw_mask()

    cpdef list unified_legal_mask(self):
        cdef int draw_size = self.draw_action_size
        cdef int card_size = self.card_action_size
        cdef list result
        if self.phase == "card":
            result = self.legal_card_mask()
            result.extend([False] * draw_size)
            return result
        result = [False] * card_size
        result.extend(self.legal_draw_mask())
        return result

    cpdef object unified_legal_mask_np(self):
        cdef int n_colors = self.config.n_colors
        cdef int hand_size = self.config.hand_size
        cdef int card_action_size = 2 * hand_size
        cdef int draw_action_size = 1 + n_colors
        cdef int total = card_action_size + draw_action_size

        mask_arr = np.zeros(total, dtype=bool)
        cdef unsigned char[::1] view = mask_arr.view(np.uint8)
        if self.terminal:
            return mask_arr

        cdef int slot, n, color
        cdef Card card
        cdef list hand
        cdef int p = self.current_player
        cdef object pending

        if self.phase == "card":
            hand = self.hands[p]
            n = len(hand)
            for slot in range(hand_size):
                if slot >= n:
                    continue
                card = <Card>hand[slot]
                if self.can_play_card(p, card):
                    view[2 * slot] = 1
                view[2 * slot + 1] = 1
        else:
            pending = self.pending_discarded_color
            if len(self.deck) > 0:
                view[card_action_size] = 1
            for color in range(n_colors):
                if (
                    len(self.discards[color]) > 0
                    and (pending is None or color != pending)
                ):
                    view[card_action_size + 1 + color] = 1
        return mask_arr

    def to_unified_action(self, int action_id, phase=None):
        cdef str p = self.phase if phase is None else phase
        if p == "card":
            if action_id < 0 or action_id >= self.card_action_size:
                raise IllegalMoveError(f"card action {action_id} is out of range")
            return action_id
        if action_id < 0 or action_id >= self.draw_action_size:
            raise IllegalMoveError(f"draw action {action_id} is out of range")
        return self.card_action_size + action_id

    cpdef int from_unified_action(self, int action_id):
        if action_id < 0 or action_id >= self.action_size:
            raise IllegalMoveError(f"action {action_id} is out of range")
        if self.phase == "card":
            if action_id >= self.card_action_size:
                raise IllegalMoveError(
                    f"card action {action_id} is illegal during card phase"
                )
            return action_id
        if action_id < self.card_action_size:
            raise IllegalMoveError(
                f"card action {action_id} is illegal during draw phase"
            )
        return action_id - self.card_action_size

    cpdef apply_action(self, int action_id):
        if self.terminal:
            raise IllegalMoveError("game is already terminal")
        cdef list mask = self.legal_mask()
        if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
            raise IllegalMoveError(
                f"illegal action {action_id} in phase {self.phase} "
                f"for player {self.current_player}"
            )
        if self.phase == "card":
            self._apply_card_action(action_id)
        else:
            self._apply_draw_action(action_id)

    cpdef apply_unified_action(self, int action_id):
        self.apply_action(self.from_unified_action(action_id))

    cpdef object apply_action_with_undo(self, int action_id):
        if self.terminal:
            raise IllegalMoveError("game is already terminal")
        cdef list mask = self.legal_mask()
        if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
            raise IllegalMoveError(
                f"illegal action {action_id} in phase {self.phase} "
                f"for player {self.current_player}"
            )
        cdef object undo
        if self.phase == "card":
            undo = self._card_action_undo(action_id)
            self._apply_card_action(action_id)
        else:
            undo = self._draw_action_undo(action_id)
            self._apply_draw_action(action_id)
        return undo

    cpdef object apply_unified_action_with_undo(self, int action_id):
        return self.apply_action_with_undo(self.from_unified_action(action_id))

    cpdef undo_action(self, object undo):
        cdef str phase = undo[0]
        if phase == "card":
            self._undo_card_action(undo)
            return
        if phase == "draw":
            self._undo_draw_action(undo)
            return
        raise ValueError(f"invalid undo phase: {phase!r}")

    cdef object _card_action_undo(self, int action_id):
        cdef int slot = action_id // 2
        cdef bint play = action_id % 2 == 0
        cdef Card card = <Card>self.hands[self.current_player][slot]
        return (
            "card",
            self.current_player,
            action_id,
            self.pending_discarded_color,
            self.terminal,
            slot,
            play,
            card,
        )

    cdef object _draw_action_undo(self, int action_id):
        cdef Card card
        cdef list source
        if action_id == 0:
            source = self.deck
        else:
            source = self.discards[action_id - 1]
        card = <Card>source[len(source) - 1]
        return (
            "draw",
            self.current_player,
            action_id,
            self.pending_discarded_color,
            self.terminal,
            self.turn_count,
            card,
        )

    cdef void _apply_card_action(self, int action_id) except *:
        cdef int slot = action_id // 2
        cdef bint play = action_id % 2 == 0
        cdef Card card = <Card>self.hands[self.current_player].pop(slot)
        if play:
            self.expeditions[self.current_player][card.color].append(card)
        else:
            self.discards[card.color].append(card)
            self.pending_discarded_color = card.color
        self.phase = "draw"
        cdef int n_colors = self.config.n_colors
        cdef int color
        cdef object pending = self.pending_discarded_color
        cdef bint any_legal_draw = False
        if len(self.deck) == 0:
            for color in range(n_colors):
                if len(self.discards[color]) > 0 and (pending is None or color != pending):
                    any_legal_draw = True
                    break
            if not any_legal_draw:
                self.terminal = True

    cdef void _apply_draw_action(self, int action_id) except *:
        cdef Card card
        cdef int color
        if action_id == 0:
            card = <Card>self.deck.pop()
        else:
            color = action_id - 1
            card = <Card>self.discards[color].pop()
        self.hands[self.current_player].append(card)
        self.pending_discarded_color = None
        self.turn_count += 1
        if len(self.deck) == 0:
            self.terminal = True
            return
        self.current_player = 1 - self.current_player
        self.phase = "card"

    cdef void _undo_card_action(self, object undo) except *:
        cdef int player = <int>undo[1]
        cdef object pending_before = undo[3]
        cdef bint terminal_before = <bint>undo[4]
        cdef int slot = <int>undo[5]
        cdef bint play = <bint>undo[6]
        cdef Card card = <Card>undo[7]
        cdef Card moved
        if play:
            moved = <Card>self.expeditions[player][card.color].pop()
        else:
            moved = <Card>self.discards[card.color].pop()
        if moved != card:
            raise ValueError("undo card mismatch")
        self.hands[player].insert(slot, card)
        self.current_player = player
        self.phase = "card"
        self.pending_discarded_color = pending_before
        self.terminal = terminal_before

    cdef void _undo_draw_action(self, object undo) except *:
        cdef int player = <int>undo[1]
        cdef int action_id = <int>undo[2]
        cdef object pending_before = undo[3]
        cdef bint terminal_before = <bint>undo[4]
        cdef int turn_count_before = <int>undo[5]
        cdef Card card = <Card>undo[6]
        cdef Card moved = <Card>self.hands[player].pop()
        if moved != card:
            raise ValueError("undo draw mismatch")
        if action_id == 0:
            self.deck.append(card)
        else:
            self.discards[action_id - 1].append(card)
        self.current_player = player
        self.phase = "draw"
        self.pending_discarded_color = pending_before
        self.turn_count = turn_count_before
        self.terminal = terminal_before

    cpdef int expedition_score(self, int player, int color):
        return score_expedition(self.expeditions[player][color], self.config)

    cpdef int total_score(self, int player):
        cdef int total = 0
        cdef int color
        cdef int n_colors = self.config.n_colors
        for color in range(n_colors):
            total += score_expedition(self.expeditions[player][color], self.config)
        return total

    cpdef int score_diff(self, int player=0):
        cdef int other = 1 - player
        return self.total_score(player) - self.total_score(other)

    def validate_invariants(self):
        self.config.validate()
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if self.phase not in ("card", "draw"):
            raise ValueError(f"invalid phase: {self.phase!r}")
        if len(self.hands) != 2:
            raise ValueError("hands must contain two players")
        if len(self.expeditions) != 2:
            raise ValueError("expeditions must contain two players")
        if len(self.discards) != self.config.n_colors:
            raise ValueError("discard pile count must match n_colors")

        all_cards = []
        all_cards.extend(self.deck)
        for player, hand in enumerate(self.hands):
            if len(hand) > self.config.hand_size:
                raise ValueError(f"hand {player} exceeds hand_size")
            all_cards.extend(hand)

        for player, expeditions in enumerate(self.expeditions):
            if len(expeditions) != self.config.n_colors:
                raise ValueError("expedition color count must match n_colors")
            for color, expedition in enumerate(expeditions):
                self._validate_expedition(player, color, expedition)
                all_cards.extend(expedition)
        for discard in self.discards:
            all_cards.extend(discard)

        for card in all_cards:
            self._validate_card(card)
        if _card_counter(all_cards) != _card_counter(build_deck(self.config)):
            raise ValueError("card conservation failed")

        if self.phase == "card" and self.pending_discarded_color is not None:
            raise ValueError("pending_discarded_color must be None during card phase")
        if self.pending_discarded_color is not None:
            color = self.pending_discarded_color
            if color < 0 or color >= self.config.n_colors:
                raise ValueError("pending_discarded_color is out of range")
            if not self.discards[color]:
                raise ValueError("pending discard color must have a discard pile card")

        any_legal = any(self.unified_legal_mask())
        if self.terminal and any_legal:
            raise ValueError("terminal state must have no legal actions")
        if not self.terminal and not any_legal:
            raise ValueError("non-terminal state must have at least one legal action")

    def _validate_card(self, Card card):
        if card.color < 0 or card.color >= self.config.n_colors:
            raise ValueError(f"card color out of range: {card}")
        if card.rank < 0 or card.rank > self.config.n_ranks:
            raise ValueError(f"card rank out of range: {card}")

    def _validate_expedition(self, int player, int color, list expedition):
        cdef bint seen_numeric = False
        cdef int last_numeric = 0
        cdef Card card
        for card in expedition:
            if card.color != color:
                raise ValueError(
                    f"player {player} expedition {color} contains wrong color"
                )
            if card.rank == 0:
                if seen_numeric:
                    raise ValueError(
                        f"player {player} expedition {color} has handshake after number"
                    )
                continue
            seen_numeric = True
            if card.rank <= last_numeric:
                raise ValueError(
                    f"player {player} expedition {color} is not strictly increasing"
                )
            last_numeric = card.rank

    def __reduce__(self):
        # support pickle via snapshot round-trip
        return (_rebuild_game_state, (self.to_snapshot(),))


def _rebuild_game_state(snapshot):
    return GameState.from_snapshot(snapshot, validate=False)


def _card_sort_key(Card card):
    return (card.color, card.rank)


cpdef int score_expedition(list expedition, config):
    cdef int n = len(expedition)
    if n == 0:
        return 0
    cdef int min_rank = config.min_rank
    cdef int handshakes = 0
    cdef int numeric_sum = 0
    cdef int i
    cdef Card card
    for i in range(n):
        card = <Card>expedition[i]
        if card.rank == 0:
            handshakes += 1
        else:
            numeric_sum += min_rank + card.rank - 1
    cdef int score = (numeric_sum + config.expedition_penalty) * (handshakes + 1)
    if n >= config.bonus_threshold:
        score += config.bonus_amount
    return score
