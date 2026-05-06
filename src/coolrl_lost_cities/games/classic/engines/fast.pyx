# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""C-array based experimental Lost Cities classic engine."""

from collections import Counter
import random

from libc.stdlib cimport free, malloc

from ..game import IllegalMoveError, LostCitiesConfig, config_from_mapping


cdef inline int _phase_card():
    return 0


cdef inline int _phase_draw():
    return 1


cdef class FastGameState:
    def __cinit__(self):
        self.deck = NULL
        self.hands = NULL
        self.expeditions = NULL
        self.expedition_lens = NULL
        self.discards = NULL
        self.discard_lens = NULL
        self.last_numeric_ranks = NULL
        self.handshake_counts = NULL
        self.numeric_sums = NULL
        self.expedition_scores = NULL

    def __init__(self, config=None):
        config = config or LostCitiesConfig()
        config.validate()
        self._configure(config)

    def __dealloc__(self):
        if self.deck != NULL:
            free(self.deck)
        if self.hands != NULL:
            free(self.hands)
        if self.expeditions != NULL:
            free(self.expeditions)
        if self.expedition_lens != NULL:
            free(self.expedition_lens)
        if self.discards != NULL:
            free(self.discards)
        if self.discard_lens != NULL:
            free(self.discard_lens)
        if self.last_numeric_ranks != NULL:
            free(self.last_numeric_ranks)
        if self.handshake_counts != NULL:
            free(self.handshake_counts)
        if self.numeric_sums != NULL:
            free(self.numeric_sums)
        if self.expedition_scores != NULL:
            free(self.expedition_scores)

    cdef void _configure(self, object config) except *:
        self.config = config
        self.n_colors = int(config.n_colors)
        self.n_ranks = int(config.n_ranks)
        self.min_rank = int(config.min_rank)
        self.n_handshakes = int(config.n_handshakes)
        self.hand_size = int(config.hand_size)
        self.expedition_penalty = int(config.expedition_penalty)
        self.bonus_threshold = int(config.bonus_threshold)
        self.bonus_amount = int(config.bonus_amount)
        self.total_cards = int(config.deck_size)
        self.cards_per_color = self.n_ranks + self.n_handshakes
        self.stride = self.n_ranks + 1

        self.deck = <int*>malloc(self.total_cards * sizeof(int))
        self.hands = <int*>malloc(2 * self.hand_size * sizeof(int))
        self.expeditions = <int*>malloc(
            2 * self.n_colors * self.cards_per_color * sizeof(int)
        )
        self.expedition_lens = <int*>malloc(2 * self.n_colors * sizeof(int))
        self.discards = <int*>malloc(self.n_colors * self.cards_per_color * sizeof(int))
        self.discard_lens = <int*>malloc(self.n_colors * sizeof(int))
        self.last_numeric_ranks = <int*>malloc(2 * self.n_colors * sizeof(int))
        self.handshake_counts = <int*>malloc(2 * self.n_colors * sizeof(int))
        self.numeric_sums = <int*>malloc(2 * self.n_colors * sizeof(int))
        self.expedition_scores = <int*>malloc(2 * self.n_colors * sizeof(int))
        if (
            self.deck == NULL
            or self.hands == NULL
            or self.expeditions == NULL
            or self.expedition_lens == NULL
            or self.discards == NULL
            or self.discard_lens == NULL
            or self.last_numeric_ranks == NULL
            or self.handshake_counts == NULL
            or self.numeric_sums == NULL
            or self.expedition_scores == NULL
        ):
            raise MemoryError()
        self._clear()

    cdef void _clear(self) noexcept:
        cdef int i
        self.deck_len = 0
        self.hand_lens[0] = 0
        self.hand_lens[1] = 0
        for i in range(2 * self.n_colors):
            self.expedition_lens[i] = 0
            self.last_numeric_ranks[i] = 0
            self.handshake_counts[i] = 0
            self.numeric_sums[i] = 0
            self.expedition_scores[i] = 0
        for i in range(self.n_colors):
            self.discard_lens[i] = 0
        self.total_scores[0] = 0
        self.total_scores[1] = 0
        self.current_player = 0
        self.phase_id = _phase_card()
        self.pending_discarded_color = -1
        self.turn_count = 0
        self.terminal = False

    @classmethod
    def empty(cls, config=None):
        return cls(config or LostCitiesConfig())

    @classmethod
    def new_game(cls, config=None, *, seed=None):
        config = config or LostCitiesConfig()
        config.validate()
        deck = _build_encoded_deck(config)
        rng = random.Random(config.seed if seed is None else seed)
        rng.shuffle(deck)
        return cls.new_game_from_deck(deck, config)

    @classmethod
    def new_game_from_deck(cls, deck, config=None):
        config = config or LostCitiesConfig()
        config.validate()
        encoded = [_encode_card_snapshot(card, config) for card in deck]
        if Counter(encoded) != Counter(_build_encoded_deck(config)):
            raise ValueError("deck must contain exactly the cards defined by config")

        cdef int i
        cdef int player
        cdef FastGameState state = cls(config)
        state.deck_len = len(encoded)
        for i, card in enumerate(encoded):
            state.deck[i] = <int>card
        for _ in range(config.hand_size):
            for player in range(2):
                state.deck_len -= 1
                state.hands[state._hand_index(player, state.hand_lens[player])] = state.deck[
                    state.deck_len
                ]
                state.hand_lens[player] += 1
        state.validate_invariants()
        return state

    @classmethod
    def from_snapshot(cls, snapshot, *, validate=True):
        config = config_from_mapping(snapshot["config"])
        cdef FastGameState state = cls(config)
        cdef int player
        cdef int color
        cdef int index
        cdef list cards

        cards = [_encode_card_snapshot(card, config) for card in snapshot["deck"]]
        state.deck_len = len(cards)
        for index, card in enumerate(cards):
            state.deck[index] = <int>card

        for player in range(2):
            cards = [
                _encode_card_snapshot(card, config) for card in snapshot["hands"][player]
            ]
            state.hand_lens[player] = len(cards)
            for index, card in enumerate(cards):
                state.hands[state._hand_index(player, index)] = <int>card

        for player in range(2):
            for color in range(state.n_colors):
                cards = [
                    _encode_card_snapshot(card, config)
                    for card in snapshot["expeditions"][player][color]
                ]
                state.expedition_lens[state._expedition_len_index(player, color)] = len(cards)
                for index, card in enumerate(cards):
                    state.expeditions[state._expedition_index(player, color, index)] = <int>card

        for color in range(state.n_colors):
            cards = [_encode_card_snapshot(card, config) for card in snapshot["discards"][color]]
            state.discard_lens[color] = len(cards)
            for index, card in enumerate(cards):
                state.discards[state._discard_index(color, index)] = <int>card

        state.current_player = int(snapshot.get("current_player", 0))
        state.phase = snapshot.get("phase", "card")
        pending = snapshot.get("pending_discarded_color")
        state.pending_discarded_color = -1 if pending is None else int(pending)
        state.turn_count = int(snapshot.get("turn_count", 0))
        state.terminal = bool(snapshot.get("terminal", False))
        state._recompute_score_caches()
        if validate:
            state.validate_invariants()
        return state

    @property
    def phase(self):
        return "card" if self.phase_id == _phase_card() else "draw"

    @phase.setter
    def phase(self, value):
        if value == "card":
            self.phase_id = _phase_card()
        elif value == "draw":
            self.phase_id = _phase_draw()
        else:
            raise ValueError(f"invalid phase: {value!r}")

    @property
    def card_action_size(self):
        return 2 * self.hand_size

    @property
    def draw_action_size(self):
        return 1 + self.n_colors

    @property
    def action_size(self):
        return self.card_action_size + self.draw_action_size

    def to_snapshot(self):
        return {
            "config": self.config.to_snapshot(),
            "deck": [self._card_snapshot(self.deck[i]) for i in range(self.deck_len)],
            "hands": [
                [
                    self._card_snapshot(self.hands[self._hand_index(player, i)])
                    for i in range(self.hand_lens[player])
                ]
                for player in range(2)
            ],
            "expeditions": [
                [
                    [
                        self._card_snapshot(
                            self.expeditions[self._expedition_index(player, color, i)]
                        )
                        for i in range(
                            self.expedition_lens[
                                self._expedition_len_index(player, color)
                            ]
                        )
                    ]
                    for color in range(self.n_colors)
                ]
                for player in range(2)
            ],
            "discards": [
                [
                    self._card_snapshot(self.discards[self._discard_index(color, i)])
                    for i in range(self.discard_lens[color])
                ]
                for color in range(self.n_colors)
            ],
            "current_player": self.current_player,
            "phase": self.phase,
            "pending_discarded_color": (
                None if self.pending_discarded_color < 0 else self.pending_discarded_color
            ),
            "turn_count": self.turn_count,
            "terminal": self.terminal,
        }

    cpdef FastGameState clone(self):
        cdef FastGameState other = FastGameState(self.config)
        cdef int i
        other.deck_len = self.deck_len
        for i in range(self.deck_len):
            other.deck[i] = self.deck[i]
        for i in range(2 * self.hand_size):
            other.hands[i] = self.hands[i]
        other.hand_lens[0] = self.hand_lens[0]
        other.hand_lens[1] = self.hand_lens[1]
        for i in range(2 * self.n_colors * self.cards_per_color):
            other.expeditions[i] = self.expeditions[i]
        for i in range(2 * self.n_colors):
            other.expedition_lens[i] = self.expedition_lens[i]
            other.last_numeric_ranks[i] = self.last_numeric_ranks[i]
            other.handshake_counts[i] = self.handshake_counts[i]
            other.numeric_sums[i] = self.numeric_sums[i]
            other.expedition_scores[i] = self.expedition_scores[i]
        for i in range(self.n_colors * self.cards_per_color):
            other.discards[i] = self.discards[i]
        for i in range(self.n_colors):
            other.discard_lens[i] = self.discard_lens[i]
        other.total_scores[0] = self.total_scores[0]
        other.total_scores[1] = self.total_scores[1]
        other.current_player = self.current_player
        other.phase_id = self.phase_id
        other.pending_discarded_color = self.pending_discarded_color
        other.turn_count = self.turn_count
        other.terminal = self.terminal
        return other

    cpdef list legal_card_mask(self):
        cdef list mask = [False] * (2 * self.hand_size)
        cdef int slot
        cdef int card
        if self.terminal:
            return mask
        for slot in range(self.hand_lens[self.current_player]):
            card = self.hands[self._hand_index(self.current_player, slot)]
            mask[2 * slot] = self.can_play_encoded_card(self.current_player, card)
            mask[2 * slot + 1] = True
        return mask

    cpdef list legal_draw_mask(self):
        cdef list mask = [False] * (1 + self.n_colors)
        cdef int color
        if self.terminal:
            return mask
        mask[0] = self.deck_len > 0
        for color in range(self.n_colors):
            mask[1 + color] = (
                self.discard_lens[color] > 0
                and (self.pending_discarded_color < 0 or color != self.pending_discarded_color)
            )
        return mask

    cpdef list legal_mask(self):
        if self.phase_id == _phase_card():
            return self.legal_card_mask()
        return self.legal_draw_mask()

    cpdef list unified_legal_mask(self):
        cdef list result
        if self.phase_id == _phase_card():
            result = self.legal_card_mask()
            result.extend([False] * (1 + self.n_colors))
            return result
        result = [False] * (2 * self.hand_size)
        result.extend(self.legal_draw_mask())
        return result

    cpdef list legal_actions(self):
        cdef int* actions = <int*>malloc(self.action_size * sizeof(int))
        if actions == NULL:
            raise MemoryError()
        cdef int count
        cdef int i
        try:
            count = self._legal_actions_c(actions)
            return [actions[i] for i in range(count)]
        finally:
            free(actions)

    cpdef list unified_legal_actions(self):
        cdef int* actions = <int*>malloc(self.action_size * sizeof(int))
        if actions == NULL:
            raise MemoryError()
        cdef int count
        cdef int i
        try:
            count = self._unified_legal_actions_c(actions)
            return [actions[i] for i in range(count)]
        finally:
            free(actions)

    cpdef int from_unified_action(self, int action_id):
        cdef int card_action_size = 2 * self.hand_size
        cdef int action_size = card_action_size + 1 + self.n_colors
        if action_id < 0 or action_id >= action_size:
            raise IllegalMoveError(f"action {action_id} is out of range")
        if self.phase_id == _phase_card():
            if action_id >= card_action_size:
                raise IllegalMoveError(
                    f"card action {action_id} is illegal during card phase"
                )
            return action_id
        if action_id < card_action_size:
            raise IllegalMoveError(
                f"card action {action_id} is illegal during draw phase"
            )
        return action_id - card_action_size

    def to_unified_action(self, int action_id, phase=None):
        cdef object p = self.phase if phase is None else phase
        if p == "card":
            if action_id < 0 or action_id >= 2 * self.hand_size:
                raise IllegalMoveError(f"card action {action_id} is out of range")
            return action_id
        if action_id < 0 or action_id >= 1 + self.n_colors:
            raise IllegalMoveError(f"draw action {action_id} is out of range")
        return 2 * self.hand_size + action_id

    cpdef apply_action(self, int action_id):
        if self.terminal:
            raise IllegalMoveError("game is already terminal")
        if not self._is_legal_action_c(action_id):
            raise IllegalMoveError(
                f"illegal action {action_id} in phase {self.phase} "
                f"for player {self.current_player}"
            )
        self._apply_action_unchecked_c(action_id)

    cpdef apply_unified_action(self, int action_id):
        self.apply_action(self.from_unified_action(action_id))

    cpdef object apply_action_with_undo(self, int action_id):
        if self.terminal:
            raise IllegalMoveError("game is already terminal")
        if not self._is_legal_action_c(action_id):
            raise IllegalMoveError(
                f"illegal action {action_id} in phase {self.phase} "
                f"for player {self.current_player}"
            )
        cdef UndoRecord undo
        self._apply_action_with_undo_c(action_id, &undo)
        return self._undo_to_tuple(&undo)

    cpdef object apply_unified_action_with_undo(self, int action_id):
        return self.apply_action_with_undo(self.from_unified_action(action_id))

    cpdef undo_action(self, object undo):
        cdef UndoRecord record
        self._tuple_to_undo(undo, &record)
        self._undo_action_c(&record)

    cpdef bint can_play_encoded_card(self, int player, int card):
        cdef int color = self._card_color(card)
        cdef int rank = self._card_rank(card)
        if color < 0 or color >= self.n_colors:
            return False
        if rank < 0 or rank > self.n_ranks:
            return False
        if rank == 0:
            return self.last_numeric_ranks[self._expedition_len_index(player, color)] == 0
        return rank > self.last_numeric_ranks[self._expedition_len_index(player, color)]

    cpdef int last_numeric_rank(self, int player, int color):
        return self.last_numeric_ranks[self._expedition_len_index(player, color)]

    cpdef int expedition_score(self, int player, int color):
        return self.expedition_scores[self._expedition_len_index(player, color)]

    cpdef int total_score(self, int player):
        return self.total_scores[player]

    cpdef int score_diff(self, int player=0):
        return self.total_score(player) - self.total_score(1 - player)

    def validate_invariants(self):
        self.config.validate()
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if self.phase_id not in (_phase_card(), _phase_draw()):
            raise ValueError("invalid phase")
        if self.pending_discarded_color >= self.n_colors:
            raise ValueError("pending_discarded_color is out of range")
        if self.hand_lens[0] > self.hand_size or self.hand_lens[1] > self.hand_size:
            raise ValueError("hand exceeds hand_size")
        if Counter(_all_cards_from_snapshot(self.to_snapshot())) != Counter(
            _build_encoded_deck(self.config)
        ):
            raise ValueError("card conservation failed")
        if self.phase_id == _phase_card() and self.pending_discarded_color >= 0:
            raise ValueError("pending_discarded_color must be None during card phase")
        if self.pending_discarded_color >= 0 and self.discard_lens[self.pending_discarded_color] == 0:
            raise ValueError("pending discard color must have a discard pile card")
        any_legal = any(self.unified_legal_mask())
        if self.terminal and any_legal:
            raise ValueError("terminal state must have no legal actions")
        if not self.terminal and not any_legal:
            raise ValueError("non-terminal state must have at least one legal action")

    cdef bint _is_legal_action_c(self, int action_id) noexcept:
        cdef int slot
        cdef int color
        if self.terminal:
            return False
        if self.phase_id == _phase_card():
            if action_id < 0 or action_id >= 2 * self.hand_size:
                return False
            slot = action_id // 2
            if slot >= self.hand_lens[self.current_player]:
                return False
            if action_id % 2 == 1:
                return True
            return self._can_play_encoded_card_c(
                self.current_player,
                self.hands[self._hand_index(self.current_player, slot)],
            )
        if action_id < 0 or action_id >= 1 + self.n_colors:
            return False
        if action_id == 0:
            return self.deck_len > 0
        color = action_id - 1
        return (
            self.discard_lens[color] > 0
            and (self.pending_discarded_color < 0 or color != self.pending_discarded_color)
        )

    cdef int _legal_actions_c(self, int* out_actions) noexcept:
        cdef int count = 0
        cdef int slot
        cdef int color
        cdef int card
        if self.terminal:
            return 0
        if self.phase_id == _phase_card():
            for slot in range(self.hand_lens[self.current_player]):
                card = self.hands[self._hand_index(self.current_player, slot)]
                if self._can_play_encoded_card_c(self.current_player, card):
                    out_actions[count] = 2 * slot
                    count += 1
                out_actions[count] = 2 * slot + 1
                count += 1
            return count
        if self.deck_len > 0:
            out_actions[count] = 0
            count += 1
        for color in range(self.n_colors):
            if (
                self.discard_lens[color] > 0
                and (self.pending_discarded_color < 0 or color != self.pending_discarded_color)
            ):
                out_actions[count] = 1 + color
                count += 1
        return count

    cdef int _unified_legal_actions_c(self, int* out_actions) noexcept:
        cdef int count = 0
        cdef int slot
        cdef int color
        cdef int card
        cdef int card_action_size = 2 * self.hand_size
        if self.terminal:
            return 0
        if self.phase_id == _phase_card():
            for slot in range(self.hand_lens[self.current_player]):
                card = self.hands[self._hand_index(self.current_player, slot)]
                if self._can_play_encoded_card_c(self.current_player, card):
                    out_actions[count] = 2 * slot
                    count += 1
                out_actions[count] = 2 * slot + 1
                count += 1
            return count
        if self.deck_len > 0:
            out_actions[count] = card_action_size
            count += 1
        for color in range(self.n_colors):
            if (
                self.discard_lens[color] > 0
                and (self.pending_discarded_color < 0 or color != self.pending_discarded_color)
            ):
                out_actions[count] = card_action_size + 1 + color
                count += 1
        return count

    cdef bint _can_play_encoded_card_c(self, int player, int card) noexcept:
        cdef int color = self._card_color(card)
        cdef int rank = self._card_rank(card)
        if color < 0 or color >= self.n_colors:
            return False
        if rank < 0 or rank > self.n_ranks:
            return False
        if rank == 0:
            return self.last_numeric_ranks[self._expedition_len_index(player, color)] == 0
        return rank > self.last_numeric_ranks[self._expedition_len_index(player, color)]

    cdef void _fill_undo_c(self, int action_id, UndoRecord* undo) noexcept:
        cdef int slot
        cdef int card
        cdef int color
        cdef int cache_index
        undo.phase_id = self.phase_id
        undo.player = self.current_player
        undo.action_id = action_id
        undo.pending_before = self.pending_discarded_color
        undo.terminal_before = self.terminal
        undo.turn_count_before = self.turn_count
        undo.slot = -1
        undo.play = 0
        undo.card = -1
        undo.color = -1
        undo.last_numeric_before = 0
        undo.handshake_count_before = 0
        undo.numeric_sum_before = 0
        undo.expedition_score_before = 0
        undo.total_score_before = self.total_scores[self.current_player]
        if self.phase_id == _phase_card():
            slot = action_id // 2
            card = self.hands[self._hand_index(self.current_player, slot)]
            color = self._card_color(card)
            cache_index = self._expedition_len_index(self.current_player, color)
            undo.slot = slot
            undo.play = action_id % 2 == 0
            undo.card = card
            undo.color = color
            undo.last_numeric_before = self.last_numeric_ranks[cache_index]
            undo.handshake_count_before = self.handshake_counts[cache_index]
            undo.numeric_sum_before = self.numeric_sums[cache_index]
            undo.expedition_score_before = self.expedition_scores[cache_index]
        elif action_id == 0:
            undo.card = self.deck[self.deck_len - 1]
        else:
            color = action_id - 1
            undo.color = color
            undo.card = self.discards[self._discard_index(color, self.discard_lens[color] - 1)]

    cdef void _apply_action_with_undo_c(self, int action_id, UndoRecord* undo) except *:
        self._fill_undo_c(action_id, undo)
        self._apply_action_unchecked_c(action_id)

    cdef void _apply_action_unchecked_c(self, int action_id) except *:
        if self.phase_id == _phase_card():
            self._apply_card_action(action_id)
        else:
            self._apply_draw_action(action_id)

    cdef object _undo_to_tuple(self, UndoRecord* undo):
        return (
            "card" if undo.phase_id == _phase_card() else "draw",
            undo.player,
            undo.action_id,
            undo.pending_before,
            undo.terminal_before,
            undo.turn_count_before,
            undo.slot,
            undo.play,
            undo.card,
            undo.color,
            undo.last_numeric_before,
            undo.handshake_count_before,
            undo.numeric_sum_before,
            undo.expedition_score_before,
            undo.total_score_before,
        )

    cdef void _tuple_to_undo(self, object data, UndoRecord* undo) except *:
        cdef str phase = data[0]
        if phase == "card":
            undo.phase_id = _phase_card()
        elif phase == "draw":
            undo.phase_id = _phase_draw()
        else:
            raise ValueError(f"invalid undo phase: {phase!r}")
        undo.player = <int>data[1]
        undo.action_id = <int>data[2]
        undo.pending_before = <int>data[3]
        undo.terminal_before = <bint>data[4]
        undo.turn_count_before = <int>data[5]
        undo.slot = <int>data[6]
        undo.play = <int>data[7]
        undo.card = <int>data[8]
        undo.color = <int>data[9]
        undo.last_numeric_before = <int>data[10]
        undo.handshake_count_before = <int>data[11]
        undo.numeric_sum_before = <int>data[12]
        undo.expedition_score_before = <int>data[13]
        undo.total_score_before = <int>data[14]

    cdef void _apply_card_action(self, int action_id) except *:
        cdef int slot = action_id // 2
        cdef bint play = action_id % 2 == 0
        cdef int player = self.current_player
        cdef int card = self.hands[self._hand_index(player, slot)]
        cdef int color = self._card_color(card)
        cdef int rank = self._card_rank(card)
        cdef int i
        cdef int length_index
        cdef int old_score
        cdef int new_score
        for i in range(slot, self.hand_lens[player] - 1):
            self.hands[self._hand_index(player, i)] = self.hands[self._hand_index(player, i + 1)]
        self.hand_lens[player] -= 1
        if play:
            length_index = self._expedition_len_index(player, color)
            old_score = self.expedition_scores[length_index]
            self.expeditions[self._expedition_index(player, color, self.expedition_lens[length_index])] = card
            self.expedition_lens[length_index] += 1
            if rank == 0:
                self.handshake_counts[length_index] += 1
            else:
                self.numeric_sums[length_index] += self.min_rank + rank - 1
                self.last_numeric_ranks[length_index] = rank
            new_score = self._score_from_summary_c(
                self.expedition_lens[length_index],
                self.handshake_counts[length_index],
                self.numeric_sums[length_index],
            )
            self.expedition_scores[length_index] = new_score
            self.total_scores[player] += new_score - old_score
        else:
            self.discards[self._discard_index(color, self.discard_lens[color])] = card
            self.discard_lens[color] += 1
            self.pending_discarded_color = color
        self.phase_id = _phase_draw()
        if self.deck_len == 0 and not self._has_any_legal_draw():
            self.terminal = True

    cdef void _apply_draw_action(self, int action_id) except *:
        cdef int player = self.current_player
        cdef int card
        cdef int color
        if action_id == 0:
            self.deck_len -= 1
            card = self.deck[self.deck_len]
        else:
            color = action_id - 1
            self.discard_lens[color] -= 1
            card = self.discards[self._discard_index(color, self.discard_lens[color])]
        self.hands[self._hand_index(player, self.hand_lens[player])] = card
        self.hand_lens[player] += 1
        self.pending_discarded_color = -1
        self.turn_count += 1
        if self.deck_len == 0:
            self.terminal = True
            return
        self.current_player = 1 - self.current_player
        self.phase_id = _phase_card()

    cdef void _undo_action_c(self, UndoRecord* undo) except *:
        if undo.phase_id == _phase_card():
            self._undo_card_action_c(undo)
        elif undo.phase_id == _phase_draw():
            self._undo_draw_action_c(undo)
        else:
            raise ValueError("invalid undo phase")

    cdef void _undo_card_action_c(self, UndoRecord* undo) except *:
        cdef int player = undo.player
        cdef int pending_before = undo.pending_before
        cdef bint terminal_before = undo.terminal_before
        cdef int slot = undo.slot
        cdef bint play = undo.play
        cdef int card = undo.card
        cdef int color = self._card_color(card)
        cdef int moved
        cdef int i
        cdef int length_index
        if play:
            length_index = self._expedition_len_index(player, color)
            self.expedition_lens[length_index] -= 1
            moved = self.expeditions[self._expedition_index(player, color, self.expedition_lens[length_index])]
            self.last_numeric_ranks[length_index] = undo.last_numeric_before
            self.handshake_counts[length_index] = undo.handshake_count_before
            self.numeric_sums[length_index] = undo.numeric_sum_before
            self.expedition_scores[length_index] = undo.expedition_score_before
            self.total_scores[player] = undo.total_score_before
        else:
            self.discard_lens[color] -= 1
            moved = self.discards[self._discard_index(color, self.discard_lens[color])]
        if moved != card:
            raise ValueError("undo card mismatch")
        for i in range(self.hand_lens[player], slot, -1):
            self.hands[self._hand_index(player, i)] = self.hands[self._hand_index(player, i - 1)]
        self.hands[self._hand_index(player, slot)] = card
        self.hand_lens[player] += 1
        self.current_player = player
        self.phase_id = _phase_card()
        self.pending_discarded_color = pending_before
        self.terminal = terminal_before

    cdef void _undo_draw_action_c(self, UndoRecord* undo) except *:
        cdef int player = undo.player
        cdef int action_id = undo.action_id
        cdef int pending_before = undo.pending_before
        cdef bint terminal_before = undo.terminal_before
        cdef int turn_count_before = undo.turn_count_before
        cdef int card = undo.card
        cdef int moved
        cdef int color
        self.hand_lens[player] -= 1
        moved = self.hands[self._hand_index(player, self.hand_lens[player])]
        if moved != card:
            raise ValueError("undo draw mismatch")
        if action_id == 0:
            self.deck[self.deck_len] = card
            self.deck_len += 1
        else:
            color = action_id - 1
            self.discards[self._discard_index(color, self.discard_lens[color])] = card
            self.discard_lens[color] += 1
        self.current_player = player
        self.phase_id = _phase_draw()
        self.pending_discarded_color = pending_before
        self.turn_count = turn_count_before
        self.terminal = terminal_before

    cdef void _recompute_score_caches(self) noexcept:
        cdef int i
        cdef int player
        cdef int color
        cdef int cache_index
        cdef int length
        cdef int rank
        cdef int card_index
        for i in range(2 * self.n_colors):
            self.last_numeric_ranks[i] = 0
            self.handshake_counts[i] = 0
            self.numeric_sums[i] = 0
            self.expedition_scores[i] = 0
        self.total_scores[0] = 0
        self.total_scores[1] = 0
        for player in range(2):
            for color in range(self.n_colors):
                cache_index = self._expedition_len_index(player, color)
                length = self.expedition_lens[cache_index]
                for card_index in range(length):
                    rank = self._card_rank(
                        self.expeditions[
                            self._expedition_index(player, color, card_index)
                        ]
                    )
                    if rank == 0:
                        self.handshake_counts[cache_index] += 1
                    else:
                        self.numeric_sums[cache_index] += self.min_rank + rank - 1
                        if rank > self.last_numeric_ranks[cache_index]:
                            self.last_numeric_ranks[cache_index] = rank
                self.expedition_scores[cache_index] = self._score_from_summary_c(
                    length,
                    self.handshake_counts[cache_index],
                    self.numeric_sums[cache_index],
                )
                self.total_scores[player] += self.expedition_scores[cache_index]

    cdef inline int _score_from_summary_c(
        self,
        int length,
        int handshakes,
        int numeric_sum,
    ) noexcept:
        cdef int score
        if length == 0:
            return 0
        score = (numeric_sum + self.expedition_penalty) * (handshakes + 1)
        if length >= self.bonus_threshold:
            score += self.bonus_amount
        return score

    cdef bint _has_any_legal_draw(self):
        cdef int color
        if self.deck_len > 0:
            return True
        for color in range(self.n_colors):
            if (
                self.discard_lens[color] > 0
                and (self.pending_discarded_color < 0 or color != self.pending_discarded_color)
            ):
                return True
        return False

    cdef inline int _hand_index(self, int player, int slot):
        return player * self.hand_size + slot

    cdef inline int _expedition_len_index(self, int player, int color):
        return player * self.n_colors + color

    cdef inline int _expedition_index(self, int player, int color, int index):
        return (player * self.n_colors + color) * self.cards_per_color + index

    cdef inline int _discard_index(self, int color, int index):
        return color * self.cards_per_color + index

    cdef inline int _encode_card(self, int color, int rank):
        return color * self.stride + rank

    cdef inline int _card_color(self, int card):
        return card // self.stride

    cdef inline int _card_rank(self, int card):
        return card % self.stride

    cdef object _card_snapshot(self, int card):
        return {"color": self._card_color(card), "rank": self._card_rank(card)}


def _build_encoded_deck(config):
    deck = []
    stride = int(config.n_ranks) + 1
    for color in range(int(config.n_colors)):
        for _ in range(int(config.n_handshakes)):
            deck.append(color * stride)
        for rank in range(1, int(config.n_ranks) + 1):
            deck.append(color * stride + rank)
    return deck


def _encode_card_snapshot(data, config):
    stride = int(config.n_ranks) + 1
    if isinstance(data, int):
        return int(data)
    if isinstance(data, dict):
        return int(data["color"]) * stride + int(data["rank"])
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return int(data[0]) * stride + int(data[1])
    color = getattr(data, "color", None)
    rank = getattr(data, "rank", None)
    if color is not None and rank is not None:
        return int(color) * stride + int(rank)
    raise ValueError(f"invalid card snapshot: {data!r}")


def _all_cards_from_snapshot(snapshot):
    cards = []
    config = config_from_mapping(snapshot["config"])
    for card in snapshot["deck"]:
        cards.append(_encode_card_snapshot(card, config))
    for hand in snapshot["hands"]:
        for card in hand:
            cards.append(_encode_card_snapshot(card, config))
    for player_expeditions in snapshot["expeditions"]:
        for expedition in player_expeditions:
            for card in expedition:
                cards.append(_encode_card_snapshot(card, config))
    for discard in snapshot["discards"]:
        for card in discard:
            cards.append(_encode_card_snapshot(card, config))
    return cards
