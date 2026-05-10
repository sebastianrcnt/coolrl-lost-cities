from ..game cimport GameState


cdef class _CachedState:
    cdef GameState _state
    cdef public object config
    cdef public object hands
    cdef public object expeditions
    cdef public object discards
    cdef public object deck
    cdef int hand_encoded[2][16]
    cdef int hand_size[2]
    cdef int expedition_top[2][8]
    cdef int expedition_count[2][8]
    cdef int expedition_handshakes[2][8]
    cdef int expedition_numeric_sum[2][8]
    cdef int expedition_last_numeric[2][8]
    cdef int discard_top[8]
    cdef int discard_count[8]
    cdef int deck_remaining
    cdef int total_scores[2]
    cdef int current_player
    cdef int phase
    cdef int turn_count
    cdef int n_colors
    cdef int n_ranks
    cdef int min_rank
    cdef int hand_capacity
    cdef int bonus_threshold
    cdef int bonus_amount
    cdef int expedition_penalty
    cdef void _build(self, GameState state) except *
    cpdef list legal_card_mask(self)
    cpdef list legal_draw_mask(self)
    cpdef bint can_play_card(self, int player, object card)
    cpdef bint can_play_encoded(self, int player, int card)
    cpdef bint has_numeric(self, int player, int color)
    cpdef int score_diff(self, int player)


cdef class HeuristicBot:
    cdef public object params
    cdef double color_commit_cache[2][8]
    cdef unsigned char color_commit_valid[2][8]
    cdef signed char playability_cache[2][8][17]
    cdef void _reset_caches(self) noexcept
    cpdef int act_cython(self, GameState state) except -1
    cdef int _card_color_c(self, _CachedState state, int card) noexcept
    cdef int _card_rank_c(self, _CachedState state, int card) noexcept
    cdef int _num_c(self, _CachedState state, int card) noexcept
    cdef int _play_action_c(self, int slot) noexcept
    cdef int _discard_action_c(self, int slot) noexcept
    cdef bint _legal_card_action_c(self, _CachedState state, int action) noexcept
    cdef bint _legal_draw_action_c(self, _CachedState state, int action) noexcept
    cdef bint _can_play_card_c(self, _CachedState state, int player, int card) noexcept
    cdef bint _has_numeric_c(self, _CachedState state, int player, int color) noexcept
    cdef int _opened_colors_c(self, _CachedState state, int player) noexcept
    cdef int _first_legal_card_c(self, _CachedState state) noexcept
    cdef int _first_legal_draw_c(self, _CachedState state) noexcept
    cdef int _act_card_c(self, _CachedState state, object derived) except -1
    cdef int _act_draw_c(self, _CachedState state, object derived) except -1
    cdef int _best_handshake_play_c(
        self, _CachedState state, int player, object derived, int deck_left
    ) except -2
    cdef int _best_number_play_c(
        self, _CachedState state, int player, object derived, int deck_left
    ) except -2
    cdef double _started_expedition_play_value_c(
        self, _CachedState state, int player, int card, object derived, int deck_left
    ) except *
    cdef bint _should_open_expedition_c(
        self, _CachedState state, int player, int color, int opening_card, object derived, int deck_left
    ) except *
    cdef double _opening_plan_value_c(
        self, _CachedState state, int player, int color, int opening_card, object derived, int deck_left
    ) except *
    cdef double _open_expedition_value_c(
        self, _CachedState state, int player, int color, int opening_card, object derived, int deck_left
    ) except *
    cdef int _best_forced_open_c(
        self, _CachedState state, int player, object derived, int deck_left
    ) except -2
    cdef int _best_discard_c(self, _CachedState state, int player, object derived) except -2
    cdef double _visible_draw_value_c(
        self, _CachedState state, int player, int card, object derived
    ) except *
    cdef double _visible_open_support_value_c(
        self, _CachedState state, int player, int card, object derived
    ) except *
    cdef bint _visible_number_can_help_open_c(
        self, _CachedState state, int player, int card, object derived
    ) except *
    cdef double _deck_draw_value_c(self, _CachedState state, object derived) except *
    cdef double _card_value_for_me_c(
        self, _CachedState state, int player, int card, object derived
    ) except *
    cdef double _card_value_for_opponent_c(
        self, _CachedState state, int opponent, int card, object derived
    ) except *
    cdef double _color_commitment_c(
        self, _CachedState state, int player, int color, object derived
    ) except *
    cdef double _public_color_commitment_for_opponent_c(
        self, _CachedState state, int opponent, int color, object derived
    ) except *
    cdef double _bonus_potential_c(
        self,
        _CachedState state,
        int player,
        int color,
        int extra_cards,
        object derived,
        int committed_cards,
        int exclude_card,
    ) except *
    cdef double _new_color_open_penalty_c(self, int opened_colors) noexcept
    cdef double _late_penalty_c(self, object derived, int deck_left) except *
