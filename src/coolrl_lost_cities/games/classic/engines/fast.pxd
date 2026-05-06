ctypedef struct UndoRecord:
    int phase_id
    int player
    int action_id
    int pending_before
    bint terminal_before
    int turn_count_before
    int slot
    int play
    int card
    int color
    int last_numeric_before
    int handshake_count_before
    int numeric_sum_before
    int expedition_score_before
    int total_score_before


cdef class FastGameState:
    cdef public object config
    cdef int n_colors
    cdef int n_ranks
    cdef int min_rank
    cdef int n_handshakes
    cdef int hand_size
    cdef int expedition_penalty
    cdef int bonus_threshold
    cdef int bonus_amount
    cdef int total_cards
    cdef int cards_per_color
    cdef int stride

    cdef int* deck
    cdef int deck_len
    cdef int* hands
    cdef int hand_lens[2]
    cdef int* expeditions
    cdef int* expedition_lens
    cdef int* discards
    cdef int* discard_lens
    cdef int* last_numeric_ranks
    cdef int* handshake_counts
    cdef int* numeric_sums
    cdef int* expedition_scores
    cdef int total_scores[2]
    cdef UndoRecord* undo_stack
    cdef int undo_stack_len
    cdef int undo_stack_capacity

    cdef public int current_player
    cdef int phase_id
    cdef public int pending_discarded_color
    cdef public int turn_count
    cdef public bint terminal

    cdef void _configure(self, object config) except *
    cdef void _clear(self) noexcept

    cpdef FastGameState clone(self)
    cpdef list legal_card_mask(self)
    cpdef list legal_draw_mask(self)
    cpdef list legal_mask(self)
    cpdef list unified_legal_mask(self)
    cpdef list legal_actions(self)
    cpdef list unified_legal_actions(self)
    cpdef int from_unified_action(self, int action_id)
    cpdef apply_action(self, int action_id)
    cpdef apply_unified_action(self, int action_id)
    cpdef object apply_action_with_undo(self, int action_id)
    cpdef object apply_unified_action_with_undo(self, int action_id)
    cpdef undo_action(self, object undo)
    cpdef int push_action(self, int action_id)
    cpdef int push_unified_action(self, int action_id)
    cpdef int pop_action(self)
    cpdef bint can_play_encoded_card(self, int player, int card)
    cpdef int last_numeric_rank(self, int player, int color)
    cpdef int expedition_score(self, int player, int color)
    cpdef int total_score(self, int player)
    cpdef int score_diff(self, int player=*)

    cdef bint _is_legal_action_c(self, int action_id) noexcept
    cdef int _legal_actions_c(self, int* out_actions) noexcept
    cdef int _unified_legal_actions_c(self, int* out_actions) noexcept
    cdef bint _can_play_encoded_card_c(self, int player, int card) noexcept
    cdef void _fill_undo_c(self, int action_id, UndoRecord* undo) noexcept
    cdef void _apply_action_with_undo_c(self, int action_id, UndoRecord* undo) except *
    cdef void _apply_action_unchecked_c(self, int action_id) except *
    cdef void _ensure_undo_capacity_c(self) except *
    cdef int _push_action_c(self, int action_id) except *
    cdef int _pop_action_c(self) except *
    cdef object _undo_to_tuple(self, UndoRecord* undo)
    cdef void _tuple_to_undo(self, object data, UndoRecord* undo) except *
    cdef void _apply_card_action(self, int action_id) except *
    cdef void _apply_draw_action(self, int action_id) except *
    cdef void _undo_action_c(self, UndoRecord* undo) except *
    cdef void _undo_card_action_c(self, UndoRecord* undo) except *
    cdef void _undo_draw_action_c(self, UndoRecord* undo) except *
    cdef void _recompute_score_caches(self) noexcept
    cdef int _score_from_summary_c(self, int length, int handshakes, int numeric_sum) noexcept
    cdef bint _has_any_legal_draw(self) noexcept
    cdef int _hand_index(self, int player, int slot)
    cdef int _expedition_len_index(self, int player, int color)
    cdef int _expedition_index(self, int player, int color, int index)
    cdef int _discard_index(self, int color, int index)
    cdef int _encode_card(self, int color, int rank)
    cdef int _card_color(self, int card)
    cdef int _card_rank(self, int card)
    cdef object _card_snapshot(self, int card)
