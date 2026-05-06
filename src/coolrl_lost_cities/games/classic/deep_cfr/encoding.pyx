# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Deterministic information-state encoding for Deep CFR."""

from coolrl_lost_cities.games.classic.game cimport GameState


cdef int DERIVED_PLAYABILITY_PER_COLOR = 19
cdef int DERIVED_PLAYABILITY_COMMON = 3
cdef int SLOT_AWARE_PLAYABILITY_PER_SLOT = 12


cdef int _base_input_dim_c(GameState state) noexcept:
    cdef int action_size = 2 * state.hand_size + 1 + state.n_colors
    cdef int card_type_size = state.n_colors * (state.n_ranks + 1)
    return (
        5
        + state.hand_size * 3
        + 2 * state.n_colors * 4
        + state.n_colors * 4
        + card_type_size
        + 3
        + 1
        + state.n_colors + 1
        + action_size
    )


cdef int input_dim_c(GameState state) noexcept:
    return _input_dim_with_flags_c(state, False, False)


cdef int _input_dim_with_flags_c(GameState state, bint derived_playability, bint slot_aware_playability) noexcept:
    cdef int total = _base_input_dim_c(state)
    if derived_playability:
        total += state.n_colors * DERIVED_PLAYABILITY_PER_COLOR + DERIVED_PLAYABILITY_COMMON
    if slot_aware_playability:
        total += state.hand_size * SLOT_AWARE_PLAYABILITY_PER_SLOT
    return total


cdef int _numeric_value_c(GameState state, int rank) noexcept:
    if rank == 0:
        return 0
    return state.min_rank + rank - 1


cdef float _max_numeric_sum_c(GameState state) noexcept:
    return <float>(state.n_ranks * (2 * state.min_rank + state.n_ranks - 1)) / 2.0


cdef float _max_score_estimate_c(GameState state) noexcept:
    cdef float max_numeric_sum = _max_numeric_sum_c(state)
    cdef float break_even = <float>abs(state.expedition_penalty)
    cdef float estimate = (max_numeric_sum - break_even) * <float>(state.n_handshakes + 1)
    if estimate < 1.0:
        return 1.0
    return estimate


cdef void _color_playability_summary_c(
    GameState state,
    int player,
    int color,
    int* is_unopened,
    int* has_only_wagers_opened,
    int* current_numeric_sum,
    int* current_wager_count,
    int* current_expedition_len,
    int* last_numeric_rank,
    int* hand_count,
    int* hand_wager_count,
    int* playable_hand_wager_count,
    int* playable_hand_numeric_sum,
    int* playable_hand_numeric_count,
    int* dead_hand_numeric_count,
    int* dead_hand_numeric_sum,
    int* recoverable_margin_no_bonus,
    int* recoverable_score_no_bonus,
    int* min_needed_to_break_even,
    int* discard_top_playable_flag,
    int* discard_top_playable_value,
    int* unknown_remaining_count,
    int* has_bonus_path,
    int* cards_needed_for_bonus,
) noexcept:
    cdef int cache_index = state._expedition_len_index(player, color)
    cdef int opponent_index = state._expedition_len_index(1 - player, color)
    cdef int slot
    cdef int card
    cdef int rank
    cdef int length = state.expedition_lens[cache_index]
    cdef int projected_numeric_sum
    cdef int projected_wager_count
    cdef int projected_len
    cdef int break_even = abs(state.expedition_penalty)
    cdef int known_color_count
    cdef int top_card

    current_numeric_sum[0] = state.numeric_sums[cache_index]
    current_wager_count[0] = state.handshake_counts[cache_index]
    current_expedition_len[0] = length
    last_numeric_rank[0] = state.last_numeric_ranks[cache_index]
    is_unopened[0] = 1 if length == 0 else 0
    has_only_wagers_opened[0] = 1 if length > 0 and last_numeric_rank[0] == 0 else 0

    hand_count[0] = 0
    hand_wager_count[0] = 0
    playable_hand_wager_count[0] = 0
    playable_hand_numeric_sum[0] = 0
    playable_hand_numeric_count[0] = 0
    dead_hand_numeric_count[0] = 0
    dead_hand_numeric_sum[0] = 0

    for slot in range(state.hand_lens[player]):
        card = state.hand_cards[state._hand_index(player, slot)]
        if state._card_color(card) != color:
            continue
        rank = state._card_rank(card)
        hand_count[0] += 1
        if rank == 0:
            hand_wager_count[0] += 1
            if last_numeric_rank[0] == 0:
                playable_hand_wager_count[0] += 1
        elif rank > last_numeric_rank[0]:
            playable_hand_numeric_count[0] += 1
            playable_hand_numeric_sum[0] += _numeric_value_c(state, rank)
        else:
            dead_hand_numeric_count[0] += 1
            dead_hand_numeric_sum[0] += _numeric_value_c(state, rank)

    projected_numeric_sum = current_numeric_sum[0] + playable_hand_numeric_sum[0]
    projected_wager_count = current_wager_count[0] + playable_hand_wager_count[0]
    recoverable_margin_no_bonus[0] = projected_numeric_sum - break_even
    recoverable_score_no_bonus[0] = recoverable_margin_no_bonus[0] * (projected_wager_count + 1)
    min_needed_to_break_even[0] = max(0, break_even - projected_numeric_sum)
    projected_len = length + playable_hand_numeric_count[0] + playable_hand_wager_count[0]
    has_bonus_path[0] = 1 if projected_len >= state.bonus_threshold else 0
    cards_needed_for_bonus[0] = max(0, state.bonus_threshold - projected_len)

    discard_top_playable_flag[0] = 0
    discard_top_playable_value[0] = 0
    if state.discard_lens[color] > 0:
        top_card = state.discard_cards[state._discard_index(color, state.discard_lens[color] - 1)]
        rank = state._card_rank(top_card)
        if state._card_color(top_card) == color and rank > 0 and rank > last_numeric_rank[0]:
            discard_top_playable_flag[0] = 1
            discard_top_playable_value[0] = _numeric_value_c(state, rank)

    known_color_count = hand_count[0]
    known_color_count += state.expedition_lens[cache_index]
    known_color_count += state.expedition_lens[opponent_index]
    known_color_count += state.discard_lens[color]
    unknown_remaining_count[0] = max(0, state.cards_per_color - known_color_count)


cdef int _append_derived_playability_features_c(GameState state, int player, float* out, int idx) noexcept:
    cdef float max_numeric_sum = _max_numeric_sum_c(state)
    cdef float max_cards_per_color = <float>max(1, state.cards_per_color)
    cdef float max_wagers = <float>max(1, state.n_handshakes)
    cdef float max_score_estimate = _max_score_estimate_c(state)
    cdef int color
    cdef int is_unopened, has_only_wagers_opened, current_numeric_sum, current_wager_count
    cdef int current_expedition_len, last_numeric_rank, hand_count, hand_wager_count
    cdef int playable_hand_wager_count, playable_hand_numeric_sum, playable_hand_numeric_count
    cdef int dead_hand_numeric_count, dead_hand_numeric_sum, recoverable_margin_no_bonus
    cdef int recoverable_score_no_bonus, min_needed_to_break_even, discard_top_playable_flag
    cdef int discard_top_playable_value, unknown_remaining_count, has_bonus_path
    cdef int cards_needed_for_bonus

    for color in range(state.n_colors):
        _color_playability_summary_c(
            state, player, color,
            &is_unopened, &has_only_wagers_opened, &current_numeric_sum,
            &current_wager_count, &current_expedition_len, &last_numeric_rank,
            &hand_count, &hand_wager_count, &playable_hand_wager_count,
            &playable_hand_numeric_sum, &playable_hand_numeric_count,
            &dead_hand_numeric_count, &dead_hand_numeric_sum, &recoverable_margin_no_bonus,
            &recoverable_score_no_bonus, &min_needed_to_break_even,
            &discard_top_playable_flag, &discard_top_playable_value,
            &unknown_remaining_count, &has_bonus_path, &cards_needed_for_bonus,
        )
        out[idx] = <float>is_unopened
        out[idx + 1] = <float>has_only_wagers_opened
        out[idx + 2] = <float>current_numeric_sum / max_numeric_sum
        out[idx + 3] = <float>current_wager_count / max_wagers
        out[idx + 4] = <float>current_expedition_len / max_cards_per_color
        out[idx + 5] = <float>last_numeric_rank / max_numeric_sum
        out[idx + 6] = <float>hand_count / max_cards_per_color
        out[idx + 7] = <float>hand_wager_count / max_wagers
        out[idx + 8] = <float>playable_hand_numeric_sum / max_numeric_sum
        out[idx + 9] = <float>playable_hand_numeric_count / max_cards_per_color
        out[idx + 10] = <float>dead_hand_numeric_count / max_cards_per_color
        out[idx + 11] = <float>dead_hand_numeric_sum / max_numeric_sum
        out[idx + 12] = <float>recoverable_margin_no_bonus / max_numeric_sum
        out[idx + 13] = <float>recoverable_score_no_bonus / max_score_estimate
        out[idx + 14] = <float>min_needed_to_break_even / max_numeric_sum
        out[idx + 15] = <float>discard_top_playable_flag
        out[idx + 16] = <float>discard_top_playable_value / max_numeric_sum
        out[idx + 17] = <float>unknown_remaining_count / max_cards_per_color
        out[idx + 18] = <float>cards_needed_for_bonus / max_cards_per_color
        idx += DERIVED_PLAYABILITY_PER_COLOR

    out[idx] = <float>state.deck_len / <float>max(1, state.total_cards)
    out[idx + 1] = <float>state.turn_count / <float>max(1, 2 * state.total_cards)
    out[idx + 2] = <float>state.deck_len / <float>max(1, 2 * state.total_cards)
    return idx + DERIVED_PLAYABILITY_COMMON


cdef int _append_slot_aware_playability_features_c(GameState state, int player, float* out, int idx) noexcept:
    cdef float max_numeric_sum = _max_numeric_sum_c(state)
    cdef float max_score_estimate = _max_score_estimate_c(state)
    cdef int slot
    cdef int card
    cdef int color
    cdef int rank
    cdef int is_unopened, has_only_wagers_opened, current_numeric_sum, current_wager_count
    cdef int current_expedition_len, last_numeric_rank, hand_count, hand_wager_count
    cdef int playable_hand_wager_count, playable_hand_numeric_sum, playable_hand_numeric_count
    cdef int dead_hand_numeric_count, dead_hand_numeric_sum, recoverable_margin_no_bonus
    cdef int recoverable_score_no_bonus, min_needed_to_break_even, discard_top_playable_flag
    cdef int discard_top_playable_value, unknown_remaining_count, has_bonus_path
    cdef int cards_needed_for_bonus
    cdef bint legal_play
    cdef bint has_numeric_started
    cdef bint is_numeric
    cdef bint is_wager
    cdef bint would_start_color_commitment
    cdef bint is_playable_to_existing
    cdef bint is_dead_numeric
    cdef bint is_wager_before_numeric
    cdef bint is_numeric_open
    cdef bint is_wager_first_open
    cdef bint is_bad_open_candidate
    cdef bint is_safe_continuation
    cdef float open_risk_score

    for slot in range(state.hand_size):
        if slot >= state.hand_lens[player]:
            for color in range(SLOT_AWARE_PLAYABILITY_PER_SLOT):
                out[idx + color] = 0.0
            idx += SLOT_AWARE_PLAYABILITY_PER_SLOT
            continue
        card = state.hand_cards[state._hand_index(player, slot)]
        color = state._card_color(card)
        rank = state._card_rank(card)
        _color_playability_summary_c(
            state, player, color,
            &is_unopened, &has_only_wagers_opened, &current_numeric_sum,
            &current_wager_count, &current_expedition_len, &last_numeric_rank,
            &hand_count, &hand_wager_count, &playable_hand_wager_count,
            &playable_hand_numeric_sum, &playable_hand_numeric_count,
            &dead_hand_numeric_count, &dead_hand_numeric_sum, &recoverable_margin_no_bonus,
            &recoverable_score_no_bonus, &min_needed_to_break_even,
            &discard_top_playable_flag, &discard_top_playable_value,
            &unknown_remaining_count, &has_bonus_path, &cards_needed_for_bonus,
        )
        legal_play = state._can_play_encoded_card_c(player, card)
        has_numeric_started = last_numeric_rank > 0
        is_numeric = rank > 0
        is_wager = rank == 0
        would_start_color_commitment = legal_play and not has_numeric_started
        is_numeric_open = would_start_color_commitment and is_numeric
        is_wager_first_open = would_start_color_commitment and is_wager and is_unopened
        is_playable_to_existing = legal_play and has_numeric_started
        is_dead_numeric = is_numeric and not legal_play and rank <= last_numeric_rank
        is_wager_before_numeric = is_wager and legal_play and not has_numeric_started
        is_bad_open_candidate = would_start_color_commitment and recoverable_score_no_bonus < 0
        open_risk_score = min(0.0, <float>recoverable_score_no_bonus) if would_start_color_commitment else 0.0
        is_safe_continuation = (not would_start_color_commitment) and is_playable_to_existing

        out[idx] = <float>recoverable_score_no_bonus / max_score_estimate
        out[idx + 1] = <float>recoverable_margin_no_bonus / max_numeric_sum
        out[idx + 2] = <float>would_start_color_commitment
        out[idx + 3] = <float>is_numeric_open
        out[idx + 4] = <float>is_wager_first_open
        out[idx + 5] = <float>is_playable_to_existing
        out[idx + 6] = <float>is_dead_numeric
        out[idx + 7] = <float>is_wager_before_numeric
        out[idx + 8] = <float>has_bonus_path
        out[idx + 9] = <float>is_bad_open_candidate
        out[idx + 10] = open_risk_score / max_score_estimate
        out[idx + 11] = <float>is_safe_continuation
        idx += SLOT_AWARE_PLAYABILITY_PER_SLOT
    return idx


cdef int encode_info_state_c(GameState state, int player, float* out) except -1:
    return _encode_info_state_with_flags_c(state, player, out, False, False)


cdef int _encode_info_state_with_flags_c(
    GameState state,
    int player,
    float* out,
    bint derived_playability,
    bint slot_aware_playability,
) except -1:
    cdef int idx = 0
    cdef int slot
    cdef int card
    cdef int color
    cdef int rank
    cdef int action_size = 2 * state.hand_size + 1 + state.n_colors
    cdef int card_type_size = state.n_colors * (state.n_ranks + 1)
    cdef int public_base
    cdef int player_index
    cdef int length
    cdef int top_card
    cdef int card_index
    cdef int max_expedition_len = state.n_ranks + state.n_handshakes
    cdef float score_denom = <float>(
        max(1, state.bonus_amount + (state.n_ranks * (state.n_ranks + 1)) * (state.n_handshakes + 1))
    )
    cdef int action_count
    cdef int actions[64]
    cdef int i

    if player < 0 or player > 1:
        raise ValueError(f"invalid player: {player}")
    if action_size > 64:
        raise ValueError("action_size exceeds fixed encoding action buffer")

    out[idx] = 1.0 if state.phase_id == 0 else 0.0
    idx += 1
    out[idx] = 1.0 if state.phase_id == 1 else 0.0
    idx += 1
    out[idx] = <float>state.current_player
    idx += 1
    out[idx] = <float>player
    idx += 1
    out[idx] = <float>state.deck_len / <float>state.total_cards
    idx += 1

    for slot in range(state.hand_size):
        if slot < state.hand_lens[player]:
            card = state.hand_cards[state._hand_index(player, slot)]
            color = state._card_color(card)
            rank = state._card_rank(card)
            out[idx] = 1.0
            out[idx + 1] = <float>(color + 1) / <float>state.n_colors
            out[idx + 2] = <float>rank / <float>state.n_ranks
        else:
            out[idx] = 0.0
            out[idx + 1] = 0.0
            out[idx + 2] = 0.0
        idx += 3

    for player_index in range(2):
        for color in range(state.n_colors):
            card_index = state._expedition_len_index(player_index, color)
            length = state.expedition_lens[card_index]
            out[idx] = <float>length / <float>max_expedition_len
            out[idx + 1] = <float>state.last_numeric_ranks[card_index] / <float>state.n_ranks
            out[idx + 2] = <float>state.handshake_counts[card_index] / <float>max(1, state.n_handshakes)
            out[idx + 3] = <float>state.expedition_scores[card_index] / score_denom
            idx += 4

    for color in range(state.n_colors):
        length = state.discard_lens[color]
        out[idx] = <float>length / <float>max_expedition_len
        if length > 0:
            top_card = state.discard_cards[state._discard_index(color, length - 1)]
            out[idx + 1] = 1.0
            out[idx + 2] = <float>(state._card_color(top_card) + 1) / <float>state.n_colors
            out[idx + 3] = <float>state._card_rank(top_card) / <float>state.n_ranks
        else:
            out[idx + 1] = 0.0
            out[idx + 2] = 0.0
            out[idx + 3] = 0.0
        idx += 4

    public_base = idx
    for i in range(card_type_size):
        out[public_base + i] = 0.0
    for player_index in range(2):
        for color in range(state.n_colors):
            card_index = state._expedition_len_index(player_index, color)
            for i in range(state.expedition_lens[card_index]):
                card = state.expedition_cards[state._expedition_index(player_index, color, i)]
                out[public_base + state._card_color(card) * (state.n_ranks + 1) + state._card_rank(card)] += 1.0
    for color in range(state.n_colors):
        for i in range(state.discard_lens[color]):
            card = state.discard_cards[state._discard_index(color, i)]
            out[public_base + state._card_color(card) * (state.n_ranks + 1) + state._card_rank(card)] += 1.0
    idx += card_type_size

    out[idx] = <float>state.total_scores[player] / score_denom
    out[idx + 1] = <float>state.total_scores[1 - player] / score_denom
    out[idx + 2] = <float>(state.total_scores[player] - state.total_scores[1 - player]) / score_denom
    idx += 3

    out[idx] = <float>state.turn_count / <float>max(1, state.total_cards * 2)
    idx += 1

    for i in range(state.n_colors + 1):
        out[idx + i] = 0.0
    if state.pending_discarded_color < 0:
        out[idx + state.n_colors] = 1.0
    else:
        out[idx + state.pending_discarded_color] = 1.0
    idx += state.n_colors + 1

    for i in range(action_size):
        out[idx + i] = 0.0
    action_count = state._unified_legal_actions_c(actions)
    for i in range(action_count):
        out[idx + actions[i]] = 1.0
    idx += action_size
    if derived_playability:
        idx = _append_derived_playability_features_c(state, player, out, idx)
    if slot_aware_playability:
        idx = _append_slot_aware_playability_features_c(state, player, out, idx)
    return idx


def input_dim(GameState state, encoding=None) -> int:
    cdef bint derived_playability = False
    cdef bint slot_aware_playability = False
    if encoding is not None:
        derived_playability = bool(encoding.derived_playability)
        slot_aware_playability = bool(encoding.slot_aware_playability)
    return _input_dim_with_flags_c(state, derived_playability, slot_aware_playability)


def encode_info_state(GameState state, int player, encoding=None):
    cdef float[::1] out_view
    cdef bint derived_playability = False
    cdef bint slot_aware_playability = False
    import numpy as np

    if encoding is not None:
        derived_playability = bool(encoding.derived_playability)
        slot_aware_playability = bool(encoding.slot_aware_playability)
    out = np.empty(_input_dim_with_flags_c(state, derived_playability, slot_aware_playability), dtype=np.float32)
    out_view = out
    _encode_info_state_with_flags_c(state, player, &out_view[0], derived_playability, slot_aware_playability)
    return out
