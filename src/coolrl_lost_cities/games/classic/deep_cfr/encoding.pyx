# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Deterministic information-state encoding for Deep CFR."""

from coolrl_lost_cities.games.classic.game cimport GameState


cdef int input_dim_c(GameState state) noexcept:
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


cdef int encode_info_state_c(GameState state, int player, float* out) except -1:
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
    return idx


def input_dim(GameState state) -> int:
    return input_dim_c(state)


def encode_info_state(GameState state, int player):
    cdef float[::1] out_view
    import numpy as np

    out = np.empty(input_dim_c(state), dtype=np.float32)
    out_view = out
    encode_info_state_c(state, player, &out_view[0])
    return out
