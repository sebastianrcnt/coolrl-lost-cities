# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Minimal deterministic information-state encoding for Deep CFR scaffolding."""

from coolrl_lost_cities.games.classic.game cimport GameState


cdef int input_dim_c(GameState state) noexcept:
    return 5 + state.hand_size * 3 + 2 * state.hand_size + 1 + state.n_colors


cdef int encode_info_state_c(GameState state, int player, float* out) except -1:
    cdef int idx = 0
    cdef int slot
    cdef int card
    cdef int color
    cdef int rank
    cdef int action_size = 2 * state.hand_size + 1 + state.n_colors
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
