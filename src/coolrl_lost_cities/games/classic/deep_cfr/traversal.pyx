# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython traversal primitives for Deep CFR smoke runs."""

from coolrl_lost_cities.games.classic.game cimport GameState


cdef unsigned int _next_u32(unsigned int* state) noexcept:
    state[0] = state[0] * 1664525 + 1013904223
    return state[0]


cdef float random_rollout_value_c(
    GameState state,
    int player,
    unsigned int seed,
    int max_steps,
) except *:
    cdef int actions[64]
    cdef int count
    cdef int action
    cdef int depth = 0
    cdef unsigned int rng = seed if seed != 0 else 1
    cdef float value

    if player < 0 or player > 1:
        raise ValueError(f"invalid player: {player}")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    while not state.terminal and depth < max_steps:
        count = state._legal_actions_c(actions)
        if count <= 0:
            break
        action = actions[_next_u32(&rng) % <unsigned int>count]
        state._push_action_c(action)
        depth += 1

    value = <float>(state.total_scores[player] - state.total_scores[1 - player])

    while depth > 0:
        state._pop_action_c()
        depth -= 1

    return value


def random_rollout_value(GameState state, int player, unsigned int seed=1, int max_steps=512):
    return random_rollout_value_c(state, player, seed, max_steps)


def root_action_values(
    GameState state,
    int player,
    unsigned int seed=1,
    int rollouts_per_action=1,
    int max_steps=512,
):
    cdef int action_size = 2 * state.hand_size + 1 + state.n_colors
    cdef int actions[64]
    cdef int count
    cdef int i
    cdef int rollout
    cdef int unified_action
    cdef int local_action
    cdef float total
    cdef float[::1] values_view
    cdef unsigned char[::1] legal_view
    import numpy as np

    if player < 0 or player > 1:
        raise ValueError(f"invalid player: {player}")
    if action_size > 64:
        raise ValueError("action_size exceeds fixed traversal action buffer")
    if rollouts_per_action <= 0:
        raise ValueError("rollouts_per_action must be positive")

    values = np.zeros(action_size, dtype=np.float32)
    legal = np.zeros(action_size, dtype=np.uint8)
    values_view = values
    legal_view = legal

    count = state._unified_legal_actions_c(actions)
    for i in range(count):
        unified_action = actions[i]
        local_action = state.from_unified_action(unified_action)
        legal_view[unified_action] = 1
        total = 0.0
        for rollout in range(rollouts_per_action):
            state._push_action_c(local_action)
            total += random_rollout_value_c(
                state,
                player,
                seed + <unsigned int>(i * rollouts_per_action + rollout + 1),
                max_steps,
            )
            state._pop_action_c()
        values_view[unified_action] = total / <float>rollouts_per_action

    return values, legal
