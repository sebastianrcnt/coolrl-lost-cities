from coolrl_lost_cities.games.classic.game cimport GameState


cdef float random_rollout_value_c(
    GameState state,
    int player,
    unsigned int seed,
    int max_steps,
) except *
