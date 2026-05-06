from coolrl_lost_cities.games.classic.game cimport GameState


cdef int input_dim_c(GameState state) noexcept
cdef int _input_dim_with_flags_c(
    GameState state,
    bint derived_playability,
    bint slot_aware_playability,
) noexcept
cdef int encode_info_state_c(GameState state, int player, float* out) except -1
cdef int _encode_info_state_with_flags_c(
    GameState state,
    int player,
    float* out,
    bint derived_playability,
    bint slot_aware_playability,
) except -1
