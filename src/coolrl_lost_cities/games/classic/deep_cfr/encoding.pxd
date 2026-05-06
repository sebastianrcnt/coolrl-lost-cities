from coolrl_lost_cities.games.classic.game cimport GameState


cdef int input_dim_c(GameState state) noexcept
cdef int encode_info_state_c(GameState state, int player, float* out) except -1

