cdef int regret_matching_c(
    const float* advantages,
    const unsigned char* legal,
    int n,
    float epsilon,
    float* out_policy,
) noexcept

cdef int normalize_legal_policy_c(
    const float* weights,
    const unsigned char* legal,
    int n,
    float* out_policy,
) noexcept

cdef int sample_policy_c(
    const float* policy,
    int n,
    double random_value,
) noexcept

