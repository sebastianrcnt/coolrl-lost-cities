# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Small Cython math helpers for future Deep CFR traversal."""


cdef int regret_matching_c(
    const float* advantages,
    const unsigned char* legal,
    int n,
    float epsilon,
    float* out_policy,
) noexcept:
    cdef int i
    cdef int legal_count = 0
    cdef float positive
    cdef float total = 0.0
    if n <= 0:
        return 0
    for i in range(n):
        if legal[i] != 0:
            legal_count += 1
            positive = advantages[i] if advantages[i] > 0.0 else 0.0
            out_policy[i] = positive
            total += positive
        else:
            out_policy[i] = 0.0
    if legal_count <= 0:
        return 0
    if total > epsilon:
        for i in range(n):
            out_policy[i] = out_policy[i] / total
        return legal_count
    for i in range(n):
        out_policy[i] = 1.0 / legal_count if legal[i] != 0 else 0.0
    return legal_count


cdef int normalize_legal_policy_c(
    const float* weights,
    const unsigned char* legal,
    int n,
    float* out_policy,
) noexcept:
    cdef int i
    cdef int legal_count = 0
    cdef float value
    cdef float total = 0.0
    if n <= 0:
        return 0
    for i in range(n):
        if legal[i] != 0:
            legal_count += 1
            value = weights[i] if weights[i] > 0.0 else 0.0
            out_policy[i] = value
            total += value
        else:
            out_policy[i] = 0.0
    if legal_count <= 0:
        return 0
    if total > 0.0:
        for i in range(n):
            out_policy[i] = out_policy[i] / total
        return legal_count
    for i in range(n):
        out_policy[i] = 1.0 / legal_count if legal[i] != 0 else 0.0
    return legal_count


cdef int sample_policy_c(
    const float* policy,
    int n,
    double random_value,
) noexcept:
    cdef int i
    cdef int fallback = -1
    cdef double cumulative = 0.0
    cdef double r = random_value
    if n <= 0:
        return -1
    if r < 0.0:
        r = 0.0
    elif r >= 1.0:
        r = 0.9999999999999999
    for i in range(n):
        if policy[i] > 0.0:
            fallback = i
            cumulative += policy[i]
            if r < cumulative:
                return i
    return fallback


def regret_matching(advantages, legal_mask, float epsilon=1.0e-8):
    cdef float[::1] adv_view
    cdef unsigned char[::1] legal_view
    cdef float[::1] out_view
    import numpy as np

    adv = np.ascontiguousarray(advantages, dtype=np.float32)
    legal = np.ascontiguousarray(legal_mask, dtype=np.uint8)
    if adv.ndim != 1 or legal.ndim != 1:
        raise ValueError("advantages and legal_mask must be one-dimensional")
    if adv.shape[0] != legal.shape[0]:
        raise ValueError("advantages and legal_mask must have the same length")
    out = np.empty_like(adv)
    if adv.shape[0] == 0:
        return out
    adv_view = adv
    legal_view = legal
    out_view = out
    regret_matching_c(&adv_view[0], &legal_view[0], adv.shape[0], epsilon, &out_view[0])
    return out


def normalize_legal_policy(weights, legal_mask):
    cdef float[::1] values_view
    cdef unsigned char[::1] legal_view
    cdef float[::1] out_view
    import numpy as np

    values = np.ascontiguousarray(weights, dtype=np.float32)
    legal = np.ascontiguousarray(legal_mask, dtype=np.uint8)
    if values.ndim != 1 or legal.ndim != 1:
        raise ValueError("weights and legal_mask must be one-dimensional")
    if values.shape[0] != legal.shape[0]:
        raise ValueError("weights and legal_mask must have the same length")
    out = np.empty_like(values)
    if values.shape[0] == 0:
        return out
    values_view = values
    legal_view = legal
    out_view = out
    normalize_legal_policy_c(&values_view[0], &legal_view[0], values.shape[0], &out_view[0])
    return out


def sample_policy(policy, double random_value):
    cdef float[::1] values_view
    import numpy as np

    values = np.ascontiguousarray(policy, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("policy must be one-dimensional")
    if values.shape[0] == 0:
        return -1
    values_view = values
    return sample_policy_c(&values_view[0], values.shape[0], random_value)
