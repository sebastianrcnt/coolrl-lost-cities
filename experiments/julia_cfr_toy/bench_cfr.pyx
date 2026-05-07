# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as cnp

ctypedef unsigned long long uint64_t

cdef int MAX_DEPTH = 10
cdef int BRANCHING = 4
cdef int HALF_DEPTH = 5
cdef int TRAVERSALS_PER_ITER = 1000
cdef int MEASURE_ITERATIONS = 100
cdef int NUM_INTERNAL_NODES = 349525
cdef uint64_t SEED = <uint64_t>0x00000000013579bd
cdef double TWO_POW_53 = 9007199254740992.0


cdef inline uint64_t splitmix64(uint64_t x) noexcept nogil:
    x = x + <uint64_t>0x9e3779b97f4a7c15
    x = (x ^ (x >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * <uint64_t>0x94d049bb133111eb
    return x ^ (x >> 31)


cdef inline uint64_t hash_key(int node_id, int depth, int action, int traversal) noexcept nogil:
    cdef uint64_t x = SEED
    x = x ^ (<uint64_t>node_id * <uint64_t>0xd6e8feb86659fd93)
    x = x ^ (<uint64_t>(depth + 1) * <uint64_t>0xa5a3564e27f886d9)
    x = x ^ (<uint64_t>(action + 11) * <uint64_t>0x9e3779b185ebca87)
    x = x ^ (<uint64_t>(traversal + 17) * <uint64_t>0xc2b2ae3d27d4eb4f)
    return splitmix64(x)


cdef inline double unit_value(uint64_t key) noexcept nogil:
    return <double>(key >> 11) / TWO_POW_53


cdef inline double terminal_value(int node_id, int depth, int action, int traversal) noexcept nogil:
    return 2.0 * unit_value(hash_key(node_id, depth, action, traversal)) - 1.0


cdef inline bint is_legal(int depth, int action) noexcept nogil:
    return depth < HALF_DEPTH or action != BRANCHING


cdef class CFRTree:
    cdef cnp.ndarray regret_arr
    cdef double[:, ::1] regret

    def __cinit__(self):
        self.regret_arr = np.zeros((NUM_INTERNAL_NODES, BRANCHING), dtype=np.float64)
        self.regret = self.regret_arr

    cdef double traverse(self, int node_id, int depth, int traversal) noexcept nogil:
        cdef double positive_sum = 0.0
        cdef int legal_count = 0
        cdef int action
        cdef int sampled_action = 1
        cdef int child_id
        cdef double positive
        cdef double uniform
        cdef double cumulative = 0.0
        cdef double r
        cdef double sampled_value
        cdef double expected = 0.0
        cdef double strategy[4]
        cdef double counterfactual[4]

        for action in range(1, BRANCHING + 1):
            strategy[action - 1] = 0.0
            counterfactual[action - 1] = 0.0
            if is_legal(depth, action):
                legal_count += 1
                positive = self.regret[node_id - 1, action - 1]
                if positive < 0.0:
                    positive = 0.0
                positive_sum += positive

        if positive_sum > 0.0:
            for action in range(1, BRANCHING + 1):
                if is_legal(depth, action):
                    positive = self.regret[node_id - 1, action - 1]
                    if positive < 0.0:
                        positive = 0.0
                    strategy[action - 1] = positive / positive_sum
        else:
            uniform = 1.0 / legal_count
            for action in range(1, BRANCHING + 1):
                if is_legal(depth, action):
                    strategy[action - 1] = uniform

        r = unit_value(hash_key(node_id, depth, 97, traversal))
        for action in range(1, BRANCHING + 1):
            cumulative += strategy[action - 1]
            if r <= cumulative:
                sampled_action = action
                break

        child_id = (node_id - 1) * BRANCHING + sampled_action + 1
        if depth + 1 >= MAX_DEPTH:
            sampled_value = terminal_value(node_id, depth, sampled_action, traversal)
        else:
            sampled_value = self.traverse(child_id, depth + 1, traversal)

        for action in range(1, BRANCHING + 1):
            if is_legal(depth, action):
                if action == sampled_action:
                    counterfactual[action - 1] = sampled_value
                else:
                    counterfactual[action - 1] = terminal_value(node_id, depth, action, traversal)

        for action in range(1, BRANCHING + 1):
            if is_legal(depth, action):
                self.regret[node_id - 1, action - 1] += counterfactual[action - 1] - sampled_value

        for action in range(1, BRANCHING + 1):
            expected += strategy[action - 1] * counterfactual[action - 1]
        return expected

    cdef double run_iteration(self, int iteration) noexcept nogil:
        cdef int base = (iteration - 1) * TRAVERSALS_PER_ITER
        cdef int offset
        cdef double value = 0.0
        for offset in range(1, TRAVERSALS_PER_ITER + 1):
            value += self.traverse(1, 0, base + offset)
        return value

    cpdef run_iterations(self, int iterations):
        cdef int iteration
        cdef double total_value = 0.0
        with nogil:
            for iteration in range(1, iterations + 1):
                total_value += self.run_iteration(iteration)
        return total_value

    cpdef root_regret(self):
        return [
            float(self.regret[0, 0]),
            float(self.regret[0, 1]),
            float(self.regret[0, 2]),
            float(self.regret[0, 3]),
        ]


def run_benchmark():
    warmup_tree = CFRTree()
    warmup_tree.run_iterations(1)

    tree = CFRTree()
    tree.run_iterations(MEASURE_ITERATIONS)
    return {
        "lang": "Cython",
        "iterations": MEASURE_ITERATIONS,
        "traversals_per_iter": TRAVERSALS_PER_ITER,
        "root_regret": tree.root_regret(),
    }
