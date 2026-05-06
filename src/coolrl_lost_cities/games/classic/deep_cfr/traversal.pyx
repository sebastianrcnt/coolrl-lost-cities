# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython traversal engine and rollout primitives for Deep CFR."""

from libc.stdlib cimport free, malloc

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots import SafeHeuristicBot
from coolrl_lost_cities.games.classic.deep_cfr.cfr_math cimport regret_matching_c
from coolrl_lost_cities.games.classic.deep_cfr.encoding cimport (
    _encode_info_state_with_flags_c,
    _input_dim_with_flags_c,
)
from coolrl_lost_cities.games.classic.deep_cfr.memory import TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.traversal_stats import TraversalStats
from coolrl_lost_cities.games.classic.game cimport GameState


DEF MAX_ACTIONS = 64


cdef unsigned int _next_u32(unsigned int* state) noexcept:
    state[0] = state[0] * 1664525 + 1013904223
    return state[0]


cdef double _next_double(unsigned int* state) noexcept:
    return <double>_next_u32(state) / 4294967296.0


cdef int _sample_policy_from_actions_c(
    const float* policy,
    const int* actions,
    int count,
    double random_value,
) noexcept:
    cdef int i
    cdef int fallback = -1
    cdef double cumulative = 0.0
    cdef double r = random_value
    if count <= 0:
        return -1
    if r < 0.0:
        r = 0.0
    elif r >= 1.0:
        r = 0.9999999999999999
    for i in range(count):
        if policy[actions[i]] > 0.0:
            fallback = actions[i]
            cumulative += policy[actions[i]]
            if r < cumulative:
                return actions[i]
    return fallback


cdef int _depth_bucket_start(int depth, int width, int max_depth) noexcept:
    cdef int start = (depth // width) * width
    if start >= max_depth:
        return max_depth
    return start


cdef class CythonDeepCFRTraverser:
    cdef object advantage_networks
    cdef object advantage_samples
    cdef object strategy_samples
    cdef object device
    cdef object encoding
    cdef object league_advantage_networks
    cdef object safe_heuristic_rollout_bot
    cdef object safe_heuristic_opponent_bot
    cdef int action_size
    cdef int input_dim
    cdef float epsilon
    cdef int strategy_sample_interval
    cdef bint store_strategy_on_traverser_nodes
    cdef bint store_strategy_on_opponent_nodes
    cdef bint has_max_depth
    cdef int max_depth
    cdef bint has_max_nodes
    cdef int max_nodes
    cdef float outcome_sampling_epsilon
    cdef bint has_value_clip
    cdef float outcome_sampling_value_clip
    cdef bint unsampled_regret_zero
    cdef bint cutoff_random_rollout
    cdef int cutoff_rollouts
    cdef int cutoff_rollout_max_steps
    cdef int opponent_policy_id
    cdef float self_play_anchor_probability
    cdef float self_play_current_weight
    cdef float self_play_recent_weight
    cdef float self_play_older_weight
    cdef float self_play_anchor_weight
    cdef int self_play_recent_window
    cdef int endpoint_depth_bucket_width
    cdef int endpoint_depth_bucket_max
    cdef bint derived_playability
    cdef bint slot_aware_playability
    cdef unsigned int rng

    def __init__(
        self,
        object advantage_networks,
        *,
        object device,
        int action_size,
        object encoding=None,
        float epsilon=1.0e-8,
        int strategy_sample_interval=1,
        bint store_strategy_on_traverser_nodes=True,
        bint store_strategy_on_opponent_nodes=True,
        object max_depth=None,
        object max_nodes=None,
        float outcome_sampling_epsilon=0.0,
        object outcome_sampling_value_clip=None,
        str outcome_unsampled_regret="negative_node_value",
        str cutoff_value_mode="score_diff",
        int cutoff_rollouts=0,
        str cutoff_rollout_policy="random",
        int cutoff_rollout_max_steps=10000,
        str opponent_policy="network",
        object league_advantage_networks=None,
        float self_play_anchor_probability=0.0,
        float self_play_current_weight=0.5,
        float self_play_recent_weight=0.3,
        float self_play_older_weight=0.2,
        float self_play_anchor_weight=0.0,
        int self_play_recent_window=5,
        int endpoint_depth_bucket_width=10,
        int endpoint_depth_bucket_max=100,
        unsigned int seed=1,
    ):
        self.advantage_networks = advantage_networks
        self.advantage_samples = []
        self.strategy_samples = []
        self.device = device
        self.action_size = action_size
        if action_size > MAX_ACTIONS:
            raise ValueError("action_size exceeds fixed traversal action buffer")
        self.encoding = encoding
        self.derived_playability = False
        self.slot_aware_playability = False
        if encoding is not None:
            self.derived_playability = bool(encoding.derived_playability)
            self.slot_aware_playability = bool(encoding.slot_aware_playability)
        self.input_dim = -1
        self.epsilon = epsilon
        self.strategy_sample_interval = max(1, strategy_sample_interval)
        self.store_strategy_on_traverser_nodes = store_strategy_on_traverser_nodes
        self.store_strategy_on_opponent_nodes = store_strategy_on_opponent_nodes
        self.has_max_depth = max_depth is not None
        self.max_depth = 0 if max_depth is None else int(max_depth)
        self.has_max_nodes = max_nodes is not None
        self.max_nodes = 0 if max_nodes is None else int(max_nodes)
        self.outcome_sampling_epsilon = min(1.0, max(0.0, outcome_sampling_epsilon))
        self.has_value_clip = outcome_sampling_value_clip is not None
        self.outcome_sampling_value_clip = (
            0.0 if outcome_sampling_value_clip is None else max(1.0e-9, float(outcome_sampling_value_clip))
        )
        if outcome_unsampled_regret not in {"negative_node_value", "zero"}:
            raise ValueError("outcome_unsampled_regret must be 'negative_node_value' or 'zero'")
        self.unsampled_regret_zero = outcome_unsampled_regret == "zero"
        if cutoff_value_mode not in {"score_diff", "random_rollout"}:
            raise ValueError("cutoff_value_mode must be 'score_diff' or 'random_rollout'")
        self.cutoff_random_rollout = cutoff_value_mode == "random_rollout"
        self.cutoff_rollouts = max(0, cutoff_rollouts)
        if cutoff_rollout_policy not in {"random", "safe_heuristic"}:
            raise ValueError("cutoff_rollout_policy must be 'random' or 'safe_heuristic'")
        self.cutoff_rollout_max_steps = max(1, cutoff_rollout_max_steps)
        self.safe_heuristic_rollout_bot = (
            SafeHeuristicBot() if cutoff_rollout_policy == "safe_heuristic" else None
        )
        if opponent_policy == "network":
            self.opponent_policy_id = 0
        elif opponent_policy == "safe_heuristic":
            self.opponent_policy_id = 1
        elif opponent_policy == "self_play_league":
            self.opponent_policy_id = 2
        else:
            raise ValueError("opponent_policy must be 'network', 'safe_heuristic', or 'self_play_league'")
        self.league_advantage_networks = [] if league_advantage_networks is None else league_advantage_networks
        self.self_play_anchor_probability = min(1.0, max(0.0, self_play_anchor_probability))
        self.self_play_current_weight = max(0.0, self_play_current_weight)
        self.self_play_recent_weight = max(0.0, self_play_recent_weight)
        self.self_play_older_weight = max(0.0, self_play_older_weight)
        self.self_play_anchor_weight = max(0.0, self_play_anchor_weight)
        self.self_play_recent_window = max(0, self_play_recent_window)
        self.safe_heuristic_opponent_bot = (
            SafeHeuristicBot()
            if self.opponent_policy_id == 1 or self.self_play_anchor_probability > 0.0
            else None
        )
        self.endpoint_depth_bucket_width = max(1, endpoint_depth_bucket_width)
        self.endpoint_depth_bucket_max = max(1, endpoint_depth_bucket_max)
        self.rng = seed if seed != 0 else 1

    cpdef tuple traverse(self, GameState state, int traverser, int iteration):
        cdef object stats = TraversalStats()
        cdef float value
        if self.input_dim < 0:
            self.input_dim = _input_dim_with_flags_c(
                state, self.derived_playability, self.slot_aware_playability
            )
        value = self._traverse(state, traverser, iteration, 0, stats)
        return value, stats

    cdef float _traverse(
        self,
        GameState state,
        int traverser,
        int iteration,
        int depth,
        object stats,
    ) except *:
        cdef int player
        cdef int fixed_action
        cdef int fixed_unified_action
        cdef int swapped_deck_index
        cdef int actions[MAX_ACTIONS]
        cdef int legal_count
        cdef int i
        cdef int action
        cdef int local_action
        cdef float child_value
        cdef float action_prob
        cdef float sampled_action_value
        cdef float node_value
        cdef float policy[MAX_ACTIONS]
        cdef float sampling_policy[MAX_ACTIONS]
        cdef unsigned char legal[MAX_ACTIONS]
        cdef object info_state

        stats.nodes += 1
        if depth > stats.max_depth_reached:
            stats.max_depth_reached = depth

        if self.has_max_nodes and stats.nodes >= self.max_nodes:
            stats.node_limit_cutoffs += 1
            self._record_endpoint(stats, depth)
            return self._cutoff_value(state, traverser, stats)
        if state.terminal:
            stats.terminals += 1
            self._record_endpoint(stats, depth)
            return <float>(state.total_scores[traverser] - state.total_scores[1 - traverser])
        if self.has_max_depth and depth >= self.max_depth:
            stats.depth_cutoffs += 1
            self._record_endpoint(stats, depth)
            return self._cutoff_value(state, traverser, stats)

        player = state.current_player
        fixed_action = self._fixed_opponent_action(state, player, traverser)
        if fixed_action >= 0:
            fixed_unified_action = self._to_unified_action_c(state, fixed_action)
            swapped_deck_index = self._sample_deck_draw_chance(state, fixed_unified_action)
            state._push_action_c(fixed_action)
            try:
                return self._traverse(state, traverser, iteration, depth + 1, stats)
            finally:
                state._pop_action_c()
                if swapped_deck_index >= 0:
                    state._swap_deck_cards_c(swapped_deck_index, state.deck_len - 1)

        info_state = self._policy(state, player, legal, policy)
        self._record_strategy(info_state, legal, policy, player, traverser, iteration, depth, stats)

        legal_count = 0
        for i in range(self.action_size):
            if legal[i] != 0:
                actions[legal_count] = i
                legal_count += 1
        if legal_count <= 0:
            stats.terminals += 1
            self._record_endpoint(stats, depth)
            return <float>(state.total_scores[traverser] - state.total_scores[1 - traverser])

        self._sampling_policy(policy, legal, sampling_policy)
        action = _sample_policy_from_actions_c(sampling_policy, actions, legal_count, _next_double(&self.rng))
        local_action = self._from_unified_action_c(state, action)
        swapped_deck_index = self._sample_deck_draw_chance(state, action)
        state._push_action_c(local_action)
        try:
            child_value = self._traverse(state, traverser, iteration, depth + 1, stats)
        finally:
            state._pop_action_c()
            if swapped_deck_index >= 0:
                state._swap_deck_cards_c(swapped_deck_index, state.deck_len - 1)

        stats.sampled_actions += 1
        action_prob = sampling_policy[action]
        if action_prob < self.epsilon:
            action_prob = self.epsilon
        sampled_action_value = child_value / action_prob
        if self.has_value_clip:
            if sampled_action_value > self.outcome_sampling_value_clip:
                sampled_action_value = self.outcome_sampling_value_clip
            elif sampled_action_value < -self.outcome_sampling_value_clip:
                sampled_action_value = -self.outcome_sampling_value_clip
        node_value = policy[action] * sampled_action_value

        if player == traverser:
            self._record_advantage(
                info_state,
                legal,
                action,
                sampled_action_value,
                node_value,
                iteration,
                player,
                stats,
            )
        return node_value

    cdef object _policy(
        self,
        GameState state,
        int player,
        unsigned char* legal,
        float* policy,
    ):
        return self._policy_from_networks(self.advantage_networks, state, player, legal, policy)

    cdef object _policy_from_networks(
        self,
        object networks,
        GameState state,
        int player,
        unsigned char* legal,
        float* policy,
    ):
        cdef float[::1] info_view
        cdef float[::1] adv_view
        cdef int actions[MAX_ACTIONS]
        cdef int action_count
        cdef int i

        info_state = np.empty(self.input_dim, dtype=np.float32)
        info_view = info_state
        _encode_info_state_with_flags_c(
            state,
            player,
            &info_view[0],
            self.derived_playability,
            self.slot_aware_playability,
        )
        for i in range(self.action_size):
            legal[i] = 0
        action_count = state._unified_legal_actions_c(actions)
        for i in range(action_count):
            legal[actions[i]] = 1
        with torch.inference_mode():
            x = torch.as_tensor(info_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = networks[player](x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        adv_view = advantages
        regret_matching_c(&adv_view[0], legal, self.action_size, self.epsilon, policy)
        return info_state

    cdef void _sampling_policy(
        self,
        const float* policy,
        const unsigned char* legal,
        float* out_policy,
    ) noexcept:
        cdef int i
        cdef int legal_count = 0
        cdef float uniform
        for i in range(self.action_size):
            if legal[i] != 0:
                legal_count += 1
        if legal_count <= 0:
            for i in range(self.action_size):
                out_policy[i] = 0.0
            return
        if self.outcome_sampling_epsilon <= 0.0:
            for i in range(self.action_size):
                out_policy[i] = policy[i]
            return
        uniform = 1.0 / <float>legal_count
        for i in range(self.action_size):
            if legal[i] != 0:
                out_policy[i] = (
                    (1.0 - self.outcome_sampling_epsilon) * policy[i]
                    + self.outcome_sampling_epsilon * uniform
                )
            else:
                out_policy[i] = 0.0

    cdef int _fixed_opponent_action(self, GameState state, int player, int traverser) except *:
        cdef int bucket
        cdef object networks
        cdef unsigned char legal[MAX_ACTIONS]
        cdef float policy[MAX_ACTIONS]
        cdef int actions[MAX_ACTIONS]
        cdef int count = 0
        cdef int i
        cdef int unified_action
        if player == traverser or self.opponent_policy_id == 0:
            return -1
        if self.opponent_policy_id == 1:
            if self.safe_heuristic_opponent_bot is None:
                self.safe_heuristic_opponent_bot = SafeHeuristicBot()
            return int(self.safe_heuristic_opponent_bot.act(state))
        bucket = self._self_play_bucket()
        if bucket == 0:
            return -1
        if bucket == 3:
            if self.safe_heuristic_opponent_bot is None:
                self.safe_heuristic_opponent_bot = SafeHeuristicBot()
            return int(self.safe_heuristic_opponent_bot.act(state))
        networks = self._self_play_snapshot_networks(bucket)
        if networks is None:
            return -1
        self._policy_from_networks(networks, state, player, legal, policy)
        for i in range(self.action_size):
            if legal[i] != 0:
                actions[count] = i
                count += 1
        if count <= 0:
            return -1
        unified_action = _sample_policy_from_actions_c(policy, actions, count, _next_double(&self.rng))
        return self._from_unified_action_c(state, unified_action)

    cdef int _self_play_bucket(self) noexcept:
        cdef int recent_count
        cdef int older_count
        cdef double weights[4]
        cdef double total
        cdef double pick
        if self.self_play_anchor_probability > 0.0 and _next_double(&self.rng) < self.self_play_anchor_probability:
            return 3
        recent_count = min(len(self.league_advantage_networks), self.self_play_recent_window)
        older_count = max(0, len(self.league_advantage_networks) - recent_count)
        weights[0] = self.self_play_current_weight
        weights[1] = self.self_play_recent_weight if recent_count > 0 else 0.0
        weights[2] = self.self_play_older_weight if older_count > 0 else 0.0
        weights[3] = self.self_play_anchor_weight
        total = weights[0] + weights[1] + weights[2] + weights[3]
        if total <= 0.0:
            return 0
        pick = _next_double(&self.rng) * total
        if pick < weights[0]:
            return 0
        pick -= weights[0]
        if pick < weights[1]:
            return 1
        pick -= weights[1]
        if pick < weights[2]:
            return 2
        return 3

    cdef object _self_play_snapshot_networks(self, int bucket):
        cdef int recent_count = min(len(self.league_advantage_networks), self.self_play_recent_window)
        cdef object candidates
        cdef int index
        if len(self.league_advantage_networks) == 0:
            return None
        if bucket == 1 and recent_count > 0:
            candidates = self.league_advantage_networks[-recent_count:]
        elif bucket == 2:
            candidates = self.league_advantage_networks[:max(0, len(self.league_advantage_networks) - recent_count)]
        else:
            candidates = self.league_advantage_networks
        if len(candidates) == 0:
            return None
        index = <int>(_next_u32(&self.rng) % <unsigned int>len(candidates))
        return candidates[index]

    cdef float _cutoff_value(self, GameState state, int traverser, object stats) except *:
        cdef int i
        cdef float total = 0.0
        if not self.cutoff_random_rollout or self.cutoff_rollouts <= 0:
            return <float>(state.total_scores[traverser] - state.total_scores[1 - traverser])
        for i in range(self.cutoff_rollouts):
            total += self._rollout_value(state, traverser, stats)
        return total / <float>self.cutoff_rollouts

    cdef float _rollout_value(self, GameState state, int traverser, object stats) except *:
        cdef int steps = 0
        cdef int actions[MAX_ACTIONS]
        cdef int count
        cdef int unified_action
        cdef int local_action
        cdef int swapped_deck_index
        cdef int* swapped_indices = <int*>malloc(self.cutoff_rollout_max_steps * sizeof(int))
        cdef float value
        if swapped_indices == NULL:
            raise MemoryError()
        while not state.terminal and steps < self.cutoff_rollout_max_steps:
            if self.safe_heuristic_rollout_bot is not None:
                local_action = int(self.safe_heuristic_rollout_bot.act(state))
                unified_action = self._to_unified_action_c(state, local_action)
            else:
                count = state._unified_legal_actions_c(actions)
                if count <= 0:
                    break
                unified_action = actions[_next_u32(&self.rng) % <unsigned int>count]
                local_action = self._from_unified_action_c(state, unified_action)
            swapped_deck_index = self._sample_deck_draw_chance(state, unified_action)
            state._push_action_c(local_action)
            swapped_indices[steps] = swapped_deck_index
            steps += 1
        stats.cutoff_rollouts += 1
        stats.cutoff_rollout_steps += steps
        if not state.terminal:
            stats.cutoff_rollout_timeouts += 1
        value = <float>(state.total_scores[traverser] - state.total_scores[1 - traverser])
        while steps > 0:
            steps -= 1
            state._pop_action_c()
            if swapped_indices[steps] >= 0:
                state._swap_deck_cards_c(swapped_indices[steps], state.deck_len - 1)
        free(swapped_indices)
        return value

    cdef int _sample_deck_draw_chance(self, GameState state, int unified_action) except *:
        cdef int deck_draw_action = 2 * state.hand_size
        cdef int sampled_index
        if state.phase_id != 1 or unified_action != deck_draw_action or state.deck_len <= 1:
            return -1
        sampled_index = <int>(_next_u32(&self.rng) % <unsigned int>state.deck_len)
        if sampled_index == state.deck_len - 1:
            return -1
        state._swap_deck_cards_c(sampled_index, state.deck_len - 1)
        return sampled_index

    cdef void _record_strategy(
        self,
        object info_state,
        const unsigned char* legal,
        const float* policy,
        int player,
        int traverser,
        int iteration,
        int depth,
        object stats,
    ):
        cdef int i
        if player == traverser:
            if not self.store_strategy_on_traverser_nodes:
                return
        elif not self.store_strategy_on_opponent_nodes:
            return
        if depth % self.strategy_sample_interval != 0:
            return
        target = np.empty(self.action_size, dtype=np.float32)
        legal_mask = np.empty(self.action_size, dtype=np.bool_)
        cdef float[::1] target_view = target
        cdef unsigned char[::1] legal_view = legal_mask.view(np.uint8)
        for i in range(self.action_size):
            target_view[i] = policy[i]
            legal_view[i] = legal[i]
        self.strategy_samples.append(
            TrainingSample(
                info_state=info_state,
                target=target,
                legal_mask=legal_mask,
                iteration=iteration,
                player=player,
            )
        )
        stats.strategy_samples += 1

    cdef void _record_advantage(
        self,
        object info_state,
        const unsigned char* legal,
        int sampled_action,
        float sampled_action_value,
        float node_value,
        int iteration,
        int player,
        object stats,
    ):
        cdef int i
        target = np.empty(self.action_size, dtype=np.float32)
        legal_mask = np.empty(self.action_size, dtype=np.bool_)
        cdef float[::1] target_view = target
        cdef unsigned char[::1] legal_view = legal_mask.view(np.uint8)
        for i in range(self.action_size):
            legal_view[i] = legal[i]
            if legal[i] == 0 or self.unsampled_regret_zero:
                target_view[i] = 0.0
            else:
                target_view[i] = -node_value
        target_view[sampled_action] = sampled_action_value - node_value
        self.advantage_samples.append(
            TrainingSample(
                info_state=info_state,
                target=target,
                legal_mask=legal_mask,
                iteration=iteration,
                player=player,
            )
        )
        stats.advantage_samples += 1

    cdef void _record_endpoint(self, object stats, int depth):
        cdef int width = self.endpoint_depth_bucket_width
        cdef int max_depth = self.endpoint_depth_bucket_max
        cdef int start = _depth_bucket_start(depth, width, max_depth)
        cdef str key
        stats.endpoint_depth_sum += depth
        if start >= max_depth:
            key = f"{max_depth}_plus"
        else:
            key = f"{start}_{start + width - 1}"
        stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1

    cdef int _from_unified_action_c(self, GameState state, int action_id) noexcept:
        if state.phase_id == 0:
            return action_id
        return action_id - 2 * state.hand_size

    cdef int _to_unified_action_c(self, GameState state, int action_id) noexcept:
        if state.phase_id == 0:
            return action_id
        return 2 * state.hand_size + action_id

    def drain_samples(self):
        advantage = self.advantage_samples
        strategy = self.strategy_samples
        self.advantage_samples = []
        self.strategy_samples = []
        return advantage, strategy


def run_cython_traversal_batch(
    object advantage_networks,
    object game_config,
    list seeds,
    int player,
    int iteration,
    *,
    object device,
    int action_size,
    object encoding=None,
    float epsilon=1.0e-8,
    int strategy_sample_interval=1,
    bint store_strategy_on_traverser_nodes=True,
    bint store_strategy_on_opponent_nodes=True,
    object max_depth=None,
    object max_nodes=None,
    float outcome_sampling_epsilon=0.0,
    object outcome_sampling_value_clip=None,
    str outcome_unsampled_regret="negative_node_value",
    str cutoff_value_mode="score_diff",
    int cutoff_rollouts=0,
    str cutoff_rollout_policy="random",
    int cutoff_rollout_max_steps=10000,
    str opponent_policy="network",
    object league_advantage_networks=None,
    float self_play_anchor_probability=0.0,
    float self_play_current_weight=0.5,
    float self_play_recent_weight=0.3,
    float self_play_older_weight=0.2,
    float self_play_anchor_weight=0.0,
    int self_play_recent_window=5,
    int endpoint_depth_bucket_width=10,
    int endpoint_depth_bucket_max=100,
    unsigned int seed=1,
):
    cdef object stats = TraversalStats()
    cdef object local_stats
    cdef object value
    cdef GameState state
    cdef int game_seed
    traverser = CythonDeepCFRTraverser(
        advantage_networks,
        device=device,
        action_size=action_size,
        encoding=encoding,
        epsilon=epsilon,
        strategy_sample_interval=strategy_sample_interval,
        store_strategy_on_traverser_nodes=store_strategy_on_traverser_nodes,
        store_strategy_on_opponent_nodes=store_strategy_on_opponent_nodes,
        max_depth=max_depth,
        max_nodes=max_nodes,
        outcome_sampling_epsilon=outcome_sampling_epsilon,
        outcome_sampling_value_clip=outcome_sampling_value_clip,
        outcome_unsampled_regret=outcome_unsampled_regret,
        cutoff_value_mode=cutoff_value_mode,
        cutoff_rollouts=cutoff_rollouts,
        cutoff_rollout_policy=cutoff_rollout_policy,
        cutoff_rollout_max_steps=cutoff_rollout_max_steps,
        opponent_policy=opponent_policy,
        league_advantage_networks=league_advantage_networks,
        self_play_anchor_probability=self_play_anchor_probability,
        self_play_current_weight=self_play_current_weight,
        self_play_recent_weight=self_play_recent_weight,
        self_play_older_weight=self_play_older_weight,
        self_play_anchor_weight=self_play_anchor_weight,
        self_play_recent_window=self_play_recent_window,
        endpoint_depth_bucket_width=endpoint_depth_bucket_width,
        endpoint_depth_bucket_max=endpoint_depth_bucket_max,
        seed=seed,
    )
    for game_seed in seeds:
        state = GameState.new_game(game_config, seed=game_seed)
        value, local_stats = traverser.traverse(state, player, iteration)
        stats.accumulate(local_stats)
    advantage_samples, strategy_samples = traverser.drain_samples()
    return stats, advantage_samples, strategy_samples
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
