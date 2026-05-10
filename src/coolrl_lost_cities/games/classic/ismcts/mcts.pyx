# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from __future__ import annotations

import math
import random

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots.heuristic_cy cimport HeuristicBot
from coolrl_lost_cities.games.classic.bots.heuristic import HeuristicBot as PyHeuristicBot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game cimport GameState

from .determinization import sample_determinization
from .info_set import canonical_info_set_key


DEF MAX_ACTIONS = 64
DEF DEFAULT_ACTION_SIZE = 64


cdef class _ArrayMap:
    cdef MctsNode node
    cdef int kind
    cdef int action_size
    cdef bint is_int

    def __init__(self, MctsNode node, int kind, bint is_int=False):
        self.node = node
        self.kind = kind
        self.action_size = node.action_size
        self.is_int = is_int

    cdef inline void _check(self, int action) except *:
        if action < 0 or action >= self.action_size:
            raise KeyError(action)

    cdef inline bint has(self, int action) noexcept:
        return 0 <= action < self.action_size and self.node.active_present[action] != 0

    cdef inline long get_int(self, int action, long default_value=0) noexcept:
        if 0 <= action < self.action_size and self.node.active_present[action] != 0:
            if self.kind == 1:
                return self.node.visits_arr[action]
            if self.kind == 3:
                return self.node.virtual_visits_arr[action]
        return default_value

    cdef inline double get_float(self, int action, double default_value=0.0) noexcept:
        if 0 <= action < self.action_size and self.node.active_present[action] != 0:
            if self.kind == 0:
                return self.node.priors_arr[action]
            if self.kind == 2:
                return self.node.value_sum_arr[action]
        return default_value

    cdef inline void _mark_active(self, int action) noexcept:
        if self.node.active_present[action] == 0:
            self.node.active_present[action] = 1
            self.node.active_actions[self.node.n_active] = action
            self.node.n_active += 1

    cdef inline void set_int(self, int action, long value) except *:
        self._check(action)
        self._mark_active(action)
        if self.kind == 1:
            self.node.visits_arr[action] = <int>value
        elif self.kind == 3:
            self.node.virtual_visits_arr[action] = <int>value
        else:
            raise TypeError("integer write to float node map")

    cdef inline void set_float(self, int action, double value) except *:
        self._check(action)
        self._mark_active(action)
        if self.kind == 0:
            self.node.priors_arr[action] = value
        elif self.kind == 2:
            self.node.value_sum_arr[action] = value
        else:
            raise TypeError("float write to integer node map")

    def get(self, action, default=None):
        cdef int a = int(action)
        if self.has(a):
            if self.is_int:
                return int(self.get_int(a, 0))
            return float(self.get_float(a, 0.0))
        return default

    def setdefault(self, action, default=None):
        cdef int a = int(action)
        if self.has(a):
            if self.is_int:
                return int(self.get_int(a, 0))
            return float(self.get_float(a, 0.0))
        if default is None:
            default = 0 if self.is_int else 0.0
        if self.is_int:
            self.set_int(a, int(default))
            return int(default)
        self.set_float(a, float(default))
        return float(default)

    def __getitem__(self, action):
        cdef int a = int(action)
        self._check(a)
        if self.node.active_present[a] == 0:
            raise KeyError(action)
        if self.is_int:
            return int(self.get_int(a, 0))
        return float(self.get_float(a, 0.0))

    def __setitem__(self, action, value):
        cdef int a = int(action)
        if self.is_int:
            self.set_int(a, int(value))
        else:
            self.set_float(a, float(value))

    def __contains__(self, action):
        return self.has(int(action))

    def __bool__(self):
        return self.node.n_active > 0

    def __len__(self):
        return self.node.n_active

    def items(self):
        cdef int i
        result = []
        cdef int action
        for i in range(self.node.n_active):
            action = self.node.active_actions[i]
            if self.is_int:
                result.append((action, int(self.get_int(action, 0))))
            else:
                result.append((action, float(self.get_float(action, 0.0))))
        return result

    def keys(self):
        cdef int i
        return [self.node.active_actions[i] for i in range(self.node.n_active)]

    def values(self):
        cdef int i
        result = []
        cdef int action
        for i in range(self.node.n_active):
            action = self.node.active_actions[i]
            if self.is_int:
                result.append(int(self.get_int(action, 0)))
            else:
                result.append(float(self.get_float(action, 0.0)))
        return result

    def __repr__(self):
        return repr(dict(self.items()))


cdef class MctsNode:
    cdef public bytes info_set_key
    cdef public int player
    cdef public object priors
    cdef public object visits
    cdef public object value_sum
    cdef public object virtual_visits
    cdef public dict children
    cdef public bint terminal
    cdef public bint expanded
    cdef int action_size
    cdef int visits_arr[MAX_ACTIONS]
    cdef double value_sum_arr[MAX_ACTIONS]
    cdef double priors_arr[MAX_ACTIONS]
    cdef int virtual_visits_arr[MAX_ACTIONS]
    cdef int active_actions[MAX_ACTIONS]
    cdef unsigned char active_present[MAX_ACTIONS]
    cdef int n_active

    def __init__(
        self,
        bytes info_set_key,
        int player,
        object priors=None,
        object visits=None,
        object value_sum=None,
        object virtual_visits=None,
        object children=None,
        bint terminal=False,
        int action_size=DEFAULT_ACTION_SIZE,
    ):
        if action_size > MAX_ACTIONS:
            raise ValueError("action_size exceeds fixed MCTS action buffer")
        self.info_set_key = info_set_key
        self.player = player
        self.terminal = terminal
        self.expanded = terminal
        self.action_size = action_size
        self.n_active = 0
        self.priors = _ArrayMap(self, 0, False)
        self.visits = _ArrayMap(self, 1, True)
        self.value_sum = _ArrayMap(self, 2, False)
        self.virtual_visits = _ArrayMap(self, 3, True)
        self.children = {} if children is None else dict(children)
        if priors is not None:
            for action, value in dict(priors).items():
                self.priors[action] = value
        if visits is not None:
            for action, value in dict(visits).items():
                self.visits[action] = value
        if value_sum is not None:
            for action, value in dict(value_sum).items():
                self.value_sum[action] = value
        if virtual_visits is not None:
            for action, value in dict(virtual_visits).items():
                self.virtual_visits[action] = value

    cpdef bint is_expanded(self):
        return self.terminal or self.expanded or self.n_active > 0

    cpdef double q(self, int action):
        cdef long n = (<_ArrayMap>self.visits).get_int(action, 0)
        if n <= 0:
            return 0.0
        return (<_ArrayMap>self.value_sum).get_float(action, 0.0) / n


cdef class SearchPathEntry:
    cdef public MctsNode node
    cdef public int action
    cdef public int parent_player
    cdef public int child_player

    def __init__(self, MctsNode node, int action, int parent_player, int child_player):
        self.node = node
        self.action = action
        self.parent_player = parent_player
        self.child_player = child_player


cdef class PendingSimulation:
    cdef public list path
    cdef public GameState leaf_state
    cdef public object leaf_node
    cdef public int leaf_player
    cdef public object info_state
    cdef public object legal_mask
    cdef public list legal_actions
    cdef public object terminal_value

    def __init__(
        self,
        list path,
        GameState leaf_state,
        object leaf_node,
        int leaf_player,
        object info_state,
        object legal_mask,
        list legal_actions,
        object terminal_value=None,
    ):
        self.path = path
        self.leaf_state = leaf_state
        self.leaf_node = leaf_node
        self.leaf_player = leaf_player
        self.info_state = info_state
        self.legal_mask = legal_mask
        self.legal_actions = legal_actions
        self.terminal_value = terminal_value


cdef class MctsTree:
    cdef public dict nodes
    cdef int action_size

    def __init__(self, int action_size=DEFAULT_ACTION_SIZE):
        self.nodes = {}
        self.action_size = action_size

    def get_or_create(self, bytes key, *, int player, bint terminal=False):
        cdef MctsNode node = self.nodes.get(key)
        if node is None:
            node = MctsNode(key, player=player, terminal=terminal, action_size=self.action_size)
            self.nodes[key] = node
        return node


cdef class IsMctsSearcher:
    cdef public object network
    cdef public object config
    cdef public object device
    cdef public object encoding
    cdef public object rng
    cdef public MctsTree tree
    cdef HeuristicBot _rollout_bot
    cdef int action_size

    def __init__(
        self,
        object network,
        object config,
        *,
        object device="cpu",
        object encoding=None,
        object rng=None,
    ):
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.encoding = encoding
        self.rng = rng or random.Random()
        self.action_size = int(getattr(network, "action_size", DEFAULT_ACTION_SIZE))
        if self.action_size > MAX_ACTIONS:
            raise ValueError("action_size exceeds fixed MCTS action buffer")
        self.tree = MctsTree(self.action_size)
        self._rollout_bot = (
            <HeuristicBot>PyHeuristicBot() if config.rollout_policy == "heuristic_balanced" else None
        )

    cdef inline int _from_unified_action_c(self, GameState state, int action_id) noexcept:
        cdef int card_action_size = 2 * state.hand_size
        if state.phase_id == 0:
            return action_id
        return action_id - card_action_size

    cdef list _unified_legal_actions_list_c(self, GameState state):
        cdef int actions[MAX_ACTIONS]
        cdef int count = state._unified_legal_actions_c(actions)
        cdef int i
        return [actions[i] for i in range(count)]

    cpdef dict search(self, GameState state, int traverser, object n_sims=None):
        cdef bytes root_key = canonical_info_set_key(state, state.current_player)
        cdef MctsNode root = self.tree.get_or_create(
            root_key, player=state.current_player, terminal=state.terminal
        )
        cdef int sims = int(n_sims or self.config.n_simulations)
        cdef int completed = 0
        cdef list pending
        cdef list legal
        cdef int action
        cdef dict result
        while completed < sims:
            pending = self.prepare_simulation_batch(state, traverser, 1)
            if not pending:
                break
            self.evaluate_and_backup(pending)
            completed += len(pending)
        legal = state.unified_legal_actions()
        result = {}
        for action in legal:
            result[action] = (<_ArrayMap>root.visits).get_int(action, 0)
        return result

    cpdef list prepare_simulation_batch(
        self,
        GameState root_state,
        int traverser,
        int max_simulations,
    ):
        cdef list pending = []
        cdef PendingSimulation item
        cdef int i
        for i in range(max_simulations):
            item = self.prepare_simulation(root_state, traverser)
            pending.append(item)
            if item.terminal_value is None and item.leaf_node is not None and not item.path:
                break
        return pending

    cpdef PendingSimulation prepare_simulation(self, GameState root_state, int traverser):
        cdef GameState state = sample_determinization(root_state, traverser, self.rng)
        cdef list path = []
        cdef int depth = 0
        cdef object cached_key = None
        cdef int player
        cdef bytes key
        cdef MctsNode node
        cdef list legal_actions
        cdef int action
        cdef int local_action
        cdef int child_player
        cdef bytes child_key
        cdef int actions[MAX_ACTIONS]
        cdef int action_count
        cdef int i
        while True:
            player = state.current_player
            if state.terminal or depth >= int(self.config.max_depth):
                return PendingSimulation(
                    path=path,
                    leaf_state=state,
                    leaf_node=None,
                    leaf_player=player,
                    info_state=None,
                    legal_mask=None,
                    legal_actions=[],
                    terminal_value=float(state.total_scores[player] - state.total_scores[1 - player]),
                )
            if cached_key is None:
                key = canonical_info_set_key(state, player)
            else:
                key = cached_key
            node = self.tree.get_or_create(key, player=player, terminal=state.terminal)
            if not node.is_expanded():
                action_count = state._unified_legal_actions_c(actions)
                legal_actions = [actions[i] for i in range(action_count)]
                if not legal_actions:
                    node.terminal = True
                    return PendingSimulation(
                        path=path,
                        leaf_state=state,
                        leaf_node=node,
                        leaf_player=player,
                        info_state=None,
                        legal_mask=None,
                        legal_actions=[],
                        terminal_value=float(state.total_scores[player] - state.total_scores[1 - player]),
                    )
                return PendingSimulation(
                    path=path,
                    leaf_state=state,
                    leaf_node=node,
                    leaf_player=player,
                    info_state=encode_info_state(state, player, self.encoding),
                    legal_mask=np.asarray(state.unified_legal_mask(), dtype=bool),
                    legal_actions=legal_actions,
                )

            action_count = state._unified_legal_actions_c(actions)
            legal_actions = [actions[i] for i in range(action_count)]
            action = self._select_action(node, legal_actions)
            (<_ArrayMap>node.virtual_visits).set_int(
                action, (<_ArrayMap>node.virtual_visits).get_int(action, 0) + 1
            )
            local_action = self._from_unified_action_c(state, action)
            state._push_action_c(local_action)
            child_player = state.current_player
            child_key = canonical_info_set_key(state, child_player)
            node.children[action] = child_key
            path.append(
                SearchPathEntry(
                    node=node,
                    action=action,
                    parent_player=player,
                    child_player=child_player,
                )
            )
            cached_key = child_key
            depth += 1

    cpdef evaluate_and_backup(self, list pending):
        cdef list network_pending = [item for item in pending if item.terminal_value is None]
        cdef dict values_by_id = {}
        cdef dict priors_by_id = {}
        cdef object infos
        cdef object masks
        cdef object x
        cdef object mask
        cdef object probs
        cdef object network_values
        cdef object network_values_np
        cdef int index
        cdef PendingSimulation item
        cdef double value
        if network_pending:
            infos = np.stack([item.info_state for item in network_pending if item.info_state is not None])
            masks = np.stack([item.legal_mask for item in network_pending if item.legal_mask is not None])
            with torch.inference_mode():
                x = torch.as_tensor(infos, dtype=torch.float32, device=self.device)
                mask = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
                probs = self.network.policy_distribution(x, mask).detach().cpu().numpy()
                _logits, network_values = self.network(x, mask)
                network_values_np = network_values.detach().cpu().numpy()
            for index, item in enumerate(network_pending):
                priors_by_id[id(item)] = probs[index]
                values_by_id[id(item)] = float(network_values_np[index])

        for item in pending:
            if item.terminal_value is not None:
                value = item.terminal_value
            else:
                value = self._expand_with_prior(
                    item.leaf_node,
                    item.leaf_state,
                    item.leaf_player,
                    item.legal_actions,
                    priors_by_id[id(item)],
                    values_by_id[id(item)],
                )
            self._backup(item.path, value, item.leaf_player)

    cpdef double _expand_with_prior(
        self,
        MctsNode node,
        GameState state,
        int player,
        list legal_actions,
        object probs,
        double network_value,
    ):
        cdef int action
        cdef object rollout_value
        legal_actions = self._unified_legal_actions_list_c(state)
        if not legal_actions:
            node.terminal = True
            return float(state.total_scores[player] - state.total_scores[1 - player])
        node.expanded = True
        for action in legal_actions:
            (<_ArrayMap>node.priors).set_float(action, float(probs[action]))
            if not (<_ArrayMap>node.visits).has(action):
                (<_ArrayMap>node.visits).set_int(action, 0)
            if not (<_ArrayMap>node.value_sum).has(action):
                (<_ArrayMap>node.value_sum).set_float(action, 0.0)
            if not (<_ArrayMap>node.virtual_visits).has(action):
                (<_ArrayMap>node.virtual_visits).set_int(action, 0)
        rollout_value = self._rollout_value(state, player) if self.config.use_rollout_value else None
        if rollout_value is None:
            return float(network_value)
        return float(rollout_value)

    cpdef int _select_action(self, MctsNode node, list legal_actions):
        cdef int total_visits = 0
        cdef int action
        cdef long n
        cdef long virtual
        cdef long n_eff
        cdef double sqrt_total
        cdef double prior
        cdef double q_eff
        cdef double score
        cdef double best_score = -float("inf")
        cdef int best_action = int(legal_actions[0])
        cdef _ArrayMap visits = <_ArrayMap>node.visits
        cdef _ArrayMap virtual_visits = <_ArrayMap>node.virtual_visits
        cdef _ArrayMap priors = <_ArrayMap>node.priors
        cdef _ArrayMap value_sum = <_ArrayMap>node.value_sum
        for action in legal_actions:
            total_visits += visits.get_int(action, 0) + virtual_visits.get_int(action, 0)
        sqrt_total = math.sqrt(max(1, total_visits))
        for action in legal_actions:
            n = visits.get_int(action, 0)
            virtual = virtual_visits.get_int(action, 0)
            n_eff = n + virtual
            prior = priors.get_float(action, 0.0)
            if n_eff <= 0:
                q_eff = 0.0
            else:
                q_eff = (
                    value_sum.get_float(action, 0.0)
                    - virtual * float(self.config.virtual_loss_value)
                ) / n_eff
            score = q_eff + float(self.config.c_puct) * prior * sqrt_total / (1 + n_eff)
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    cpdef _backup(self, list path, double leaf_value, int leaf_player):
        cdef double value = float(leaf_value)
        cdef int value_player = int(leaf_player)
        cdef SearchPathEntry entry
        cdef double parent_value
        cdef long current_virtual
        cdef _ArrayMap visits
        cdef _ArrayMap virtual_visits
        cdef _ArrayMap value_sum
        for entry in reversed(path):
            parent_value = value if value_player == entry.parent_player else -value
            virtual_visits = <_ArrayMap>entry.node.virtual_visits
            visits = <_ArrayMap>entry.node.visits
            value_sum = <_ArrayMap>entry.node.value_sum
            current_virtual = virtual_visits.get_int(entry.action, 0)
            virtual_visits.set_int(entry.action, max(0, current_virtual - 1))
            visits.set_int(entry.action, visits.get_int(entry.action, 0) + 1)
            value_sum.set_float(
                entry.action,
                value_sum.get_float(entry.action, 0.0) + parent_value,
            )
            value = parent_value
            value_player = entry.parent_player

    cpdef _release_virtual_path(self, list path):
        cdef SearchPathEntry entry
        cdef _ArrayMap virtual_visits
        cdef long current_virtual
        for entry in path:
            virtual_visits = <_ArrayMap>entry.node.virtual_visits
            current_virtual = virtual_visits.get_int(entry.action, 0)
            virtual_visits.set_int(entry.action, max(0, current_virtual - 1))

    cpdef object _rollout_value(self, GameState state, int player):
        cdef int steps = 0
        cdef int actions[MAX_ACTIONS]
        cdef int count
        cdef int unified_action
        cdef int action
        cdef int max_depth = int(self.config.max_depth)
        while not state.terminal and steps < max_depth:
            if self._rollout_bot is not None:
                action = self._rollout_bot.act_cython(state)
                if not state._is_legal_action_c(action):
                    count = state._unified_legal_actions_c(actions)
                    if count <= 0:
                        break
                    unified_action = actions[self.rng.randrange(count)]
                    action = self._from_unified_action_c(state, unified_action)
            else:
                count = state._unified_legal_actions_c(actions)
                if count <= 0:
                    break
                unified_action = actions[self.rng.randrange(count)]
                action = self._from_unified_action_c(state, unified_action)
            state._push_action_c(action)
            steps += 1
        while steps > 0:
            state._pop_action_c()
            steps -= 1
        return float(state.total_scores[player] - state.total_scores[1 - player])
