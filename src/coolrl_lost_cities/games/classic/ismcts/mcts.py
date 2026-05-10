from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots.heuristic import HeuristicBot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState

from .config import MctsConfig
from .determinization import sample_determinization
from .info_set import canonical_info_set_key
from .network import AlphaZeroNet


@dataclass
class MctsNode:
    info_set_key: bytes
    player: int
    priors: dict[int, float] = field(default_factory=dict)
    visits: dict[int, int] = field(default_factory=dict)
    value_sum: dict[int, float] = field(default_factory=dict)
    virtual_visits: dict[int, int] = field(default_factory=dict)
    children: dict[int, bytes] = field(default_factory=dict)
    terminal: bool = False

    def is_expanded(self) -> bool:
        return self.terminal or bool(self.priors)

    def q(self, action: int) -> float:
        n = self.visits.get(action, 0)
        if n <= 0:
            return 0.0
        return self.value_sum.get(action, 0.0) / n


@dataclass
class SearchPathEntry:
    node: MctsNode
    action: int
    parent_player: int
    child_player: int


@dataclass
class PendingSimulation:
    path: list[SearchPathEntry]
    leaf_state: GameState
    leaf_node: MctsNode | None
    leaf_player: int
    info_state: np.ndarray | None
    legal_mask: np.ndarray | None
    legal_actions: list[int]
    terminal_value: float | None = None


class MctsTree:
    def __init__(self) -> None:
        self.nodes: dict[bytes, MctsNode] = {}

    def get_or_create(self, key: bytes, *, player: int, terminal: bool = False) -> MctsNode:
        node = self.nodes.get(key)
        if node is None:
            node = MctsNode(key, player=player, terminal=terminal)
            self.nodes[key] = node
        return node


class IsMctsSearcher:
    def __init__(
        self,
        network: AlphaZeroNet,
        config: MctsConfig,
        *,
        device: torch.device | str = "cpu",
        encoding=None,
        rng: random.Random | None = None,
    ) -> None:
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.encoding = encoding
        self.rng = rng or random.Random()
        self.tree = MctsTree()
        self._rollout_bot = (
            HeuristicBot() if config.rollout_policy == "heuristic_balanced" else None
        )

    def search(
        self,
        state: GameState,
        traverser: int,
        n_sims: int | None = None,
    ) -> dict[int, int]:
        root_key = canonical_info_set_key(state, state.current_player)
        root = self.tree.get_or_create(
            root_key, player=state.current_player, terminal=state.terminal
        )
        sims = int(n_sims or self.config.n_simulations)
        completed = 0
        while completed < sims:
            batch_size = min(self.config.parallel_simulations, sims - completed)
            pending = self.prepare_simulation_batch(state, traverser, batch_size)
            if not pending:
                break
            self.evaluate_and_backup(pending)
            completed += len(pending)
        legal = state.unified_legal_actions()
        return {action: root.visits.get(action, 0) for action in legal}

    def prepare_simulation_batch(
        self,
        root_state: GameState,
        traverser: int,
        max_simulations: int,
    ) -> list[PendingSimulation]:
        pending: list[PendingSimulation] = []
        for _ in range(max_simulations):
            item = self.prepare_simulation(root_state, traverser)
            pending.append(item)
            if item.terminal_value is None and item.leaf_node is not None and not item.path:
                break
        return pending

    def prepare_simulation(self, root_state: GameState, traverser: int) -> PendingSimulation:
        state = sample_determinization(root_state, traverser, self.rng)
        path: list[SearchPathEntry] = []
        depth = 0
        # Cache the info-set key for the current node so we don't recompute it
        # after applying an action (the child's key becomes the next iter's key).
        cached_key: bytes | None = None
        while True:
            player = int(state.current_player)
            if state.terminal or depth >= self.config.max_depth:
                return PendingSimulation(
                    path=path,
                    leaf_state=state,
                    leaf_node=None,
                    leaf_player=player,
                    info_state=None,
                    legal_mask=None,
                    legal_actions=[],
                    terminal_value=float(state.score_diff(player)),
                )
            key = cached_key if cached_key is not None else canonical_info_set_key(state, player)
            node = self.tree.get_or_create(key, player=player, terminal=state.terminal)
            if not node.is_expanded():
                legal_actions = state.unified_legal_actions()
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
                        terminal_value=float(state.score_diff(player)),
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

            # Reuse the same legal_actions list for selection (avoids one
            # extra unified_legal_actions() call inside _select_action).
            legal_actions = state.unified_legal_actions()
            action = self._select_action(node, legal_actions)
            node.virtual_visits[action] = node.virtual_visits.get(action, 0) + 1
            child = state.clone()
            child.apply_unified_action(action)
            child_player = int(child.current_player)
            child_key = canonical_info_set_key(child, child_player)
            node.children[action] = child_key
            path.append(
                SearchPathEntry(
                    node=node,
                    action=action,
                    parent_player=player,
                    child_player=child_player,
                )
            )
            state = child
            cached_key = child_key
            depth += 1

    def evaluate_and_backup(self, pending: list[PendingSimulation]) -> None:
        network_pending = [item for item in pending if item.terminal_value is None]
        values_by_id: dict[int, float] = {}
        priors_by_id: dict[int, np.ndarray] = {}
        if network_pending:
            infos = np.stack(
                [item.info_state for item in network_pending if item.info_state is not None]
            )
            masks = np.stack(
                [item.legal_mask for item in network_pending if item.legal_mask is not None]
            )
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
                assert item.leaf_node is not None
                value = self._expand_with_prior(
                    item.leaf_node,
                    item.leaf_state,
                    item.leaf_player,
                    item.legal_actions,
                    priors_by_id[id(item)],
                    values_by_id[id(item)],
                )
            self._backup(item.path, value, item.leaf_player)

    def _expand_with_prior(
        self,
        node: MctsNode,
        state: GameState,
        player: int,
        legal_actions: list[int],
        probs: np.ndarray,
        network_value: float,
    ) -> float:
        legal_actions = state.unified_legal_actions()
        if not legal_actions:
            node.terminal = True
            return float(state.score_diff(player))
        for action in legal_actions:
            node.priors[action] = float(probs[action])
            node.visits.setdefault(action, 0)
            node.value_sum.setdefault(action, 0.0)
            node.virtual_visits.setdefault(action, 0)
        rollout_value = (
            self._rollout_value(state, player) if self.config.use_rollout_value else None
        )
        if rollout_value is None:
            return float(network_value)
        return rollout_value

    def _select_action(self, node: MctsNode, legal_actions: list[int]) -> int:
        total_visits = sum(
            node.visits.get(action, 0) + node.virtual_visits.get(action, 0)
            for action in legal_actions
        )
        sqrt_total = math.sqrt(max(1, total_visits))
        best_score = -float("inf")
        best_action = legal_actions[0]
        for action in legal_actions:
            n = node.visits.get(action, 0)
            virtual = node.virtual_visits.get(action, 0)
            n_eff = n + virtual
            prior = node.priors.get(action, 0.0)
            if n_eff <= 0:
                q_eff = 0.0
            else:
                q_eff = (
                    node.value_sum.get(action, 0.0) - virtual * self.config.virtual_loss_value
                ) / n_eff
            score = q_eff + self.config.c_puct * prior * sqrt_total / (1 + n_eff)
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def _backup(
        self,
        path: list[SearchPathEntry],
        leaf_value: float,
        leaf_player: int,
    ) -> None:
        value = float(leaf_value)
        value_player = int(leaf_player)
        for entry in reversed(path):
            parent_value = value if value_player == entry.parent_player else -value
            current_virtual = entry.node.virtual_visits.get(entry.action, 0)
            entry.node.virtual_visits[entry.action] = max(0, current_virtual - 1)
            entry.node.visits[entry.action] = entry.node.visits.get(entry.action, 0) + 1
            entry.node.value_sum[entry.action] = (
                entry.node.value_sum.get(entry.action, 0.0) + parent_value
            )
            value = parent_value
            value_player = entry.parent_player

    def _release_virtual_path(self, path: list[SearchPathEntry]) -> None:
        for entry in path:
            current_virtual = entry.node.virtual_visits.get(entry.action, 0)
            entry.node.virtual_visits[entry.action] = max(0, current_virtual - 1)

    def _rollout_value(self, state: GameState, player: int) -> float | None:
        rollout = state.clone()
        steps = 0
        while not rollout.terminal and steps < self.config.max_depth:
            legal = rollout.unified_legal_actions()
            if not legal:
                break
            if self._rollout_bot is not None:
                phase_action = self._rollout_bot.act(rollout)
                action = rollout.to_unified_action(phase_action)
                if action not in legal:
                    action = self.rng.choice(legal)
            else:
                action = self.rng.choice(legal)
            rollout.apply_unified_action(action)
            steps += 1
        return float(rollout.score_diff(player))
