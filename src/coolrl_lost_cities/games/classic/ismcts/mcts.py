from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

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
    children: dict[int, bytes] = field(default_factory=dict)
    terminal: bool = False

    def is_expanded(self) -> bool:
        return self.terminal or bool(self.priors)

    def q(self, action: int) -> float:
        n = self.visits.get(action, 0)
        if n <= 0:
            return 0.0
        return self.value_sum.get(action, 0.0) / n


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
        for _ in range(sims):
            det = sample_determinization(state, traverser, self.rng)
            self._simulate(det, depth=0)
        legal = state.unified_legal_actions()
        return {action: root.visits.get(action, 0) for action in legal}

    def _simulate(self, state: GameState, *, depth: int) -> float:
        player = int(state.current_player)
        if state.terminal or depth >= self.config.max_depth:
            return float(state.score_diff(player))

        key = canonical_info_set_key(state, player)
        node = self.tree.get_or_create(key, player=player, terminal=state.terminal)
        if not node.is_expanded():
            value = self._expand_and_evaluate(node, state, player)
            return value

        action = self._select_action(node, state.unified_legal_actions())
        child = state.clone()
        child.apply_unified_action(action)
        child_value = self._simulate(child, depth=depth + 1)
        value = child_value if child.current_player == player else -child_value
        node.visits[action] = node.visits.get(action, 0) + 1
        node.value_sum[action] = node.value_sum.get(action, 0.0) + value
        child_key = canonical_info_set_key(child, child.current_player)
        node.children[action] = child_key
        return value

    def _expand_and_evaluate(self, node: MctsNode, state: GameState, player: int) -> float:
        legal_actions = state.unified_legal_actions()
        if not legal_actions:
            node.terminal = True
            return float(state.score_diff(player))
        info = encode_info_state(state, player, self.encoding)
        legal_mask = np.asarray(state.unified_legal_mask(), dtype=bool)
        with torch.inference_mode():
            x = torch.as_tensor(info[None, :], dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(legal_mask[None, :], dtype=torch.bool, device=self.device)
            probs = self.network.policy_distribution(x, mask).squeeze(0).detach().cpu().numpy()
            _logits, network_value = self.network(x, mask)
        for action in legal_actions:
            node.priors[action] = float(probs[action])
            node.visits.setdefault(action, 0)
            node.value_sum.setdefault(action, 0.0)
        rollout_value = (
            self._rollout_value(state, player) if self.config.use_rollout_value else None
        )
        if rollout_value is None:
            return float(network_value.item())
        return rollout_value

    def _select_action(self, node: MctsNode, legal_actions: list[int]) -> int:
        total_visits = sum(node.visits.get(action, 0) for action in legal_actions)
        sqrt_total = math.sqrt(max(1, total_visits))
        best_score = -float("inf")
        best_action = legal_actions[0]
        for action in legal_actions:
            n = node.visits.get(action, 0)
            prior = node.priors.get(action, 0.0)
            score = node.q(action) + self.config.c_puct * prior * sqrt_total / (1 + n)
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def _rollout_value(self, state: GameState, player: int) -> float | None:
        rollout = state.clone()
        steps = 0
        while not rollout.terminal and steps < self.config.max_depth:
            legal = rollout.unified_legal_actions()
            if not legal:
                break
            action = self.rng.choice(legal)
            rollout.apply_unified_action(action)
            steps += 1
        return float(rollout.score_diff(player))
