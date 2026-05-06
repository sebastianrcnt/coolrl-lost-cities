from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.cfr_math import regret_matching
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.game import GameState


@dataclass
class TraversalStats:
    nodes: int = 0
    terminals: int = 0
    depth_cutoffs: int = 0
    node_limit_cutoffs: int = 0
    max_depth_reached: int = 0
    advantage_samples: int = 0
    strategy_samples: int = 0
    sampled_actions: int = 0
    endpoint_depth_sum: int = 0
    endpoint_depth_buckets: dict[str, int] = field(default_factory=dict)

    def accumulate(self, other: TraversalStats) -> None:
        self.nodes += other.nodes
        self.terminals += other.terminals
        self.depth_cutoffs += other.depth_cutoffs
        self.node_limit_cutoffs += other.node_limit_cutoffs
        self.max_depth_reached = max(self.max_depth_reached, other.max_depth_reached)
        self.advantage_samples += other.advantage_samples
        self.strategy_samples += other.strategy_samples
        self.sampled_actions += other.sampled_actions
        self.endpoint_depth_sum += other.endpoint_depth_sum
        for key, value in other.endpoint_depth_buckets.items():
            self.endpoint_depth_buckets[key] = self.endpoint_depth_buckets.get(key, 0) + value

    @property
    def endpoints(self) -> int:
        return self.terminals + self.depth_cutoffs + self.node_limit_cutoffs

    @property
    def avg_endpoint_depth(self) -> float:
        return self.endpoint_depth_sum / max(1, self.endpoints)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "traversal_nodes": self.nodes,
            "traversal_terminals": self.terminals,
            "traversal_depth_cutoffs": self.depth_cutoffs,
            "traversal_node_limit_cutoffs": self.node_limit_cutoffs,
            "traversal_max_depth_reached": self.max_depth_reached,
            "traversal_advantage_samples": self.advantage_samples,
            "traversal_strategy_samples": self.strategy_samples,
            "traversal_sampled_actions": self.sampled_actions,
            "traversal_avg_endpoint_depth": self.avg_endpoint_depth,
            **{
                f"traversal_endpoint_depth_bucket_{key}": value
                for key, value in self.endpoint_depth_buckets.items()
            },
        }


class DeepCFRTraverser:
    def __init__(
        self,
        advantage_networks: list[torch.nn.Module],
        advantage_memory: ReservoirMemory,
        strategy_memory: ReservoirMemory,
        *,
        device: torch.device,
        action_size: int,
        epsilon: float = 1.0e-8,
        strategy_sample_interval: int = 1,
        store_strategy_on_traverser_nodes: bool = True,
        store_strategy_on_opponent_nodes: bool = True,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.advantage_networks = advantage_networks
        self.advantage_memory = advantage_memory
        self.strategy_memory = strategy_memory
        self.device = device
        self.action_size = action_size
        self.epsilon = float(epsilon)
        self.strategy_sample_interval = max(1, int(strategy_sample_interval))
        self.store_strategy_on_traverser_nodes = store_strategy_on_traverser_nodes
        self.store_strategy_on_opponent_nodes = store_strategy_on_opponent_nodes
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.rng = rng or np.random.default_rng()

    def traverse(
        self, state: GameState, traverser: int, iteration: int
    ) -> tuple[float, TraversalStats]:
        stats = TraversalStats()
        value = self._traverse(state, traverser, iteration, depth=0, stats=stats)
        return value, stats

    def _traverse(
        self,
        state: GameState,
        traverser: int,
        iteration: int,
        *,
        depth: int,
        stats: TraversalStats,
    ) -> float:
        stats.nodes += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth)

        if self.max_nodes is not None and stats.nodes >= self.max_nodes:
            stats.node_limit_cutoffs += 1
            self._record_endpoint(stats, depth)
            return float(state.score_diff(traverser))
        if state.terminal:
            stats.terminals += 1
            self._record_endpoint(stats, depth)
            return float(state.score_diff(traverser))
        if self.max_depth is not None and depth >= self.max_depth:
            stats.depth_cutoffs += 1
            self._record_endpoint(stats, depth)
            return float(state.score_diff(traverser))

        player = state.current_player
        info_state, legal, policy = self._policy(state, player)
        self._record_strategy(info_state, legal, policy, player, traverser, iteration, depth, stats)

        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            stats.terminals += 1
            self._record_endpoint(stats, depth)
            return float(state.score_diff(traverser))

        action = self._sample_action(policy, legal_actions)
        local_action = state.from_unified_action(int(action))
        state.push_action(local_action)
        try:
            child_value = self._traverse(
                state,
                traverser,
                iteration,
                depth=depth + 1,
                stats=stats,
            )
        finally:
            state.pop_action()

        stats.sampled_actions += 1
        action_prob = max(float(policy[action]), self.epsilon)
        sampled_action_value = child_value / action_prob
        node_value = float(policy[action]) * sampled_action_value

        if player == traverser:
            regrets = np.where(legal, -node_value, 0.0).astype(np.float32)
            regrets[action] = np.float32(sampled_action_value - node_value)
            self.advantage_memory.add(
                TrainingSample(
                    info_state=info_state,
                    target=regrets,
                    legal_mask=legal,
                    iteration=iteration,
                    player=player,
                )
            )
            stats.advantage_samples += 1

        return node_value

    def _policy(self, state: GameState, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info_state = encode_info_state(state, player)
        legal = np.asarray(state.unified_legal_mask(), dtype=bool)
        with torch.inference_mode():
            x = torch.as_tensor(info_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = (
                self.advantage_networks[player](x)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        policy = regret_matching(advantages, legal, self.epsilon).astype(np.float32)
        return info_state, legal, policy

    def _sample_action(self, policy: np.ndarray, legal_actions: np.ndarray) -> int:
        probs = policy[legal_actions].astype(np.float64)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
        else:
            probs /= total
        return int(self.rng.choice(legal_actions, p=probs))

    def _record_strategy(
        self,
        info_state: np.ndarray,
        legal: np.ndarray,
        policy: np.ndarray,
        player: int,
        traverser: int,
        iteration: int,
        depth: int,
        stats: TraversalStats,
    ) -> None:
        if player == traverser:
            if not self.store_strategy_on_traverser_nodes:
                return
        elif not self.store_strategy_on_opponent_nodes:
            return
        if depth % self.strategy_sample_interval != 0:
            return
        self.strategy_memory.add(
            TrainingSample(
                info_state=info_state,
                target=policy,
                legal_mask=legal,
                iteration=iteration,
                player=player,
            )
        )
        stats.strategy_samples += 1

    def _record_endpoint(self, stats: TraversalStats, depth: int) -> None:
        stats.endpoint_depth_sum += depth
        start = min(depth // 10 * 10, 100)
        key = "100_plus" if start >= 100 else f"{start}_{start + 9}"
        stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1
