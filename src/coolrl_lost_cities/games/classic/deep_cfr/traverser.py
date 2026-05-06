from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots import SafeHeuristicBot
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
    cutoff_rollouts: int = 0
    cutoff_rollout_steps: int = 0
    cutoff_rollout_timeouts: int = 0
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
        self.cutoff_rollouts += other.cutoff_rollouts
        self.cutoff_rollout_steps += other.cutoff_rollout_steps
        self.cutoff_rollout_timeouts += other.cutoff_rollout_timeouts
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
            "traversal_cutoff_rollouts": self.cutoff_rollouts,
            "traversal_cutoff_rollout_steps": self.cutoff_rollout_steps,
            "traversal_cutoff_rollout_timeouts": self.cutoff_rollout_timeouts,
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
        outcome_sampling_epsilon: float = 0.0,
        outcome_sampling_value_clip: float | None = None,
        outcome_unsampled_regret: str = "negative_node_value",
        cutoff_value_mode: str = "score_diff",
        cutoff_rollouts: int = 0,
        cutoff_rollout_policy: str = "random",
        cutoff_rollout_max_steps: int = 10_000,
        opponent_policy: str = "network",
        league_advantage_networks: list[list[torch.nn.Module]] | None = None,
        self_play_anchor_probability: float = 0.0,
        self_play_current_weight: float = 0.5,
        self_play_recent_weight: float = 0.3,
        self_play_older_weight: float = 0.2,
        self_play_anchor_weight: float = 0.0,
        self_play_recent_window: int = 5,
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
        self.outcome_sampling_epsilon = min(1.0, max(0.0, float(outcome_sampling_epsilon)))
        self.outcome_sampling_value_clip = (
            None
            if outcome_sampling_value_clip is None
            else max(1.0e-9, float(outcome_sampling_value_clip))
        )
        self.outcome_unsampled_regret = outcome_unsampled_regret
        if self.outcome_unsampled_regret not in {"negative_node_value", "zero"}:
            raise ValueError("outcome_unsampled_regret must be 'negative_node_value' or 'zero'")
        self.cutoff_value_mode = cutoff_value_mode
        if self.cutoff_value_mode not in {"score_diff", "random_rollout"}:
            raise ValueError("cutoff_value_mode must be 'score_diff' or 'random_rollout'")
        self.cutoff_rollouts = max(0, int(cutoff_rollouts))
        self.cutoff_rollout_policy = cutoff_rollout_policy
        if self.cutoff_rollout_policy not in {"random", "safe_heuristic"}:
            raise ValueError("cutoff_rollout_policy must be 'random' or 'safe_heuristic'")
        self.cutoff_rollout_max_steps = max(1, int(cutoff_rollout_max_steps))
        self.opponent_policy = opponent_policy
        if self.opponent_policy not in {"network", "safe_heuristic", "self_play_league"}:
            raise ValueError(
                "opponent_policy must be 'network', 'safe_heuristic', or 'self_play_league'"
            )
        self.league_advantage_networks = league_advantage_networks or []
        self.self_play_anchor_probability = min(1.0, max(0.0, float(self_play_anchor_probability)))
        self.self_play_current_weight = max(0.0, float(self_play_current_weight))
        self.self_play_recent_weight = max(0.0, float(self_play_recent_weight))
        self.self_play_older_weight = max(0.0, float(self_play_older_weight))
        self.self_play_anchor_weight = max(0.0, float(self_play_anchor_weight))
        self.self_play_recent_window = max(0, int(self_play_recent_window))
        self.rng = rng or np.random.default_rng()
        self._safe_heuristic_rollout_bot = (
            SafeHeuristicBot() if self.cutoff_rollout_policy == "safe_heuristic" else None
        )
        self._safe_heuristic_opponent_bot = (
            SafeHeuristicBot()
            if self.opponent_policy == "safe_heuristic" or self.self_play_anchor_probability > 0.0
            else None
        )

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
            return self._cutoff_value(state, traverser, stats)
        if state.terminal:
            stats.terminals += 1
            self._record_endpoint(stats, depth)
            return float(state.score_diff(traverser))
        if self.max_depth is not None and depth >= self.max_depth:
            stats.depth_cutoffs += 1
            self._record_endpoint(stats, depth)
            return self._cutoff_value(state, traverser, stats)

        player = state.current_player
        fixed_action = self._fixed_opponent_action(state, player, traverser)
        if fixed_action is not None:
            unified_action = state.to_unified_action(fixed_action)
            swapped_deck_index = self._sample_deck_draw_chance(state, unified_action)
            state.push_action(fixed_action)
            try:
                return self._traverse(
                    state,
                    traverser,
                    iteration,
                    depth=depth + 1,
                    stats=stats,
                )
            finally:
                state.pop_action()
                if swapped_deck_index is not None:
                    state.swap_deck_cards(swapped_deck_index, len(state.deck) - 1)

        info_state, legal, policy = self._policy(state, player)
        self._record_strategy(info_state, legal, policy, player, traverser, iteration, depth, stats)

        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            stats.terminals += 1
            self._record_endpoint(stats, depth)
            return float(state.score_diff(traverser))

        sampling_policy = self._sampling_policy(policy, legal)
        action = self._sample_action(sampling_policy, legal_actions)
        local_action = state.from_unified_action(int(action))
        swapped_deck_index = self._sample_deck_draw_chance(state, int(action))
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
            if swapped_deck_index is not None:
                state.swap_deck_cards(swapped_deck_index, len(state.deck) - 1)

        stats.sampled_actions += 1
        action_prob = max(float(sampling_policy[action]), self.epsilon)
        sampled_action_value = child_value / action_prob
        if self.outcome_sampling_value_clip is not None:
            sampled_action_value = float(
                np.clip(
                    sampled_action_value,
                    -self.outcome_sampling_value_clip,
                    self.outcome_sampling_value_clip,
                )
            )
        node_value = float(policy[action]) * sampled_action_value

        if player == traverser:
            if self.outcome_unsampled_regret == "zero":
                regrets = np.zeros_like(policy, dtype=np.float32)
            else:
                regrets = np.where(legal, -node_value, 0.0).astype(np.float32)
            regrets[action] = np.float32(sampled_action_value - node_value)
            self.advantage_memory.add(
                TrainingSample(
                    info_state=info_state,
                    target=regrets,
                    legal_mask=legal,
                    iteration=iteration,
                    player=player,
                ),
                self.rng,
            )
            stats.advantage_samples += 1

        return node_value

    def _policy(self, state: GameState, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._policy_from_networks(self.advantage_networks, state, player)

    def _policy_from_networks(
        self,
        networks: list[torch.nn.Module],
        state: GameState,
        player: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info_state = encode_info_state(state, player)
        legal = np.asarray(state.unified_legal_mask(), dtype=bool)
        with torch.inference_mode():
            x = torch.as_tensor(info_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = networks[player](x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        policy = regret_matching(advantages, legal, self.epsilon).astype(np.float32)
        return info_state, legal, policy

    def _fixed_opponent_action(
        self,
        state: GameState,
        player: int,
        traverser: int,
    ) -> int | None:
        if player == traverser or self.opponent_policy == "network":
            return None
        if self.opponent_policy == "safe_heuristic":
            if self._safe_heuristic_opponent_bot is None:
                self._safe_heuristic_opponent_bot = SafeHeuristicBot()
            return self._safe_heuristic_opponent_bot.act(state)
        bucket = self._self_play_bucket()
        if bucket == "current":
            return None
        if bucket == "anchor":
            if self._safe_heuristic_opponent_bot is None:
                self._safe_heuristic_opponent_bot = SafeHeuristicBot()
            return self._safe_heuristic_opponent_bot.act(state)
        networks = self._self_play_snapshot_networks(bucket)
        if networks is None:
            return None
        _, legal, policy = self._policy_from_networks(networks, state, player)
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            return None
        unified_action = self._sample_action(policy, legal_actions)
        return state.from_unified_action(unified_action)

    def _self_play_bucket(self) -> str:
        if (
            self.self_play_anchor_probability > 0.0
            and self.rng.random() < self.self_play_anchor_probability
        ):
            return "anchor"
        recent_count = min(len(self.league_advantage_networks), self.self_play_recent_window)
        older_count = max(0, len(self.league_advantage_networks) - recent_count)
        labels = ["current", "recent", "older", "anchor"]
        weights = np.asarray(
            [
                self.self_play_current_weight,
                self.self_play_recent_weight if recent_count > 0 else 0.0,
                self.self_play_older_weight if older_count > 0 else 0.0,
                self.self_play_anchor_weight,
            ],
            dtype=np.float64,
        )
        total = float(weights.sum())
        if total <= 0.0:
            return "current"
        weights /= total
        return str(self.rng.choice(labels, p=weights))

    def _self_play_snapshot_networks(self, bucket: str) -> list[torch.nn.Module] | None:
        if not self.league_advantage_networks:
            return None
        recent_count = min(len(self.league_advantage_networks), self.self_play_recent_window)
        if bucket == "recent" and recent_count > 0:
            candidates = self.league_advantage_networks[-recent_count:]
        elif bucket == "older":
            candidates = self.league_advantage_networks[
                : max(0, len(self.league_advantage_networks) - recent_count)
            ]
        else:
            candidates = self.league_advantage_networks
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _sample_action(self, policy: np.ndarray, legal_actions: np.ndarray) -> int:
        probs = policy[legal_actions].astype(np.float64)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
        else:
            probs /= total
        return int(self.rng.choice(legal_actions, p=probs))

    def _sampling_policy(self, policy: np.ndarray, legal: np.ndarray) -> np.ndarray:
        legal_count = int(np.count_nonzero(legal))
        if legal_count <= 0:
            return np.zeros_like(policy, dtype=np.float32)
        if self.outcome_sampling_epsilon <= 0.0:
            return policy.astype(np.float32)
        uniform = legal.astype(np.float32) / float(legal_count)
        return (
            (1.0 - self.outcome_sampling_epsilon) * policy + self.outcome_sampling_epsilon * uniform
        ).astype(np.float32)

    def _cutoff_value(self, state: GameState, traverser: int, stats: TraversalStats) -> float:
        if self.cutoff_value_mode == "score_diff" or self.cutoff_rollouts <= 0:
            return float(state.score_diff(traverser))
        total = 0.0
        for _ in range(self.cutoff_rollouts):
            total += self._rollout_value(state, traverser, stats)
        return total / float(self.cutoff_rollouts)

    def _rollout_value(self, state: GameState, traverser: int, stats: TraversalStats) -> float:
        rollout_state = state.clone()
        steps = 0
        while not rollout_state.terminal and steps < self.cutoff_rollout_max_steps:
            action = self._rollout_action(rollout_state)
            if action is None:
                break
            unified_action = rollout_state.to_unified_action(action)
            self._sample_deck_draw_chance(rollout_state, unified_action)
            rollout_state.apply_action(action)
            steps += 1
        stats.cutoff_rollouts += 1
        stats.cutoff_rollout_steps += steps
        if not rollout_state.terminal:
            stats.cutoff_rollout_timeouts += 1
        return float(rollout_state.score_diff(traverser))

    def _rollout_action(self, state: GameState) -> int | None:
        if self.cutoff_rollout_policy == "safe_heuristic":
            if self._safe_heuristic_rollout_bot is None:
                self._safe_heuristic_rollout_bot = SafeHeuristicBot()
            return self._safe_heuristic_rollout_bot.act(state)
        legal_actions = state.unified_legal_actions()
        if not legal_actions:
            return None
        return state.from_unified_action(int(self.rng.choice(legal_actions)))

    def _sample_deck_draw_chance(self, state: GameState, unified_action: int) -> int | None:
        deck_draw_action = 2 * state.config.hand_size
        if state.phase != "draw" or unified_action != deck_draw_action or len(state.deck) <= 1:
            return None
        sampled_index = int(self.rng.integers(0, len(state.deck)))
        if sampled_index == len(state.deck) - 1:
            return None
        state.swap_deck_cards(sampled_index, len(state.deck) - 1)
        return sampled_index

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
            ),
            self.rng,
        )
        stats.strategy_samples += 1

    def _record_endpoint(self, stats: TraversalStats, depth: int) -> None:
        stats.endpoint_depth_sum += depth
        start = min(depth // 10 * 10, 100)
        key = "100_plus" if start >= 100 else f"{start}_{start + 9}"
        stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1
