from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepCFRConfig:
    iterations: int = 1
    traversals_per_iteration: int = 2
    max_traversal_depth: int | None = 8
    max_nodes_per_traversal: int | None = 10_000
    regret_matching_epsilon: float = 1.0e-8
    outcome_sampling_epsilon: float = 0.0
    outcome_sampling_value_clip: float | None = None
    outcome_unsampled_regret: str = "negative_node_value"
    cutoff_value_mode: str = "score_diff"
    cutoff_rollouts: int = 0
    cutoff_rollout_policy: str = "random"
    cutoff_rollout_max_steps: int = 10_000
    strategy_sample_interval: int = 1
    store_strategy_on_traverser_nodes: bool = True
    store_strategy_on_opponent_nodes: bool = True
    advantage_train_steps: int = 1
    strategy_train_steps: int = 1
    batch_size: int = 32
    hidden_size: int = 64
    learning_rate: float = 1.0e-3
    seed: int = 1
