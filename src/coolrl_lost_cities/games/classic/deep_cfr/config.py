from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    advantage_memory_capacity: int = 2_000_000
    strategy_memory_capacity: int = 2_000_000
    advantage_train_steps: int = 1
    strategy_train_steps: int = 1
    batch_size: int = 32
    hidden_size: int = 64
    learning_rate: float = 1.0e-3
    seed: int = 1
    checkpoint_dir: str = "runs/deep_cfr/default"
    save_every_iteration: bool = True
    eval_every: int = 0
    eval_games: int = 10
    eval_opponents: tuple[str, ...] = ("random",)
    eval_max_steps: int = 10_000
    num_workers: int = 0
    traversal_worker_chunk_size: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)


def config_from_dict(data: dict[str, Any]) -> DeepCFRConfig:
    values = dict(data)
    if "eval_opponents" in values:
        values["eval_opponents"] = tuple(values["eval_opponents"])
    return DeepCFRConfig(**values)
