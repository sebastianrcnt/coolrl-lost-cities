from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    iterations: int = 1
    seed: int = 1
    device: str = "cpu"


class NetworkConfig(StrictModel):
    hidden_size: int = 64


class TraversalConfig(StrictModel):
    traversals_per_iteration: int = 2
    max_depth: int | None = 8
    max_nodes: int | None = 10_000
    regret_matching_epsilon: float = 1.0e-8
    outcome_sampling_epsilon: float = 0.0
    outcome_sampling_value_clip: float | None = None
    outcome_unsampled_regret: str = "negative_node_value"
    cutoff_value_mode: str = "score_diff"
    cutoff_rollouts: int = 0
    cutoff_rollout_policy: str = "random"
    cutoff_rollout_max_steps: int = 10_000
    opponent_policy: str = "network"
    strategy_sample_interval: int = 1
    store_strategy_on_traverser_nodes: bool = True
    store_strategy_on_opponent_nodes: bool = True
    num_workers: int | str = 0
    worker_chunk_size: int = 4

    @field_validator("outcome_unsampled_regret")
    @classmethod
    def _validate_unsampled_regret(cls, value: str) -> str:
        if value not in {"negative_node_value", "zero"}:
            raise ValueError("must be 'negative_node_value' or 'zero'")
        return value

    @field_validator("cutoff_value_mode")
    @classmethod
    def _validate_cutoff_value_mode(cls, value: str) -> str:
        if value not in {"score_diff", "random_rollout"}:
            raise ValueError("must be 'score_diff' or 'random_rollout'")
        return value

    @field_validator("cutoff_rollout_policy")
    @classmethod
    def _validate_cutoff_rollout_policy(cls, value: str) -> str:
        if value not in {"random", "safe_heuristic"}:
            raise ValueError("must be 'random' or 'safe_heuristic'")
        return value

    @field_validator("opponent_policy")
    @classmethod
    def _validate_opponent_policy(cls, value: str) -> str:
        if value not in {"network", "safe_heuristic", "self_play_league"}:
            raise ValueError("must be 'network', 'safe_heuristic', or 'self_play_league'")
        return value

    def resolved_num_workers(self, batches: int | None = None) -> int:
        if isinstance(self.num_workers, str):
            token = self.num_workers.strip().lower()
            if token == "auto":
                guess = max(1, (os.cpu_count() or 2) // 2)
                return min(guess, batches) if batches is not None and batches > 0 else guess
            return max(0, int(token))
        return max(0, int(self.num_workers))


class SelfPlayLeagueConfig(StrictModel):
    snapshot_every: int = 1
    max_snapshots: int = 20
    anchor_probability: float = 0.0
    current_weight: float = 0.5
    recent_weight: float = 0.3
    older_weight: float = 0.2
    anchor_weight: float = 0.0
    recent_window: int = 5


class OptimizationConfig(StrictModel):
    advantage_train_steps: int = 1
    strategy_train_steps: int = 1
    batch_size: int = 32
    learning_rate: float = 1.0e-3


class MemoryConfig(StrictModel):
    advantage_capacity: int = 2_000_000
    strategy_capacity: int = 2_000_000


class CheckpointConfig(StrictModel):
    directory: str = "runs/deep_cfr/default"
    save_every_iteration: bool = True

    @property
    def path(self) -> Path:
        return Path(self.directory)


class EvaluationConfig(StrictModel):
    eval_every: int = 0
    games: int = 10
    opponents: tuple[str, ...] = ("random",)
    max_steps: int = 10_000


class DeepCFRConfig(StrictModel):
    run: RunConfig = Field(default_factory=RunConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    traversal: TraversalConfig = Field(default_factory=TraversalConfig)
    self_play: SelfPlayLeagueConfig = Field(default_factory=SelfPlayLeagueConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint.path


def config_from_dict(data: Mapping[str, Any]) -> DeepCFRConfig:
    return DeepCFRConfig.model_validate(data)


def load_config(path: str | Path) -> DeepCFRConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return config_from_dict(data)
