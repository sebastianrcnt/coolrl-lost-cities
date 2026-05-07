from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from coolrl_lost_cities.games.classic.game import LostCitiesConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    experiment_name: str = "deep_cfr"
    max_iterations: int | None = None
    max_minutes: float | None = None
    seed: int = 1
    device: str = "auto"
    use_amp: bool = False

    @field_validator("device")
    @classmethod
    def _normalize_device(cls, value: str) -> str:
        token = value.strip().lower()
        if token in {"cuda", "cpu", "auto"}:
            return token
        return token


class RulesConfig(StrictModel):
    n_colors: int = 5
    n_ranks: int = 9
    min_rank: int = 2
    n_handshakes: int = 3
    hand_size: int = 8
    expedition_penalty: int = -20
    bonus_threshold: int = 8
    bonus_amount: int = 20

    def to_lost_cities_config(self, seed: int | None = None) -> LostCitiesConfig:
        config = LostCitiesConfig(
            n_colors=self.n_colors,
            n_ranks=self.n_ranks,
            min_rank=self.min_rank,
            n_handshakes=self.n_handshakes,
            hand_size=self.hand_size,
            expedition_penalty=self.expedition_penalty,
            bonus_threshold=self.bonus_threshold,
            bonus_amount=self.bonus_amount,
            seed=seed,
        )
        config.validate()
        return config


class EncodingConfig(StrictModel):
    derived_playability: bool = False
    slot_aware_playability: bool = False


class NetworkConfig(StrictModel):
    kind: str = "mlp"
    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "relu"
    color_attention_layers: int = 0
    color_attention_heads: int = 4

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"mlp", "color_shared"}:
            raise ValueError("must be 'mlp' or 'color_shared'")
        return token

    @field_validator("activation")
    @classmethod
    def _validate_activation(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"relu", "gelu"}:
            raise ValueError("must be 'relu' or 'gelu'")
        return token


class TraversalConfig(StrictModel):
    traversals_per_player: int = 8
    sampling_mode: str = "outcome"
    max_depth: int | None = None
    max_nodes_per_traversal: int | None = 10_000
    regret_matching_epsilon: float = 1.0e-8
    outcome_sampling_epsilon: float = 0.0
    outcome_sampling_value_clip: float | None = None
    outcome_unsampled_regret: str = "negative_node_value"
    cutoff_value_mode: str = "score_diff"
    cutoff_rollouts: int = 0
    cutoff_rollout_policy: str = "random"
    cutoff_rollout_max_steps: int = 10_000
    opponent_policy: str = "self_play_league"
    strategy_sample_interval: int = 1
    store_strategy_on_traverser_nodes: bool = True
    store_strategy_on_opponent_nodes: bool = True
    num_workers: int | str = 0
    worker_chunk_size: int = 4
    progress_every_traversals: int = 0
    endpoint_depth_bucket_width: int = 100
    endpoint_depth_bucket_max: int = 1000
    inference_backend: str = "local"

    @field_validator("sampling_mode")
    @classmethod
    def _validate_sampling_mode(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"outcome", "external"}:
            raise ValueError("must be 'outcome' or 'external'")
        return token

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
        if value not in {"network", "safe_heuristic", "self_play_league", "average_strategy"}:
            raise ValueError(
                "must be 'network', 'safe_heuristic', 'self_play_league', or 'average_strategy'"
            )
        return value

    @field_validator("inference_backend")
    @classmethod
    def _validate_inference_backend(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"local", "server"}:
            raise ValueError("must be 'local' or 'server'")
        return token

    def resolved_num_workers(self, batches: int | None = None) -> int:
        if isinstance(self.num_workers, str):
            token = self.num_workers.strip().lower()
            if token == "auto":
                guess = max(1, (os.cpu_count() or 2) // 2)
                return min(guess, batches) if batches is not None and batches > 0 else guess
            workers = max(0, int(token))
        else:
            workers = max(0, int(self.num_workers))
        return min(workers, batches) if batches is not None and batches > 0 else workers


class RegretMatchingConfig(StrictModel):
    all_negative_fallback: str = "uniform"

    @field_validator("all_negative_fallback")
    @classmethod
    def _validate_all_negative_fallback(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"uniform", "argmax_tiebreak"}:
            raise ValueError("must be 'uniform' or 'argmax_tiebreak'")
        return token


class TrainingWeightingConfig(StrictModel):
    mode: str = "none"
    lcfr_alpha: float = 1.0
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_gamma: float = 2.0

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"none", "lcfr", "dcfr"}:
            raise ValueError("must be 'none', 'lcfr', or 'dcfr'")
        return token


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
    advantage_batch_size: int = 256
    strategy_batch_size: int = 256
    advantage_updates_per_iteration: int = 64
    strategy_updates_per_iteration: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0


class MemoryConfig(StrictModel):
    advantage_capacity: int = 2_000_000
    strategy_capacity: int = 2_000_000


class CheckpointConfig(StrictModel):
    save_every: int = 1
    save_latest: bool = True
    progress_interval_seconds: float = 20.0
    exact_resume: bool = False


class EvaluationConfig(StrictModel):
    eval_every: int = 50
    games: int = 10
    opponents: tuple[str, ...] = ("random", "safe_heuristic")
    max_steps: int = 10_000
    on_max_steps: str = "score_diff"
    batch_size: int = 64
    device: str = "trainer"
    num_workers: int = 4

    @field_validator("on_max_steps")
    @classmethod
    def _validate_on_max_steps(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"score_diff", "loss", "draw"}:
            raise ValueError("must be 'score_diff', 'loss', or 'draw'")
        return token

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        token = value.strip().lower()
        if token not in {"trainer", "auto", "cpu", "cuda"}:
            raise ValueError("must be 'trainer', 'auto', 'cpu', or 'cuda'")
        return token

    def resolved_num_workers(self, opponent_count: int | None = None) -> int:
        workers = max(1, int(self.num_workers))
        if opponent_count is not None:
            workers = min(workers, max(1, int(opponent_count)))
        return workers


class InferenceServerConfig(StrictModel):
    device: str = "cuda"
    num_slots: int | None = None
    max_batch: int = 256
    batch_window_us: int = 200
    weight_sync_every: int = 1
    use_amp: bool = False

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        token = value.strip().lower()
        if token in {"auto", "cpu", "cuda"}:
            return token
        return token


class DeepCFRConfig(StrictModel):
    run: RunConfig = Field(default_factory=RunConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    traversal: TraversalConfig = Field(default_factory=TraversalConfig)
    regret_matching: RegretMatchingConfig = Field(default_factory=RegretMatchingConfig)
    training_weighting: TrainingWeightingConfig = Field(default_factory=TrainingWeightingConfig)
    self_play: SelfPlayLeagueConfig = Field(default_factory=SelfPlayLeagueConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference_server: InferenceServerConfig = Field(default_factory=InferenceServerConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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
