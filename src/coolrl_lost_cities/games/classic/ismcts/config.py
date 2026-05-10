from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from coolrl_lost_cities.games.classic.deep_cfr.config import (
    CheckpointConfig,
    EncodingConfig,
    EvaluationConfig,
    NetworkConfig,
    OptimizationConfig,
    RulesConfig,
    RunConfig,
    StrictModel,
)


class MctsConfig(StrictModel):
    n_simulations: int = 50
    c_puct: float = 1.5
    max_depth: int = 200
    use_rollout_value: bool = True

    @field_validator("n_simulations", "max_depth")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class TemperatureConfig(StrictModel):
    training: float = 1.0
    eval: float = 0.0


class TrainingConfig(StrictModel):
    games_per_iter: int = 10
    gradient_steps_per_iter: int = 10
    batch_size: int = 128
    replay_capacity: int = 100_000

    @field_validator("games_per_iter", "gradient_steps_per_iter", "batch_size", "replay_capacity")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class IsMctsConfig(StrictModel):
    run: RunConfig = Field(default_factory=lambda: RunConfig(experiment_name="ismcts"))
    rules: RulesConfig = Field(default_factory=RulesConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    network: NetworkConfig = Field(
        default_factory=lambda: NetworkConfig(hidden_size=512, num_layers=3)
    )
    mcts: MctsConfig = Field(default_factory=MctsConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def config_from_dict(data: Mapping[str, Any]) -> IsMctsConfig:
    return IsMctsConfig.model_validate(data)


def load_config(path: str | Path) -> IsMctsConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return config_from_dict(data)
