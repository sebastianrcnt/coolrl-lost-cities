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
    rollout_policy: str = "random"
    parallel_simulations: int = 8
    virtual_loss_value: float = 1.0
    eval_with_mcts: bool = True
    eval_n_simulations: int = 0
    root_dirichlet_alpha: float = 0.0
    root_dirichlet_epsilon: float = 0.0
    # Divisor applied to Q values inside PUCT to bring them onto roughly the
    # same scale as the exploration bonus. With value_scale=100 score units,
    # raw Q can swing ±100 while c_puct * prior * sqrt(N) is ~1-10, so a single
    # bad backup permanently kills an action. Setting q_scale=100 normalizes Q
    # to ~[-1, 1] (consistent with AlphaZero's convention).
    q_scale: float = 100.0

    @field_validator("n_simulations", "max_depth", "parallel_simulations")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("rollout_policy")
    @classmethod
    def _rollout_policy(cls, value: str) -> str:
        if value not in {"random", "heuristic_balanced"}:
            raise ValueError("rollout_policy must be 'random' or 'heuristic_balanced'")
        return value


class TemperatureConfig(StrictModel):
    training: float = 1.0
    eval: float = 0.0


class TrainingConfig(StrictModel):
    games_per_iter: int = 10
    gradient_steps_per_iter: int = 10
    batch_size: int = 128
    replay_capacity: int = 100_000
    interleave_games: int = 8
    interleave_max_batch: int = 64
    num_workers: int = 1
    worker_device: str = "cpu"
    # Multiplier on the value-head MSE loss (already normalized by value_scale**2).
    # Default 1.0 keeps current behavior; raising it (e.g. 50-100) makes the value
    # head learn faster relative to policy loss. Useful when value_prediction_error
    # is large but loss/value is tiny because of the normalization.
    value_loss_weight: float = 1.0
    # Optional KL anchor to a reference (e.g. behavior-cloned) policy. The
    # reference network is loaded once at trainer start and frozen; on every
    # gradient step we add `kl_anchor_beta * KL(current || reference)` to the
    # loss. Anchors self-play training to the pretrained policy and prevents
    # catastrophic forgetting / drift to weak self-play equilibria.
    kl_anchor_ckpt: str | None = None
    kl_anchor_beta: float = 0.0
    # Mirror-descent target mixing for policy loss. Alternative to kl_anchor;
    # blends MCTS visit distribution with the reference (BC) policy in log
    # space, then trains the network to match. pi_target = softmax(
    #   alpha * log(pi_mcts) + (1 - alpha) * log(pi_ref)
    # ). Anneal alpha from low (rely on BC) to high (rely on MCTS) over
    # training. Requires kl_anchor_ckpt to be set as the reference source.
    md_target_ref_ckpt: str | None = None
    md_target_alpha_start: float = 0.3
    md_target_alpha_end: float = 0.8
    md_target_alpha_iters: int = 500

    @field_validator(
        "games_per_iter",
        "gradient_steps_per_iter",
        "batch_size",
        "replay_capacity",
        "interleave_games",
        "interleave_max_batch",
        "num_workers",
    )
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
