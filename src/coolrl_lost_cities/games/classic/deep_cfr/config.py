from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepCFRConfig:
    iterations: int = 1
    traversals_per_iteration: int = 2
    rollouts_per_action: int = 1
    max_rollout_steps: int = 512
    advantage_train_steps: int = 1
    strategy_train_steps: int = 1
    batch_size: int = 32
    hidden_size: int = 64
    learning_rate: float = 1.0e-3
    seed: int = 1
