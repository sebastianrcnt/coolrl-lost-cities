from __future__ import annotations

from coolrl_lost_cities.games.classic.game import LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer


def test_deep_cfr_trainer_smoke_run() -> None:
    trainer = DeepCFRTrainer(
        DeepCFRConfig(
            iterations=1,
            traversals_per_iteration=1,
            rollouts_per_action=1,
            max_rollout_steps=64,
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=2,
            hidden_size=16,
            seed=23,
        ),
        LostCitiesConfig(seed=23),
    )

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].advantage_samples == 2
    assert metrics[0].strategy_samples == 2
    assert metrics[0].advantage_loss >= 0.0
    assert metrics[0].strategy_loss >= 0.0
