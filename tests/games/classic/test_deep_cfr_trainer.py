from __future__ import annotations

import numpy as np
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer
from coolrl_lost_cities.games.classic.deep_cfr.traverser import DeepCFRTraverser


def test_deep_cfr_trainer_smoke_run() -> None:
    trainer = DeepCFRTrainer(
        DeepCFRConfig(
            iterations=1,
            traversals_per_iteration=1,
            max_traversal_depth=3,
            max_nodes_per_traversal=64,
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
    assert metrics[0].advantage_samples > 0
    assert metrics[0].strategy_samples > 0
    assert metrics[0].traversal_nodes > 0
    assert metrics[0].traversal_max_depth_reached <= 3
    assert metrics[0].advantage_loss >= 0.0
    assert metrics[0].strategy_loss >= 0.0


def test_deep_cfr_recursive_traverser_restores_state_and_collects_samples() -> None:
    trainer = DeepCFRTrainer(
        DeepCFRConfig(
            iterations=1,
            traversals_per_iteration=1,
            max_traversal_depth=2,
            max_nodes_per_traversal=32,
            batch_size=2,
            hidden_size=16,
            seed=29,
        ),
        LostCitiesConfig(seed=29),
    )
    state = GameState.new_game(LostCitiesConfig(seed=29), seed=29)
    before = state.to_snapshot()
    traverser = DeepCFRTraverser(
        trainer.advantage_networks,
        trainer.advantage_memory,
        trainer.strategy_memory,
        device=trainer.device,
        action_size=trainer.action_size,
        max_depth=2,
        max_nodes=32,
        rng=np.random.default_rng(29),
    )

    value, stats = traverser.traverse(state, traverser=0, iteration=1)

    assert isinstance(value, float)
    assert state.to_snapshot() == before
    assert stats.nodes > 0
    assert stats.depth_cutoffs + stats.terminals + stats.node_limit_cutoffs > 0
    assert stats.strategy_samples > 0
    assert stats.advantage_samples > 0
    assert len(trainer.strategy_memory) == stats.strategy_samples
    assert len(trainer.advantage_memory) == stats.advantage_samples
    sample = trainer.advantage_memory.all()[0]
    assert sample.legal_mask.dtype == bool
    assert sample.target.shape == sample.legal_mask.shape
