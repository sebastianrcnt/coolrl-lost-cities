from __future__ import annotations

import numpy as np
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
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
            save_every_iteration=False,
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
            save_every_iteration=False,
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


def test_deep_cfr_traverser_supports_outcome_sampling_and_rollout_cutoffs() -> None:
    trainer = DeepCFRTrainer(
        DeepCFRConfig(
            iterations=1,
            traversals_per_iteration=1,
            max_traversal_depth=1,
            max_nodes_per_traversal=32,
            outcome_sampling_epsilon=0.25,
            outcome_sampling_value_clip=100.0,
            outcome_unsampled_regret="zero",
            cutoff_value_mode="random_rollout",
            cutoff_rollouts=2,
            cutoff_rollout_policy="random",
            cutoff_rollout_max_steps=16,
            batch_size=2,
            hidden_size=16,
            seed=31,
            save_every_iteration=False,
        ),
        LostCitiesConfig(seed=31),
    )
    state = GameState.new_game(LostCitiesConfig(seed=31), seed=31)
    before = state.to_snapshot()
    traverser = DeepCFRTraverser(
        trainer.advantage_networks,
        trainer.advantage_memory,
        trainer.strategy_memory,
        device=trainer.device,
        action_size=trainer.action_size,
        max_depth=1,
        max_nodes=32,
        outcome_sampling_epsilon=0.25,
        outcome_sampling_value_clip=100.0,
        outcome_unsampled_regret="zero",
        cutoff_value_mode="random_rollout",
        cutoff_rollouts=2,
        cutoff_rollout_policy="random",
        cutoff_rollout_max_steps=16,
        rng=np.random.default_rng(31),
    )

    _, stats = traverser.traverse(state, traverser=0, iteration=1)

    assert state.to_snapshot() == before
    assert stats.depth_cutoffs > 0
    assert stats.cutoff_rollouts == stats.depth_cutoffs * 2
    assert stats.cutoff_rollout_steps > 0
    sample = trainer.advantage_memory.all()[0]
    unsampled_legal = sample.legal_mask.copy()
    unsampled_legal[np.nonzero(sample.target)[0]] = False
    assert np.all(sample.target[unsampled_legal] == 0.0)


def test_reservoir_memory_caps_samples_and_filters_player_batches() -> None:
    memory = ReservoirMemory(capacity=3)
    rng = np.random.default_rng(37)
    for index in range(10):
        memory.add(
            TrainingSample(
                info_state=np.asarray([index], dtype=np.float32),
                target=np.asarray([index], dtype=np.float32),
                legal_mask=np.asarray([True]),
                iteration=index,
                player=index % 2,
            ),
            rng,
        )

    assert len(memory) == 3
    assert memory.seen == 10
    player_one = memory.sample(8, rng, player=1)
    assert player_one
    assert all(sample.player == 1 for sample in player_one)


def test_deep_cfr_trainer_saves_loads_and_evaluates_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "deep_cfr"
    trainer = DeepCFRTrainer(
        DeepCFRConfig(
            iterations=1,
            traversals_per_iteration=1,
            max_traversal_depth=2,
            max_nodes_per_traversal=32,
            batch_size=2,
            hidden_size=16,
            seed=41,
            checkpoint_dir=str(checkpoint_dir),
            save_every_iteration=True,
            eval_every=1,
            eval_games=2,
            eval_opponents=("random",),
        ),
        LostCitiesConfig(seed=41),
    )

    metrics = trainer.train()
    latest = checkpoint_dir / "latest.pt"
    restored = DeepCFRTrainer(
        DeepCFRConfig(
            hidden_size=16,
            seed=41,
            checkpoint_dir=str(checkpoint_dir),
            save_every_iteration=False,
        ),
        LostCitiesConfig(seed=41),
    )
    restored.load_checkpoint(latest)

    assert latest.exists()
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "metrics.jsonl").exists()
    assert (checkpoint_dir / "runtime_progress.json").exists()
    assert (checkpoint_dir / "train.log").exists()
    assert restored.iteration == 1
    assert "eval_random_games" in metrics[0].eval_metrics
