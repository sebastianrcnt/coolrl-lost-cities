from __future__ import annotations

import numpy as np
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.benchmark import (
    benchmark_traversal,
    benchmark_traversal_modes,
)
from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig, load_config
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer
from coolrl_lost_cities.games.classic.deep_cfr.traverser import DeepCFRTraverser


def _deep_cfr_config(data: dict) -> DeepCFRConfig:
    return DeepCFRConfig.model_validate(data)


def test_deep_cfr_loads_smoke_yaml_config() -> None:
    config = load_config("configs/deep_cfr/smoke.yaml")

    assert config.run.iterations == 1
    assert config.network.hidden_size == 16
    assert config.traversal.traversals_per_iteration == 1
    assert config.checkpoint.directory == "runs/deep_cfr/smoke"


def test_deep_cfr_loads_mapped_legacy_reproduction_config() -> None:
    config = load_config(
        "configs/deep_cfr/pure_self_play_zero_pit_poc_full_depth_slot_aware_playability.yaml"
    )

    assert config.run.experiment_name.endswith("slot_aware_playability")
    assert config.run.seed == 79
    assert config.run.max_iterations is None
    assert config.run.max_hours == 4
    assert config.encoding.derived_playability is True
    assert config.encoding.slot_aware_playability is True
    assert config.network.hidden_size == 256
    assert config.network.num_layers == 3
    assert config.traversal.resolved_traversals_per_player() == 70
    assert config.traversal.max_depth is None
    assert config.traversal.resolved_max_nodes() == 1000
    assert config.traversal.resolved_worker_chunk_size() == 8
    assert config.optimization.resolved_advantage_batch_size() == 1024
    assert config.optimization.resolved_strategy_batch_size() == 1024
    assert config.optimization.resolved_advantage_train_steps() == 256
    assert config.optimization.resolved_strategy_train_steps() == 256
    assert config.optimization.weight_decay == 0.0001
    assert config.optimization.grad_clip == 1.0
    assert config.evaluation.on_max_steps == "score_diff"
    assert config.checkpoint.save_iteration_interval == 10


def test_deep_cfr_playability_encoding_extends_input_shape() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=61), seed=61)
    base_dim = input_dim(state)
    derived_config = _deep_cfr_config({"encoding": {"derived_playability": True}})
    slot_config = _deep_cfr_config(
        {"encoding": {"derived_playability": True, "slot_aware_playability": True}}
    )

    derived_dim = input_dim(state, derived_config.encoding)
    slot_dim = input_dim(state, slot_config.encoding)

    assert derived_dim == base_dim + state.config.n_colors * 19 + 3
    assert slot_dim == derived_dim + state.config.hand_size * 12
    assert encode_info_state(state, 0, slot_config.encoding).shape == (slot_dim,)


def test_deep_cfr_trainer_uses_playability_encoding() -> None:
    config = _deep_cfr_config(
        {
            "run": {"iterations": 1, "seed": 62},
            "encoding": {"derived_playability": True, "slot_aware_playability": True},
            "network": {"hidden_size": 16},
            "traversal": {"traversals_per_iteration": 1, "max_depth": 1, "max_nodes": 16},
            "optimization": {
                "advantage_train_steps": 1,
                "strategy_train_steps": 1,
                "batch_size": 2,
            },
            "checkpoint": {"save_every_iteration": False},
        }
    )
    game_config = LostCitiesConfig(seed=62)
    trainer = DeepCFRTrainer(config, game_config)

    metrics = trainer.train()

    probe = GameState.new_game(game_config, seed=62)
    assert trainer.input_dim == input_dim(probe, config.encoding)
    assert metrics[0].advantage_samples > 0


def test_deep_cfr_trainer_smoke_run() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"iterations": 1, "seed": 23},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 1,
                    "max_depth": 3,
                    "max_nodes": 64,
                },
                "optimization": {
                    "advantage_train_steps": 1,
                    "strategy_train_steps": 1,
                    "batch_size": 2,
                },
                "checkpoint": {"save_every_iteration": False},
            }
        ),
        LostCitiesConfig(seed=23),
    )

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].advantage_samples > 0
    assert metrics[0].strategy_samples > 0
    assert metrics[0].traversal_nodes > 0
    assert metrics[0].traversal_max_depth_reached <= 3
    assert metrics[0].traversal_endpoints > 0
    assert metrics[0].traversal_avg_endpoint_depth >= 0.0
    assert metrics[0].advantage_loss >= 0.0
    assert metrics[0].strategy_loss >= 0.0


def test_deep_cfr_recursive_traverser_restores_state_and_collects_samples() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"iterations": 1, "seed": 29},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 1,
                    "max_depth": 2,
                    "max_nodes": 32,
                },
                "optimization": {"batch_size": 2},
                "checkpoint": {"save_every_iteration": False},
            }
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
        _deep_cfr_config(
            {
                "run": {"iterations": 1, "seed": 31},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 1,
                    "max_depth": 1,
                    "max_nodes": 32,
                    "outcome_sampling_epsilon": 0.25,
                    "outcome_sampling_value_clip": 100.0,
                    "outcome_unsampled_regret": "zero",
                    "cutoff_value_mode": "random_rollout",
                    "cutoff_rollouts": 2,
                    "cutoff_rollout_policy": "random",
                    "cutoff_rollout_max_steps": 16,
                },
                "optimization": {"batch_size": 2},
                "checkpoint": {"save_every_iteration": False},
            }
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
        _deep_cfr_config(
            {
                "run": {"iterations": 1, "seed": 41},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 1,
                    "max_depth": 2,
                    "max_nodes": 32,
                },
                "optimization": {"batch_size": 2},
                "checkpoint": {
                    "directory": str(checkpoint_dir),
                    "save_every_iteration": True,
                },
                "evaluation": {"eval_every": 1, "games": 2, "opponents": ("random",)},
            }
        ),
        LostCitiesConfig(seed=41),
    )

    metrics = trainer.train()
    latest = checkpoint_dir / "latest.pt"
    restored = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"seed": 41},
                "network": {"hidden_size": 16},
                "checkpoint": {
                    "directory": str(checkpoint_dir),
                    "save_every_iteration": False,
                },
            }
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


def test_deep_cfr_trainer_multiprocessing_smoke_run(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"iterations": 1, "seed": 43},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 2,
                    "max_depth": 2,
                    "max_nodes": 32,
                    "num_workers": 2,
                    "worker_chunk_size": 1,
                },
                "optimization": {"batch_size": 2},
                "checkpoint": {
                    "directory": str(tmp_path / "mp"),
                    "save_every_iteration": False,
                },
            }
        ),
        LostCitiesConfig(seed=43),
    )

    metrics = trainer.train()

    assert metrics[0].traversal_nodes > 0
    assert metrics[0].advantage_samples > 0


def test_deep_cfr_traversal_benchmark_smoke() -> None:
    result = benchmark_traversal(
        _deep_cfr_config(
            {
                "run": {"seed": 47},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_iteration": 1, "max_depth": 2},
                "checkpoint": {"save_every_iteration": False},
            }
        )
    )

    assert result["traversal_nodes"] > 0
    assert result["nodes_per_second"] > 0.0
    comparison = benchmark_traversal_modes(
        _deep_cfr_config(
            {
                "run": {"seed": 48},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_iteration": 1, "max_depth": 2},
                "checkpoint": {"save_every_iteration": False},
            }
        )
    )
    assert comparison["summary"]["speedup"] > 0.0


def test_deep_cfr_self_play_league_records_snapshots(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"iterations": 2, "seed": 53},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 1,
                    "max_depth": 2,
                    "max_nodes": 32,
                    "opponent_policy": "self_play_league",
                },
                "self_play": {
                    "snapshot_every": 1,
                    "max_snapshots": 1,
                    "anchor_probability": 1.0,
                },
                "optimization": {"batch_size": 2},
                "checkpoint": {
                    "directory": str(tmp_path / "league"),
                    "save_every_iteration": False,
                },
            }
        ),
        LostCitiesConfig(seed=53),
    )

    metrics = trainer.train()

    assert len(metrics) == 2
    assert len(trainer.self_play_league_snapshots) == 1


def test_deep_cfr_weighted_self_play_league_uses_snapshot_bucket(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"iterations": 2, "seed": 59},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_iteration": 1,
                    "max_depth": 2,
                    "max_nodes": 32,
                    "opponent_policy": "self_play_league",
                },
                "self_play": {
                    "snapshot_every": 1,
                    "max_snapshots": 2,
                    "current_weight": 0.0,
                    "recent_weight": 1.0,
                    "older_weight": 0.0,
                    "anchor_weight": 0.0,
                    "recent_window": 1,
                },
                "optimization": {"batch_size": 2},
                "checkpoint": {
                    "directory": str(tmp_path / "weighted-league"),
                    "save_every_iteration": False,
                },
            }
        ),
        LostCitiesConfig(seed=59),
    )

    metrics = trainer.train()

    assert len(metrics) == 2
    assert len(trainer.self_play_league_snapshots) == 2
    assert metrics[1].traversal_nodes > 0
