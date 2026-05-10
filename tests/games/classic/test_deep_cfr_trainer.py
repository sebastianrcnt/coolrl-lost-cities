from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.deep_cfr.traversal import (
    CythonDeepCFRTraverser,
    run_cython_traversal_batch,
)
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.benchmark import (
    benchmark_traversal,
    benchmark_traversal_modes,
)
from coolrl_lost_cities.games.classic.deep_cfr.checkpoints import load_checkpoint
from coolrl_lost_cities.games.classic.deep_cfr.cli import (
    _kebab_slug,
    _resolve_run_dir,
    _train_overrides_from_args,
    _with_overrides,
)
from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig, load_config
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import evaluate_strategy_network
from coolrl_lost_cities.games.classic.deep_cfr.interleaved_traversal import (
    _apply_first_open_prior,
    _first_open_recoverable_score,
    run_interleaved_traversal_batch,
)
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.trainer import (
    DeepCFRTrainer,
    eval_skipped_warning,
)


def _deep_cfr_config(data: dict) -> DeepCFRConfig:
    return DeepCFRConfig.model_validate(data)


def test_deep_cfr_loads_smoke_yaml_config() -> None:
    config = load_config("configs/deep_cfr/smoke.yaml")

    assert config.run.max_iterations == 1
    assert config.network.hidden_size == 16
    assert config.traversal.traversals_per_player == 1
    assert config.run.experiment_name == "smoke"


def test_deep_cfr_loads_mapped_legacy_reproduction_config() -> None:
    config = load_config("configs/archive/deep-cfr-selfplay-full-depth-slot-playability.yaml")

    assert config.run.experiment_name.endswith("slot-playability")
    assert config.run.seed == 79
    assert config.run.max_iterations is None
    assert config.run.max_minutes == 240
    assert config.encoding.derived_playability is True
    assert config.encoding.slot_aware_playability is True
    assert config.network.hidden_size == 256
    assert config.network.num_layers == 3
    assert config.traversal.traversals_per_player == 70
    assert config.traversal.sampling_mode == "outcome"
    assert config.traversal.max_depth is None
    assert config.traversal.max_nodes_per_traversal == 1000
    assert config.traversal.worker_chunk_size == 8
    assert config.traversal.progress_every_traversals == 10
    assert config.optimization.advantage_batch_size == 1024
    assert config.optimization.strategy_batch_size == 1024
    assert config.optimization.advantage_updates_per_iteration == 256
    assert config.optimization.strategy_updates_per_iteration == 256
    assert config.optimization.weight_decay == 0.0001
    assert config.optimization.grad_clip == 1.0
    assert config.evaluation.on_max_steps == "score_diff"
    assert config.evaluation.batch_size == 64
    assert config.evaluation.device == "trainer"
    assert config.evaluation.resolved_num_workers() == 4
    assert config.regret_matching.all_negative_fallback == "uniform"
    assert config.training_weighting.mode == "none"
    assert config.checkpoint.save_every == 10


def test_deep_cfr_train_cli_accepts_run_and_traversal_config_overrides() -> None:
    args = type(
        "Args",
        (),
        {
            "config_overrides": [
                "run.max_iterations=1",
                "run.max_minutes=null",
                "traversal.traversals_per_player=1",
                "traversal.num_workers=0",
                "regret_matching.all_negative_fallback=argmax_tiebreak",
                "training_weighting.mode=lcfr",
                "checkpoint.save_latest=false",
                "checkpoint.save_every=0",
            ],
        },
    )()
    config = load_config("configs/archive/deep-cfr-selfplay-full-depth-slot-playability.yaml")

    overridden = _with_overrides(config, _train_overrides_from_args(args))

    assert overridden.run.max_iterations == 1
    assert overridden.run.max_minutes is None
    assert overridden.traversal.traversals_per_player == 1
    assert overridden.traversal.resolved_num_workers() == 0
    assert overridden.regret_matching.all_negative_fallback == "argmax_tiebreak"
    assert overridden.training_weighting.mode == "lcfr"
    assert overridden.checkpoint.save_every == 0
    assert overridden.checkpoint.save_latest is False


def test_deep_cfr_config_accepts_external_sampling_mode() -> None:
    config = _deep_cfr_config(
        {
            "traversal": {
                "sampling_mode": "external",
                "store_strategy_on_traverser_nodes": False,
            }
        }
    )

    assert config.traversal.sampling_mode == "external"


def test_deep_cfr_config_accepts_interleaved_scheduler() -> None:
    config = _deep_cfr_config(
        {"traversal": {"scheduler": "interleaved", "opponent_policy": "average_strategy"}}
    )

    assert config.traversal.scheduler == "interleaved"
    assert config.traversal.opponent_policy == "average_strategy"


def test_deep_cfr_config_rejects_unsupported_interleaved_options() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "opponent_policy='network', 'average_strategy', 'discard_only', or 'heuristic_balanced'"
        ),
    ):
        _deep_cfr_config(
            {"traversal": {"scheduler": "interleaved", "opponent_policy": "self_play_league"}}
        )
    with pytest.raises(ValueError, match="requires inference_backend='local'"):
        _deep_cfr_config(
            {
                "traversal": {
                    "scheduler": "interleaved",
                    "opponent_policy": "network",
                    "inference_backend": "server",
                }
            }
        )


def test_deep_cfr_config_accepts_discard_only_with_interleaved() -> None:
    config = _deep_cfr_config(
        {"traversal": {"scheduler": "interleaved", "opponent_policy": "discard_only"}}
    )
    assert config.traversal.opponent_policy == "discard_only"


def test_deep_cfr_config_accepts_heuristic_balanced_with_interleaved() -> None:
    config = _deep_cfr_config(
        {"traversal": {"scheduler": "interleaved", "opponent_policy": "heuristic_balanced"}}
    )
    assert config.traversal.opponent_policy == "heuristic_balanced"


def test_deep_cfr_config_rejects_discard_only_with_recursive() -> None:
    with pytest.raises(ValueError, match="discard_only.*scheduler='interleaved'"):
        _deep_cfr_config(
            {"traversal": {"scheduler": "recursive", "opponent_policy": "discard_only"}}
        )


def test_deep_cfr_train_cli_checkpoint_save_overrides() -> None:
    args = type(
        "Args",
        (),
        {
            "config_overrides": [
                "checkpoint.save_latest=true",
                "checkpoint.save_every=1",
            ],
        },
    )()

    overridden = _with_overrides(DeepCFRConfig(), _train_overrides_from_args(args))

    assert overridden.checkpoint.save_latest is True
    assert overridden.checkpoint.save_every == 1


def test_deep_cfr_train_cli_accepts_generic_config_overrides() -> None:
    args = type(
        "Args",
        (),
        {
            "config_overrides": [
                "traversal.sampling_mode=external",
                "traversal.store_strategy_on_traverser_nodes=false",
                "traversal.max_depth=null",
                "optimization.advantage_batch_size=64",
                "checkpoint.save_latest=true",
            ],
        },
    )()

    overridden = _with_overrides(DeepCFRConfig(), _train_overrides_from_args(args))

    assert overridden.traversal.sampling_mode == "external"
    assert overridden.traversal.max_depth is None
    assert overridden.optimization.advantage_batch_size == 64
    assert overridden.checkpoint.save_latest is True


def test_deep_cfr_iteration_weights_use_sample_age() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 12},
                "network": {"hidden_size": 16},
                "checkpoint": {"save_every": 0},
                "training_weighting": {"mode": "lcfr", "lcfr_alpha": 1.0},
            }
        ),
        LostCitiesConfig(seed=12),
    )
    trainer.iteration = 10

    weights = trainer._iteration_weights(torch.tensor([1.0, 5.0, 10.0], device=trainer.device), 1.0)

    assert np.allclose(weights.detach().cpu().numpy(), np.asarray([0.1, 0.5, 1.0]))
    assert trainer.config.training_weighting.mode == "lcfr"


def test_deep_cfr_batched_evaluation_matches_batch_size_one() -> None:
    config = _deep_cfr_config(
        {
            "network": {"hidden_size": 16},
            "encoding": {"derived_playability": True, "slot_aware_playability": True},
        }
    )
    game_config = LostCitiesConfig(seed=123)
    state = GameState.new_game(game_config, seed=123)
    network = DeepCFRMLP.from_config(
        input_dim(state, config.encoding),
        game_config.action_size,
        config.network,
    )
    network.eval()
    kwargs = {
        "strategy_network": network,
        "config": game_config,
        "games": 8,
        "seed": 55,
        "opponent": "random",
        "device": "cpu",
        "max_steps": 200,
        "encoding": config.encoding,
    }

    batch_one = evaluate_strategy_network(**kwargs, batch_size=1)
    batched = evaluate_strategy_network(**kwargs, batch_size=64)

    for key in [
        "games",
        "wins0",
        "wins1",
        "draws",
        "avg_score0",
        "avg_score1",
        "avg_score_diff0",
        "avg_game_length",
        "max_step_timeouts",
        "play_action_rate",
    ]:
        assert batched[key] == batch_one[key]
    assert np.isclose(batched["policy_entropy"], batch_one["policy_entropy"])


def test_deep_cfr_resolve_run_dir_uses_keep_flag_and_kebab_slug() -> None:
    config = _deep_cfr_config({"run": {"experiment_name": "Color Shared Attn v2"}})

    tmp_path = _resolve_run_dir(config, keep=False)
    keep_path = _resolve_run_dir(config, keep=True)

    assert tmp_path.parent == Path("runs/tmp")
    assert keep_path.parent == Path("runs")
    assert tmp_path.name.endswith("_color-shared-attn-v2")
    assert keep_path.name.endswith("_color-shared-attn-v2")
    assert _kebab_slug("Foo BAR_baz!!") == "foo-bar-baz"
    assert _kebab_slug("   ") == "run"


def test_deep_cfr_playability_encoding_extends_input_shape() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=61), seed=61)
    base_dim = input_dim(state)
    derived_config = _deep_cfr_config({"encoding": {"derived_playability": True}})
    slot_config = _deep_cfr_config(
        {"encoding": {"derived_playability": True, "slot_aware_playability": True}}
    )

    derived_dim = input_dim(state, derived_config.encoding)
    slot_dim = input_dim(state, slot_config.encoding)

    assert derived_dim == base_dim + state.config.n_colors * 15 + 3
    assert slot_dim == derived_dim + state.config.hand_size * 6
    assert encode_info_state(state, 0, slot_config.encoding).shape == (slot_dim,)


def test_deep_cfr_slot_aware_playability_encoding_zero_fills_empty_hand_slots() -> None:
    config = _deep_cfr_config(
        {"encoding": {"derived_playability": True, "slot_aware_playability": True}}
    )
    state = GameState.new_game(LostCitiesConfig(seed=63), seed=63)
    state.phase = "draw"
    state.pending_discarded_color = -1
    state.apply_action(0)

    encoded = encode_info_state(state, 0, config.encoding)

    assert np.isfinite(encoded).all()


def test_deep_cfr_trainer_uses_playability_encoding() -> None:
    config = _deep_cfr_config(
        {
            "run": {"max_iterations": 1, "seed": 62},
            "encoding": {"derived_playability": True, "slot_aware_playability": True},
            "network": {"hidden_size": 16},
            "traversal": {
                "traversals_per_player": 1,
                "max_depth": 1,
                "max_nodes_per_traversal": 16,
            },
            "optimization": {
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
                "advantage_batch_size": 2,
                "strategy_batch_size": 2,
            },
            "checkpoint": {"save_every": 0},
        }
    )
    game_config = LostCitiesConfig(seed=62)
    trainer = DeepCFRTrainer(config, game_config)

    metrics = trainer.train()

    probe = GameState.new_game(game_config, seed=62)
    assert trainer.input_dim == input_dim(probe, config.encoding)
    assert metrics[0].advantage_samples > 0


def test_deep_cfr_trainer_forwards_metrics_to_extra_trackers(tmp_path) -> None:
    events: list[tuple[str, object]] = []

    class _CaptureTracker:
        def log_event(self, message: str) -> None:
            events.append(("event", message))

        def log_metrics(self, metrics: dict, *, step: int) -> None:
            events.append(("metrics", step))

        def close(self) -> None:
            events.append(("close", None))

    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 71},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 16,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=71),
        run_dir=tmp_path / "extra-tracker",
        extra_trackers=[_CaptureTracker()],
    )

    trainer.train()

    assert any(kind == "metrics" for kind, _ in events)
    assert any(kind == "event" for kind, _ in events)
    assert ("close", None) in events
    assert (tmp_path / "extra-tracker" / "metrics.jsonl").exists()


def test_deep_cfr_trainer_smoke_run() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 23},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 3,
                    "max_nodes_per_traversal": 64,
                },
                "optimization": {
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                },
                "checkpoint": {"save_every": 0},
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


def test_deep_cfr_trainer_interleaved_scheduler_smoke_run(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 24},
                "network": {"hidden_size": 16},
                "traversal": {
                    "scheduler": "interleaved",
                    "opponent_policy": "network",
                    "traversals_per_player": 2,
                    "max_depth": 3,
                    "max_nodes_per_traversal": 64,
                    "interleave_width": 4,
                    "interleave_max_batch": 8,
                },
                "optimization": {
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                },
                "checkpoint": {"save_every": 0, "save_latest": False},
                "evaluation": {"eval_every": 0},
            }
        ),
        LostCitiesConfig(seed=24),
        run_dir=tmp_path / "interleaved",
    )

    metrics = trainer.train()
    runtime = metrics[0].runtime_metrics

    assert len(metrics) == 1
    assert metrics[0].advantage_samples > 0
    assert metrics[0].strategy_samples > 0
    assert metrics[0].traversal_nodes > 0
    assert runtime["interleaved/batches"] > 0
    assert runtime["interleaved/requests"] > 0
    assert runtime["interleaved/max_batch_size"] >= 1
    assert runtime["interleaved/avg_batch_size"] >= 1.0


def test_deep_cfr_trainer_discard_only_opponent_smoke_run(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 25},
                "network": {"hidden_size": 16},
                "traversal": {
                    "scheduler": "interleaved",
                    "opponent_policy": "discard_only",
                    "traversals_per_player": 2,
                    "max_depth": 3,
                    "max_nodes_per_traversal": 64,
                    "interleave_width": 4,
                    "interleave_max_batch": 8,
                },
                "optimization": {
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                },
                "checkpoint": {"save_every": 0, "save_latest": False},
                "evaluation": {"eval_every": 0},
            }
        ),
        LostCitiesConfig(seed=25),
        run_dir=tmp_path / "discard_only",
    )

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].advantage_samples > 0
    assert metrics[0].traversal_nodes > 0


def test_deep_cfr_trainer_heuristic_balanced_opponent_smoke_run(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 27},
                "network": {"hidden_size": 16},
                "traversal": {
                    "scheduler": "interleaved",
                    "opponent_policy": "heuristic_balanced",
                    "traversals_per_player": 2,
                    "max_depth": 3,
                    "max_nodes_per_traversal": 64,
                    "interleave_width": 4,
                    "interleave_max_batch": 8,
                },
                "optimization": {
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                },
                "checkpoint": {"save_every": 0, "save_latest": False},
                "evaluation": {"eval_every": 0},
            }
        ),
        LostCitiesConfig(seed=27),
        run_dir=tmp_path / "heuristic_balanced",
    )

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].advantage_samples > 0
    assert metrics[0].traversal_nodes > 0


def test_deep_cfr_interleaved_scheduler_matches_recursive_single_traversal() -> None:
    config = _deep_cfr_config(
        {
            "run": {"seed": 26},
            "network": {"hidden_size": 16},
            "traversal": {
                "opponent_policy": "network",
                "max_depth": 3,
                "max_nodes_per_traversal": 64,
                "outcome_unsampled_regret": "zero",
            },
        }
    )
    game_config = LostCitiesConfig(seed=26)
    probe = GameState.new_game(game_config, seed=26)
    action_size = game_config.action_size
    torch.manual_seed(26)
    networks = [
        DeepCFRMLP.from_config(input_dim(probe, config.encoding), action_size, config.network)
        for _ in range(2)
    ]
    for network in networks:
        network.eval()
    common = {
        "device": torch.device("cpu"),
        "action_size": action_size,
        "encoding": config.encoding,
        "epsilon": config.traversal.regret_matching_epsilon,
        "strategy_sample_interval": config.traversal.strategy_sample_interval,
        "store_strategy_on_traverser_nodes": config.traversal.store_strategy_on_traverser_nodes,
        "store_strategy_on_opponent_nodes": config.traversal.store_strategy_on_opponent_nodes,
        "max_depth": config.traversal.max_depth,
        "max_nodes": config.traversal.max_nodes_per_traversal,
        "outcome_sampling_epsilon": config.traversal.outcome_sampling_epsilon,
        "outcome_sampling_value_clip": config.traversal.outcome_sampling_value_clip,
        "endpoint_depth_bucket_width": config.traversal.endpoint_depth_bucket_width,
        "endpoint_depth_bucket_max": config.traversal.endpoint_depth_bucket_max,
        "seed": 2601,
    }

    recursive_stats, recursive_advantage, recursive_strategy = run_cython_traversal_batch(
        networks,
        game_config,
        [260],
        0,
        1,
        **common,
        strategy_network=None,
        sampling_mode=config.traversal.sampling_mode,
        outcome_unsampled_regret=config.traversal.outcome_unsampled_regret,
        cutoff_value_mode=config.traversal.cutoff_value_mode,
        cutoff_rollouts=config.traversal.cutoff_rollouts,
        cutoff_rollout_policy=config.traversal.cutoff_rollout_policy,
        cutoff_rollout_max_steps=config.traversal.cutoff_rollout_max_steps,
        opponent_policy=config.traversal.opponent_policy,
        all_negative_fallback=config.regret_matching.all_negative_fallback,
        league_advantage_networks=[],
        self_play_anchor_probability=config.self_play.anchor_probability,
        self_play_current_weight=config.self_play.current_weight,
        self_play_recent_weight=config.self_play.recent_weight,
        self_play_older_weight=config.self_play.older_weight,
        self_play_anchor_weight=config.self_play.anchor_weight,
        self_play_recent_window=config.self_play.recent_window,
    )
    interleaved_stats, interleaved_advantage, interleaved_strategy, _runtime = (
        run_interleaved_traversal_batch(
            networks,
            None,
            game_config,
            [260],
            0,
            1,
            **common,
            outcome_unsampled_regret=config.traversal.outcome_unsampled_regret,
            opponent_policy=config.traversal.opponent_policy,
            interleave_width=4,
            interleave_max_batch=8,
        )
    )

    assert interleaved_stats.to_dict() == recursive_stats.to_dict()
    assert len(interleaved_advantage) == len(recursive_advantage)
    assert len(interleaved_strategy) == len(recursive_strategy)
    assert np.allclose(
        [sample.target.sum() for sample in interleaved_advantage],
        [sample.target.sum() for sample in recursive_advantage],
        atol=1.0e-5,
    )
    for interleaved_sample, recursive_sample in zip(
        interleaved_advantage, recursive_advantage, strict=True
    ):
        assert np.allclose(interleaved_sample.target, recursive_sample.target, atol=1.0e-5)
    assert np.allclose(
        [sample.target.sum() for sample in interleaved_strategy],
        [sample.target.sum() for sample in recursive_strategy],
        atol=1.0e-6,
    )


def test_deep_cfr_interleaved_scheduler_matches_average_strategy_opponent() -> None:
    config = _deep_cfr_config(
        {
            "run": {"seed": 27},
            "network": {"hidden_size": 16},
            "traversal": {
                "opponent_policy": "average_strategy",
                "store_strategy_on_opponent_nodes": False,
                "max_depth": 4,
                "max_nodes_per_traversal": 64,
            },
        }
    )
    game_config = LostCitiesConfig(seed=27)
    probe = GameState.new_game(game_config, seed=27)
    action_size = game_config.action_size
    torch.manual_seed(27)
    networks = [
        DeepCFRMLP.from_config(input_dim(probe, config.encoding), action_size, config.network)
        for _ in range(2)
    ]
    strategy_network = DeepCFRMLP.from_config(
        input_dim(probe, config.encoding), action_size, config.network
    )
    for network in [*networks, strategy_network]:
        network.eval()
    common = {
        "device": torch.device("cpu"),
        "action_size": action_size,
        "encoding": config.encoding,
        "epsilon": config.traversal.regret_matching_epsilon,
        "strategy_sample_interval": config.traversal.strategy_sample_interval,
        "store_strategy_on_traverser_nodes": config.traversal.store_strategy_on_traverser_nodes,
        "store_strategy_on_opponent_nodes": config.traversal.store_strategy_on_opponent_nodes,
        "max_depth": config.traversal.max_depth,
        "max_nodes": config.traversal.max_nodes_per_traversal,
        "outcome_sampling_epsilon": config.traversal.outcome_sampling_epsilon,
        "outcome_sampling_value_clip": config.traversal.outcome_sampling_value_clip,
        "endpoint_depth_bucket_width": config.traversal.endpoint_depth_bucket_width,
        "endpoint_depth_bucket_max": config.traversal.endpoint_depth_bucket_max,
        "seed": 2701,
    }

    recursive_stats, recursive_advantage, recursive_strategy = run_cython_traversal_batch(
        networks,
        game_config,
        [270],
        0,
        1,
        **common,
        strategy_network=strategy_network,
        sampling_mode=config.traversal.sampling_mode,
        outcome_unsampled_regret=config.traversal.outcome_unsampled_regret,
        cutoff_value_mode=config.traversal.cutoff_value_mode,
        cutoff_rollouts=config.traversal.cutoff_rollouts,
        cutoff_rollout_policy=config.traversal.cutoff_rollout_policy,
        cutoff_rollout_max_steps=config.traversal.cutoff_rollout_max_steps,
        opponent_policy=config.traversal.opponent_policy,
        all_negative_fallback=config.regret_matching.all_negative_fallback,
        league_advantage_networks=[],
        self_play_anchor_probability=config.self_play.anchor_probability,
        self_play_current_weight=config.self_play.current_weight,
        self_play_recent_weight=config.self_play.recent_weight,
        self_play_older_weight=config.self_play.older_weight,
        self_play_anchor_weight=config.self_play.anchor_weight,
        self_play_recent_window=config.self_play.recent_window,
    )
    interleaved_stats, interleaved_advantage, interleaved_strategy, _runtime = (
        run_interleaved_traversal_batch(
            networks,
            strategy_network,
            game_config,
            [270],
            0,
            1,
            **common,
            outcome_unsampled_regret=config.traversal.outcome_unsampled_regret,
            opponent_policy=config.traversal.opponent_policy,
            interleave_width=4,
            interleave_max_batch=8,
        )
    )

    assert interleaved_stats.to_dict() == recursive_stats.to_dict()
    assert len(interleaved_advantage) == len(recursive_advantage)
    assert len(interleaved_strategy) == len(recursive_strategy)
    assert np.allclose(
        [sample.target.sum() for sample in interleaved_advantage],
        [sample.target.sum() for sample in recursive_advantage],
        atol=1.0e-5,
    )
    assert np.allclose(
        [sample.target.sum() for sample in interleaved_strategy],
        [sample.target.sum() for sample in recursive_strategy],
        atol=1.0e-6,
    )


def test_deep_cfr_trainer_supports_lcfr_and_dcfr_loss_weighting() -> None:
    for mode in ("lcfr", "dcfr"):
        trainer = DeepCFRTrainer(
            _deep_cfr_config(
                {
                    "run": {"max_iterations": 1, "seed": 24},
                    "network": {"hidden_size": 16},
                    "traversal": {
                        "traversals_per_player": 1,
                        "max_depth": 2,
                        "max_nodes_per_traversal": 32,
                    },
                    "optimization": {
                        "advantage_updates_per_iteration": 1,
                        "strategy_updates_per_iteration": 1,
                        "advantage_batch_size": 2,
                        "strategy_batch_size": 2,
                    },
                    "training_weighting": {"mode": mode},
                    "checkpoint": {"save_every": 0},
                }
            ),
            LostCitiesConfig(seed=24),
        )

        metrics = trainer.train()

        assert len(metrics) == 1
        assert metrics[0].advantage_loss >= 0.0
        assert metrics[0].strategy_loss >= 0.0


def test_deep_cfr_trainer_amp_cpu_falls_back_to_fp32() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 25, "device": "cpu", "use_amp": True},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
                },
                "optimization": {
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                },
                "checkpoint": {"save_every": 0},
                "evaluation": {"eval_every": 0},
            }
        ),
        LostCitiesConfig(seed=25),
        device="cpu",
    )

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].runtime_metrics["amp/grad_scale"] == 1.0
    assert metrics[0].runtime_metrics["amp/nonfinite_loss_count"] == 0
    assert metrics[0].advantage_loss >= 0.0
    assert metrics[0].strategy_loss >= 0.0


def test_deep_cfr_trainer_amp_cuda_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 26, "device": "cuda", "use_amp": True},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
                },
                "optimization": {
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                },
                "checkpoint": {"save_every": 0},
                "evaluation": {"eval_every": 0},
            }
        ),
        LostCitiesConfig(seed=26),
        device="cuda",
    )

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].runtime_metrics["amp/grad_scale"] > 0.0
    assert metrics[0].runtime_metrics["amp/nonfinite_loss_count"] == 0
    assert np.isfinite(metrics[0].advantage_loss)
    assert np.isfinite(metrics[0].strategy_loss)


def test_deep_cfr_cython_traverser_restores_state_and_collects_samples() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 29},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=29),
    )
    state = GameState.new_game(LostCitiesConfig(seed=29), seed=29)
    before = state.to_snapshot()
    traverser = CythonDeepCFRTraverser(
        trainer.advantage_networks,
        device=trainer.device,
        action_size=trainer.action_size,
        max_depth=2,
        max_nodes=32,
        seed=29,
    )

    value, stats = traverser.traverse(state, traverser=0, iteration=1)
    advantage_samples, strategy_samples = traverser.drain_samples()

    assert isinstance(value, float)
    assert state.to_snapshot() == before
    assert stats.nodes > 0
    assert stats.depth_cutoffs + stats.terminals + stats.node_limit_cutoffs > 0
    assert stats.strategy_samples > 0
    assert stats.advantage_samples > 0
    assert len(strategy_samples) == stats.strategy_samples
    assert len(advantage_samples) == stats.advantage_samples
    sample = advantage_samples[0]
    assert sample.legal_mask.dtype == bool
    assert sample.target.shape == sample.legal_mask.shape


def test_deep_cfr_cython_traverser_supports_outcome_sampling_and_rollout_cutoffs() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 31},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 1,
                    "max_nodes_per_traversal": 32,
                    "outcome_sampling_epsilon": 0.25,
                    "outcome_sampling_value_clip": 100.0,
                    "outcome_unsampled_regret": "zero",
                    "cutoff_value_mode": "random_rollout",
                    "cutoff_rollouts": 2,
                    "cutoff_rollout_policy": "random",
                    "cutoff_rollout_max_steps": 16,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=31),
    )
    state = GameState.new_game(LostCitiesConfig(seed=31), seed=31)
    before = state.to_snapshot()
    traverser = CythonDeepCFRTraverser(
        trainer.advantage_networks,
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
        seed=31,
    )

    _, stats = traverser.traverse(state, traverser=0, iteration=1)
    advantage_samples, _ = traverser.drain_samples()

    assert state.to_snapshot() == before
    assert stats.depth_cutoffs > 0
    assert stats.cutoff_rollouts == stats.depth_cutoffs * 2
    assert stats.cutoff_rollout_steps > 0
    sample = advantage_samples[0]
    unsampled_legal = sample.legal_mask.copy()
    unsampled_legal[np.nonzero(sample.target)[0]] = False
    assert np.all(sample.target[unsampled_legal] == 0.0)


def test_deep_cfr_cython_traverser_supports_external_sampling() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 33},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "sampling_mode": "external",
                    "store_strategy_on_traverser_nodes": False,
                    "max_depth": 1,
                    "max_nodes_per_traversal": 64,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=33),
    )
    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].advantage_samples > 0
    state = GameState.new_game(LostCitiesConfig(seed=33), seed=33)
    before = state.to_snapshot()
    traverser = CythonDeepCFRTraverser(
        trainer.advantage_networks,
        device=trainer.device,
        action_size=trainer.action_size,
        sampling_mode="external",
        max_depth=1,
        max_nodes=64,
        seed=33,
    )

    value, stats = traverser.traverse(state, traverser=0, iteration=1)
    advantage_samples, strategy_samples = traverser.drain_samples()

    assert isinstance(value, float)
    assert state.to_snapshot() == before
    assert stats.nodes > 0
    assert stats.depth_cutoffs > 0
    assert stats.advantage_samples > 0
    assert stats.strategy_samples > 0
    assert len(advantage_samples) == stats.advantage_samples
    assert len(strategy_samples) == stats.strategy_samples
    sample = advantage_samples[0]
    assert sample.legal_mask.dtype == bool
    assert sample.target.shape == sample.legal_mask.shape
    assert np.count_nonzero(sample.target[sample.legal_mask]) > 1


def test_deep_cfr_cython_traverser_records_regret_fallback_metrics() -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 37},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_player": 1, "max_depth": 1},
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
                "regret_matching": {"all_negative_fallback": "argmax_tiebreak"},
            }
        ),
        LostCitiesConfig(seed=37),
    )
    for network in trainer.advantage_networks:
        for parameter in network.parameters():
            parameter.data.zero_()
    state = GameState.new_game(LostCitiesConfig(seed=37), seed=37)
    traverser = CythonDeepCFRTraverser(
        trainer.advantage_networks,
        device=trainer.device,
        action_size=trainer.action_size,
        max_depth=1,
        all_negative_fallback="argmax_tiebreak",
        seed=37,
    )

    _, stats = traverser.traverse(state, traverser=0, iteration=1)
    metrics = stats.to_dict()
    action_count_sum = (
        stats.regret_fallback_action_play_existing
        + stats.regret_fallback_action_open_new
        + stats.regret_fallback_action_discard
        + stats.regret_fallback_action_draw_deck
        + stats.regret_fallback_action_draw_pile
    )

    assert stats.regret_matching_decisions > 0
    assert stats.regret_fallback_count > 0
    assert action_count_sum == stats.regret_fallback_count
    assert metrics["traversal_regret_fallback_rate"] > 0.0
    assert "traversal_regret_fallback_open_new_selected_rate" in metrics
    assert metrics["traversal_regret_fallback_legal_actions_mean"] > 0.0
    assert "traversal_regret_fallback_open_new_available_rate" in metrics
    assert "traversal_regret_fallback_open_new_selection_over_availability" in metrics
    assert "traversal_regret_fallback_depth_bucket_0_49" in metrics
    assert "traversal_regret_fallback_opened_colors_count_0" in metrics
    assert metrics["traversal_regret_fallback_argmax_tie_rate"] > 0.0
    assert metrics["traversal_regret_fallback_argmax_tie_size_mean"] > 0.0


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


def test_reservoir_memory_filters_first_open_batches() -> None:
    memory = ReservoirMemory()
    rng = np.random.default_rng(37)
    for index in range(6):
        memory.add(
            TrainingSample(
                info_state=np.asarray([index], dtype=np.float32),
                target=np.asarray([index], dtype=np.float32),
                legal_mask=np.asarray([True]),
                iteration=index,
                player=0,
                is_first_open=index % 2 == 0,
            ),
            rng,
        )

    first_open = memory.sample(8, rng, first_open_only=True)

    assert len(first_open) == 3
    assert all(sample.is_first_open for sample in first_open)
    assert memory.count(first_open_only=True) == 3


def test_reservoir_memory_updates_first_open_index_on_replacement() -> None:
    memory = ReservoirMemory(capacity=1)
    replacement_rng = np.random.default_rng(1)
    sample_rng = np.random.default_rng(37)
    memory.add(
        TrainingSample(
            info_state=np.asarray([1], dtype=np.float32),
            target=np.asarray([1], dtype=np.float32),
            legal_mask=np.asarray([True]),
            iteration=1,
            player=0,
            is_first_open=True,
        ),
        replacement_rng,
    )
    memory.add(
        TrainingSample(
            info_state=np.asarray([2], dtype=np.float32),
            target=np.asarray([2], dtype=np.float32),
            legal_mask=np.asarray([True]),
            iteration=2,
            player=0,
            is_first_open=False,
        ),
        replacement_rng,
    )

    assert memory.count(first_open_only=True) == 0
    with pytest.raises(ValueError, match="cannot sample from empty memory"):
        memory.sample(1, sample_rng, first_open_only=True)


def test_deep_cfr_trainer_can_oversample_first_open_advantage_batches(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"seed": 41},
                "network": {"hidden_size": 16},
                "optimization": {
                    "advantage_batch_size": 4,
                    "advantage_first_open_fraction": 0.5,
                },
            }
        ),
        run_dir=tmp_path,
        device="cpu",
    )
    for index in range(8):
        trainer.advantage_memories[0].add(
            TrainingSample(
                info_state=np.asarray([index], dtype=np.float32),
                target=np.asarray([index], dtype=np.float32),
                legal_mask=np.asarray([True]),
                iteration=index,
                player=0,
                is_first_open=index in {1, 3},
            ),
            trainer.rng,
        )

    batch = trainer._sample_advantage_batch(player=0)

    assert len(batch) == 4
    assert sum(sample.is_first_open for sample in batch) >= 2


def test_deep_cfr_trainer_saves_loads_and_evaluates_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "deep_cfr"
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 41},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 1},
                "evaluation": {"eval_every": 1, "games": 2, "opponents": ("random",)},
            }
        ),
        LostCitiesConfig(seed=41),
        run_dir=checkpoint_dir,
    )

    metrics = trainer.train()
    latest = checkpoint_dir / "latest.pt"
    restored = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"seed": 41},
                "network": {"hidden_size": 16},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=41),
        run_dir=checkpoint_dir,
    )
    restored.load_checkpoint(latest)

    assert latest.exists()
    assert load_checkpoint(latest)["resume_semantics"] == "networks_optimizers_iteration_only"
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "metrics.jsonl").exists()
    assert (checkpoint_dir / "runtime_progress.json").exists()
    assert (checkpoint_dir / "train.log").exists()
    train_log = (checkpoint_dir / "train.log").read_text(encoding="utf-8")
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", train_log)
    assert "Iteration complete" in train_log
    assert "[i=1]" in train_log
    assert "reservoir memories and RNG state are not restored" in train_log
    assert restored.iteration == 1
    assert "eval/random/games" in metrics[0].eval_metrics
    assert "eval/random/play_action_rate" in metrics[0].eval_metrics
    assert "eval/random/policy_entropy" in metrics[0].eval_metrics
    assert "eval/random/avg_opened_colors" in metrics[0].eval_metrics
    assert "eval/random/bad_open_actions" in metrics[0].eval_metrics
    assert "eval/random/positive_expedition_rate" in metrics[0].eval_metrics
    assert (
        "eval/random/first_open_recoverable_score_mean_for_positive_final"
        in metrics[0].eval_metrics
    )


def test_deep_cfr_trainer_always_saves_latest_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "latest-each-iteration"
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 42},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_player": 1, "max_depth": 1},
                "checkpoint": {"save_every": 10},
            }
        ),
        LostCitiesConfig(seed=42),
        run_dir=checkpoint_dir,
    )

    trainer.train()

    assert (checkpoint_dir / "latest.pt").exists()
    assert not (checkpoint_dir / "iteration_00001.pt").exists()


def test_deep_cfr_exact_resume_is_explicitly_not_implemented(tmp_path) -> None:
    checkpoint_dir = tmp_path / "exact"
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 44},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_player": 1, "max_depth": 1},
                "checkpoint": {"save_every": 1},
            }
        ),
        LostCitiesConfig(seed=44),
        run_dir=checkpoint_dir,
    )
    trainer.train()
    exact = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "network": {"hidden_size": 16},
                "checkpoint": {"exact_resume": True},
            }
        ),
        LostCitiesConfig(seed=44),
        run_dir=checkpoint_dir,
    )

    try:
        exact.load_checkpoint(checkpoint_dir / "latest.pt")
    except NotImplementedError as exc:
        assert "exact_resume" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError")


def test_deep_cfr_trainer_multiprocessing_smoke_run(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 43},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
                    "num_workers": 8,
                    "worker_chunk_size": 1,
                    "progress_every_traversals": 1,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=43),
        run_dir=tmp_path / "mp",
    )

    metrics = trainer.train()
    train_log = (tmp_path / "mp" / "train.log").read_text(encoding="utf-8")

    assert metrics[0].traversal_nodes > 0
    assert metrics[0].advantage_samples > 0
    assert "Traversal multiprocessing enabled" in train_log
    assert "Traversal worker count capped" in train_log
    assert "Traversal multiprocessing progress" in train_log


def test_deep_cfr_traversal_benchmark_smoke() -> None:
    result = benchmark_traversal(
        _deep_cfr_config(
            {
                "run": {"seed": 47},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_player": 1, "max_depth": 2},
                "checkpoint": {"save_every": 0},
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
                "traversal": {"traversals_per_player": 1, "max_depth": 2},
                "checkpoint": {"save_every": 0},
            }
        )
    )
    assert comparison["summary"]["speedup"] > 0.0


def test_deep_cfr_self_play_league_records_snapshots(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 2, "seed": 53},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
                    "opponent_policy": "self_play_league",
                },
                "self_play": {
                    "snapshot_every": 1,
                    "max_snapshots": 1,
                    "anchor_probability": 1.0,
                },
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=53),
        run_dir=tmp_path / "league",
    )

    metrics = trainer.train()

    assert len(metrics) == 2
    assert len(trainer.self_play_league_snapshots) == 1


def test_deep_cfr_weighted_self_play_league_uses_snapshot_bucket(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 2, "seed": 59},
                "network": {"hidden_size": 16},
                "traversal": {
                    "traversals_per_player": 1,
                    "max_depth": 2,
                    "max_nodes_per_traversal": 32,
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
                "optimization": {"advantage_batch_size": 2, "strategy_batch_size": 2},
                "checkpoint": {"save_every": 0},
            }
        ),
        LostCitiesConfig(seed=59),
        run_dir=tmp_path / "weighted-league",
    )

    metrics = trainer.train()

    assert len(metrics) == 2
    assert len(trainer.self_play_league_snapshots) == 2
    assert metrics[1].traversal_nodes > 0


def test_eval_skipped_warning_returns_none_when_eval_disabled() -> None:
    assert eval_skipped_warning(0, max_iterations=10, eval_every=0) is None
    assert eval_skipped_warning(0, max_iterations=10, eval_every=-5) is None


def test_eval_skipped_warning_returns_none_when_max_iterations_unbounded() -> None:
    assert eval_skipped_warning(0, max_iterations=None, eval_every=50) is None


def test_eval_skipped_warning_returns_none_when_eval_will_run_within_budget() -> None:
    assert eval_skipped_warning(0, max_iterations=50, eval_every=50) is None
    assert eval_skipped_warning(0, max_iterations=51, eval_every=50) is None
    assert eval_skipped_warning(0, max_iterations=200, eval_every=50) is None


def test_eval_skipped_warning_returns_none_when_starting_just_before_an_eval() -> None:
    # Resuming at iteration=49: next iter (50) is an eval.
    assert eval_skipped_warning(49, max_iterations=50, eval_every=50) is None


def test_eval_skipped_warning_warns_when_max_below_first_eval() -> None:
    warning = eval_skipped_warning(0, max_iterations=10, eval_every=50)
    assert warning is not None
    assert "max_iterations=10" in warning
    assert "iteration 50" in warning


def test_eval_skipped_warning_warns_on_resume_when_no_more_evals_fit() -> None:
    # Resumed at iter 100 (just past the last scheduled eval).
    # Next eval is 150, but we only have budget up to 110.
    warning = eval_skipped_warning(100, max_iterations=110, eval_every=50)
    assert warning is not None
    assert "iteration 150" in warning


def test_eval_skipped_warning_no_warn_when_resume_lands_exactly_on_next_eval() -> None:
    # iter=50 means iteration 50 already evaluated; next is 100. Budget 100 fits.
    assert eval_skipped_warning(50, max_iterations=100, eval_every=50) is None


def test_eval_skipped_warning_warns_when_eval_every_one_but_max_is_zero() -> None:
    # Pathological: max_iterations=0 means no iterations will run.
    warning = eval_skipped_warning(0, max_iterations=0, eval_every=1)
    assert warning is not None


def test_deep_cfr_trainer_logs_eval_skipped_warning_on_start(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 90},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_player": 1, "max_depth": 1},
                "optimization": {
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                },
                "checkpoint": {"save_every": 0, "save_latest": False},
                "evaluation": {"eval_every": 50, "games": 2, "opponents": ["random"]},
            }
        ),
        LostCitiesConfig(seed=90),
        run_dir=tmp_path,
    )

    trainer.train()
    log_text = (tmp_path / "train.log").read_text()
    assert "WARNING evaluation will not run" in log_text


def test_deep_cfr_trainer_does_not_log_eval_warning_when_eval_disabled(tmp_path) -> None:
    trainer = DeepCFRTrainer(
        _deep_cfr_config(
            {
                "run": {"max_iterations": 1, "seed": 91},
                "network": {"hidden_size": 16},
                "traversal": {"traversals_per_player": 1, "max_depth": 1},
                "optimization": {
                    "advantage_batch_size": 2,
                    "strategy_batch_size": 2,
                    "advantage_updates_per_iteration": 1,
                    "strategy_updates_per_iteration": 1,
                },
                "checkpoint": {"save_every": 0, "save_latest": False},
                "evaluation": {"eval_every": 0},
            }
        ),
        LostCitiesConfig(seed=91),
        run_dir=tmp_path,
    )

    trainer.train()
    log_text = (tmp_path / "train.log").read_text()
    assert "WARNING evaluation will not run" not in log_text


def test_first_open_prior_overrides_unsampled_play_targets_with_signed_alpha() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=123), seed=123)
    player = int(state.current_player)
    card_action_size = state.config.hand_size * 2
    legal_mask = np.zeros(card_action_size + 5, dtype=bool)
    play_actions: list[int] = []
    for slot, card in enumerate(state.hand_slots(player)):
        if card is None:
            continue
        play_actions.append(2 * slot)
        legal_mask[2 * slot] = True
        legal_mask[2 * slot + 1] = True
    assert play_actions, "fresh game state should have legal play actions"

    sampled_action = play_actions[0]
    target = np.zeros(legal_mask.shape[0], dtype=np.float32)
    target[sampled_action] = 7.0

    alpha = 5.0
    _apply_first_open_prior(target, state, player, legal_mask, sampled_action, alpha)

    assert target[sampled_action] == pytest.approx(7.0)

    expeditions = state.expeditions[player]
    hand = state.hand_slots(player)
    for action in play_actions:
        if action == sampled_action:
            continue
        card = hand[action // 2]
        color = int(card.color)
        if expeditions[color]:
            assert target[action] == 0.0
            continue
        score = _first_open_recoverable_score(state, player, color)
        expected = alpha if score >= 0.0 else -alpha
        assert target[action] == pytest.approx(expected)
        assert target[action + 1] == 0.0


def test_first_open_prior_zero_alpha_is_noop() -> None:
    state = GameState.new_game(LostCitiesConfig(seed=7), seed=7)
    player = int(state.current_player)
    card_action_size = state.config.hand_size * 2
    legal_mask = np.zeros(card_action_size + 5, dtype=bool)
    for slot, card in enumerate(state.hand_slots(player)):
        if card is None:
            continue
        legal_mask[2 * slot] = True
        legal_mask[2 * slot + 1] = True
    target = np.zeros(legal_mask.shape[0], dtype=np.float32)
    _apply_first_open_prior(target, state, player, legal_mask, sampled_action=0, alpha=0.0)
    assert np.all(target == 0.0)


def test_interleaved_regret_matching_argmax_tiebreak_concentrates_on_best() -> None:
    from coolrl_lost_cities.games.classic.deep_cfr.interleaved_traversal import _regret_matching

    advantages = np.array([-1.0, -0.5, -0.5, -2.0, -0.5], dtype=np.float32)
    legal_mask = np.array([True, True, True, True, True])

    uniform_policy, fallback_u, _, _ = _regret_matching(
        advantages, legal_mask, epsilon=1.0e-8, fallback_mode="uniform"
    )
    argmax_policy, fallback_a, tie_size, _ = _regret_matching(
        advantages, legal_mask, epsilon=1.0e-8, fallback_mode="argmax_tiebreak"
    )

    assert fallback_u is True
    assert fallback_a is True
    assert tie_size == 3
    assert np.allclose(uniform_policy, np.full(5, 0.2, dtype=np.float32))
    expected_argmax = np.zeros(5, dtype=np.float32)
    expected_argmax[1] = 1.0
    assert np.allclose(argmax_policy, expected_argmax)


def test_interleaved_regret_matching_no_fallback_unchanged_by_mode() -> None:
    from coolrl_lost_cities.games.classic.deep_cfr.interleaved_traversal import _regret_matching

    advantages = np.array([1.0, 3.0, 0.0, 2.0], dtype=np.float32)
    legal_mask = np.array([True, True, True, True])
    policy_uniform, fallback_u, _, _ = _regret_matching(
        advantages, legal_mask, epsilon=1.0e-8, fallback_mode="uniform"
    )
    policy_argmax, fallback_a, _, _ = _regret_matching(
        advantages, legal_mask, epsilon=1.0e-8, fallback_mode="argmax_tiebreak"
    )
    assert fallback_u is False
    assert fallback_a is False
    assert np.allclose(policy_uniform, policy_argmax)
    assert np.allclose(policy_uniform.sum(), 1.0)


def test_evaluation_opponents_for_iteration_core_only_when_extended_disabled() -> None:
    from coolrl_lost_cities.games.classic.deep_cfr.config import EvaluationConfig

    cfg = EvaluationConfig(
        eval_every=5,
        opponents=("random", "discard_only", "heuristic_cautious"),
        extended_eval_every=0,
        extended_opponents=("heuristic_balanced",),
    )
    assert cfg.opponents_for_iteration(0) == ()
    assert cfg.opponents_for_iteration(3) == ()
    assert cfg.opponents_for_iteration(5) == (
        "random",
        "discard_only",
        "heuristic_cautious",
    )
    assert cfg.opponents_for_iteration(50) == (
        "random",
        "discard_only",
        "heuristic_cautious",
    )


def test_evaluation_opponents_for_iteration_extends_on_extended_cadence() -> None:
    from coolrl_lost_cities.games.classic.deep_cfr.config import EvaluationConfig

    cfg = EvaluationConfig(
        eval_every=5,
        opponents=("random", "discard_only", "heuristic_cautious"),
        extended_eval_every=50,
        extended_opponents=(
            "heuristic_balanced",
            "heuristic_aggressive",
            "heuristic_noisy",
        ),
    )
    assert cfg.opponents_for_iteration(5) == (
        "random",
        "discard_only",
        "heuristic_cautious",
    )
    assert cfg.opponents_for_iteration(45) == (
        "random",
        "discard_only",
        "heuristic_cautious",
    )
    assert cfg.opponents_for_iteration(50) == (
        "random",
        "discard_only",
        "heuristic_cautious",
        "heuristic_balanced",
        "heuristic_aggressive",
        "heuristic_noisy",
    )
    assert cfg.opponents_for_iteration(100) == (
        "random",
        "discard_only",
        "heuristic_cautious",
        "heuristic_balanced",
        "heuristic_aggressive",
        "heuristic_noisy",
    )


def test_evaluation_opponents_for_iteration_dedupes_overlap() -> None:
    from coolrl_lost_cities.games.classic.deep_cfr.config import EvaluationConfig

    cfg = EvaluationConfig(
        eval_every=5,
        opponents=("random", "heuristic_cautious"),
        extended_eval_every=10,
        extended_opponents=("heuristic_cautious", "heuristic_balanced"),
    )
    assert cfg.opponents_for_iteration(10) == (
        "random",
        "heuristic_cautious",
        "heuristic_balanced",
    )
