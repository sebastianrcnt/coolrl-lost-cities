from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.memory import TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.traversal import run_cython_traversal_batch
from coolrl_lost_cities.games.classic.deep_cfr.traversal_stats import TraversalStats
from coolrl_lost_cities.games.classic.game import LostCitiesConfig

_TORCH_THREADS_CONFIGURED = False


def _configure_worker_torch_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    _TORCH_THREADS_CONFIGURED = True


@dataclass(frozen=True)
class TraversalWorkerBatch:
    player: int
    iteration: int
    seeds: list[int]
    config: dict[str, Any]
    game_config: dict[str, Any]
    input_dim: int
    action_size: int
    advantage_networks: list[dict[str, Any]]
    league_advantage_networks: list[list[dict[str, Any]]]
    worker_seed: int
    strategy_network: dict[str, Any] | None = None


@dataclass(frozen=True)
class TraversalWorkerResult:
    player: int
    stats: TraversalStats
    advantage_samples: list[TrainingSample]
    strategy_samples: list[TrainingSample]
    traversals: int


def run_traversal_worker_batch(batch: TraversalWorkerBatch) -> TraversalWorkerResult:
    _configure_worker_torch_threads()

    cfg = config_from_dict(batch.config)
    device = torch.device("cpu")
    networks = [
        DeepCFRMLP.from_config(batch.input_dim, batch.action_size, cfg.network).to(device)
        for _ in range(2)
    ]
    for network, state_dict in zip(networks, batch.advantage_networks, strict=True):
        network.load_state_dict(state_dict)
        network.eval()
    league_networks: list[list[torch.nn.Module]] = []
    for snapshot in batch.league_advantage_networks:
        snapshot_networks = [
            DeepCFRMLP.from_config(batch.input_dim, batch.action_size, cfg.network).to(device)
            for _ in range(2)
        ]
        for network, state_dict in zip(snapshot_networks, snapshot, strict=True):
            network.load_state_dict(state_dict)
            network.eval()
        league_networks.append(snapshot_networks)
    strategy_network: torch.nn.Module | None = None
    if batch.strategy_network is not None:
        strategy_network = DeepCFRMLP.from_config(
            batch.input_dim, batch.action_size, cfg.network
        ).to(device)
        strategy_network.load_state_dict(batch.strategy_network)
        strategy_network.eval()
    game_config = LostCitiesConfig(**batch.game_config)
    total_stats, advantage_samples, strategy_samples = run_cython_traversal_batch(
        networks,
        game_config,
        batch.seeds,
        batch.player,
        batch.iteration,
        device=device,
        strategy_network=strategy_network,
        action_size=batch.action_size,
        encoding=cfg.encoding,
        epsilon=cfg.traversal.regret_matching_epsilon,
        strategy_sample_interval=cfg.traversal.strategy_sample_interval,
        store_strategy_on_traverser_nodes=cfg.traversal.store_strategy_on_traverser_nodes,
        store_strategy_on_opponent_nodes=cfg.traversal.store_strategy_on_opponent_nodes,
        max_depth=cfg.traversal.max_depth,
        max_nodes=cfg.traversal.resolved_max_nodes(),
        sampling_mode=cfg.traversal.sampling_mode,
        outcome_sampling_epsilon=cfg.traversal.outcome_sampling_epsilon,
        outcome_sampling_value_clip=cfg.traversal.outcome_sampling_value_clip,
        outcome_unsampled_regret=cfg.traversal.outcome_unsampled_regret,
        cutoff_value_mode=cfg.traversal.cutoff_value_mode,
        cutoff_rollouts=cfg.traversal.cutoff_rollouts,
        cutoff_rollout_policy=cfg.traversal.cutoff_rollout_policy,
        cutoff_rollout_max_steps=cfg.traversal.cutoff_rollout_max_steps,
        opponent_policy=cfg.traversal.opponent_policy,
        all_negative_fallback=cfg.regret_matching.all_negative_fallback,
        league_advantage_networks=league_networks,
        self_play_anchor_probability=cfg.self_play.anchor_probability,
        self_play_current_weight=cfg.self_play.current_weight,
        self_play_recent_weight=cfg.self_play.recent_weight,
        self_play_older_weight=cfg.self_play.older_weight,
        self_play_anchor_weight=cfg.self_play.anchor_weight,
        self_play_recent_window=cfg.self_play.recent_window,
        endpoint_depth_bucket_width=cfg.traversal.endpoint_depth_bucket_width,
        endpoint_depth_bucket_max=cfg.traversal.endpoint_depth_bucket_max,
        seed=batch.worker_seed,
    )
    return TraversalWorkerResult(
        player=batch.player,
        stats=total_stats,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        traversals=len(batch.seeds),
    )
