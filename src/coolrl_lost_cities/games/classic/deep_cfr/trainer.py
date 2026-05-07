from __future__ import annotations

import copy
import json
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from coolrl_lost_cities.games.classic.deep_cfr.config import (
    DeepCFRConfig,
    EncodingConfig,
    NetworkConfig,
)
from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import evaluate_strategy_network
from coolrl_lost_cities.games.classic.deep_cfr.inference_server import (
    BatchStatsMessage,
    InferenceServerController,
)
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.tracking import (
    CompositeRunTracker,
    ConsoleRunTracker,
    FileRunTracker,
    RunTracker,
)
from coolrl_lost_cities.games.classic.deep_cfr.traversal import run_cython_traversal_batch
from coolrl_lost_cities.games.classic.deep_cfr.traversal_stats import TraversalStats
from coolrl_lost_cities.games.classic.deep_cfr.workers import (
    TraversalWorkerBatch,
    initialize_traversal_worker,
    run_traversal_worker_batch,
)
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig


def _resolve_torch_device(device: str) -> torch.device:
    token = device.strip().lower()
    if token == "auto":
        token = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(token)


@dataclass(frozen=True)
class EvaluationWorkerJob:
    opponent: str
    strategy_state_dict: dict[str, torch.Tensor]
    game_config: dict
    network_config: dict
    encoding_config: dict
    input_dim: int
    action_size: int
    games: int
    seed: int
    device: str
    max_steps: int
    batch_size: int


def run_evaluation_worker(job: EvaluationWorkerJob) -> tuple[str, dict[str, float | int]]:
    device = _resolve_torch_device(job.device)
    game_config = LostCitiesConfig(**job.game_config)
    network_config = NetworkConfig.model_validate(job.network_config)
    encoding_config = EncodingConfig.model_validate(job.encoding_config)
    network = DeepCFRMLP.from_config(job.input_dim, job.action_size, network_config).to(device)
    network.load_state_dict(job.strategy_state_dict)
    network.eval()
    result = evaluate_strategy_network(
        network,
        game_config,
        games=job.games,
        seed=job.seed,
        opponent=job.opponent,
        device=device,
        max_steps=job.max_steps,
        encoding=encoding_config,
        batch_size=job.batch_size,
    )
    return job.opponent, result


@dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    advantage_samples: int
    strategy_samples: int
    advantage_loss: float
    strategy_loss: float
    traversal_nodes: int
    traversal_terminals: int
    traversal_depth_cutoffs: int
    traversal_node_limit_cutoffs: int
    traversal_max_depth_reached: int
    traversal_endpoint_depth_sum: int
    traversal_endpoints: int
    traversal_avg_endpoint_depth: float
    traversal_endpoint_depth_buckets: dict[str, int]
    runtime_metrics: dict[str, float | int]
    eval_metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, float | int]:
        data = {
            "iteration": self.iteration,
            "samples/advantage": self.advantage_samples,
            "samples/strategy": self.strategy_samples,
            "loss/advantage": self.advantage_loss,
            "loss/strategy": self.strategy_loss,
            "traversal/nodes": self.traversal_nodes,
            "traversal/terminals": self.traversal_terminals,
            "traversal/depth_cutoffs": self.traversal_depth_cutoffs,
            "traversal/node_limit_cutoffs": self.traversal_node_limit_cutoffs,
            "traversal/max_depth_reached": self.traversal_max_depth_reached,
            "traversal/endpoint_depth_sum": self.traversal_endpoint_depth_sum,
            "traversal/endpoints": self.traversal_endpoints,
            "traversal/avg_endpoint_depth": self.traversal_avg_endpoint_depth,
            **{
                f"traversal/endpoint_depth_bucket_{key}": value
                for key, value in self.traversal_endpoint_depth_buckets.items()
            },
        }
        data.update(self.runtime_metrics)
        data.update(self.eval_metrics)
        return data


def _format_iteration_summary(metrics: IterationMetrics, data: dict[str, float | int]) -> str:
    iteration_seconds = float(data.get("time/iteration_seconds", 0.0) or 0.0)
    parts = [
        f"[i={metrics.iteration}] Iteration complete",
        f"traversal_nodes={metrics.traversal_nodes}",
        f"nodes_per_second={_format_summary_value(data['time/nodes_per_second'])}",
        f"advantage_loss={_format_summary_value(metrics.advantage_loss)}",
        f"strategy_loss={_format_summary_value(metrics.strategy_loss)}",
        f"iteration_seconds={_format_summary_value(iteration_seconds)}",
        f"iters_per_hour={3600.0 / iteration_seconds:.1f}"
        if iteration_seconds
        else "iters_per_hour=n/a",
    ]
    if metrics.eval_metrics:
        eval_seconds = float(data.get("time/evaluation_seconds", 0.0) or 0.0)
        fraction = eval_seconds / iteration_seconds if iteration_seconds > 0.0 else 0.0
        parts.append(f"eval_seconds={_format_summary_value(eval_seconds)}({fraction * 100:.0f}%)")
    for key in sorted(metrics.eval_metrics):
        if key.endswith("/win_rate0") or key.endswith("/avg_score_diff0"):
            parts.append(f"{key}={_format_summary_value(metrics.eval_metrics[key])}")
    return " ".join(parts)


def _format_summary_value(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _next_eval_iteration(current_iteration: int, eval_every: int) -> int | None:
    if eval_every <= 0:
        return None
    return ((current_iteration // eval_every) + 1) * eval_every


def eval_skipped_warning(
    current_iteration: int,
    max_iterations: int | None,
    eval_every: int,
) -> str | None:
    if eval_every <= 0 or max_iterations is None:
        return None
    next_eval = _next_eval_iteration(current_iteration, eval_every)
    if next_eval is None or next_eval <= max_iterations:
        return None
    return (
        f"WARNING evaluation will not run: eval_every={eval_every} "
        f"but max_iterations={max_iterations} (current iteration={current_iteration}). "
        f"Next scheduled eval at iteration {next_eval}."
    )


class DeepCFRTrainer:
    def __init__(
        self,
        config: DeepCFRConfig | None = None,
        game_config: LostCitiesConfig | None = None,
        *,
        run_dir: str | Path | None = None,
        device: str = "cpu",
        tracker: RunTracker | None = None,
        extra_trackers: list[RunTracker] | None = None,
    ) -> None:
        self.config = config or DeepCFRConfig()
        self.game_config = game_config or self.config.rules.to_lost_cities_config(
            seed=self.config.run.seed
        )
        self.device = _resolve_torch_device(device)

        probe = GameState.new_game(self.game_config, seed=self.config.run.seed)
        self.input_dim = input_dim(probe, self.config.encoding)
        self.action_size = 2 * probe.config.hand_size + 1 + probe.config.n_colors

        torch.manual_seed(self.config.run.seed)
        self.advantage_networks = [
            DeepCFRMLP.from_config(self.input_dim, self.action_size, self.config.network).to(
                self.device
            )
            for _ in range(2)
        ]
        self.strategy_network = DeepCFRMLP.from_config(
            self.input_dim, self.action_size, self.config.network
        ).to(self.device)
        self.advantage_optimizers = [
            torch.optim.AdamW(
                network.parameters(),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay,
            )
            for network in self.advantage_networks
        ]
        self.strategy_optimizer = torch.optim.AdamW(
            self.strategy_network.parameters(),
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
        )
        self.advantage_memories = [
            ReservoirMemory(self.config.memory.advantage_capacity) for _ in range(2)
        ]
        self.strategy_memory = ReservoirMemory(self.config.memory.strategy_capacity)
        self.rng = np.random.default_rng(self.config.run.seed + 101)
        self.iteration = 0
        self.run_dir = Path(run_dir) if run_dir is not None else Path("runs/tmp/default")
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.progress_path = self.run_dir / "runtime_progress.json"
        self.log_path = self.run_dir / "train.log"
        if tracker is not None:
            self.tracker = tracker
        else:
            trackers: list[RunTracker] = [
                FileRunTracker(
                    log_path=self.log_path,
                    metrics_path=self.metrics_path,
                    progress_path=self.progress_path,
                ),
                ConsoleRunTracker(),
            ]
            if extra_trackers:
                trackers.extend(extra_trackers)
            self.tracker = CompositeRunTracker(trackers)
        self.self_play_league_snapshots: list[list[dict]] = []
        self._runtime_metrics: dict[str, float | int] = {}
        self._inference_server: InferenceServerController | None = None
        self._last_inference_weight_sync_iteration: int | None = None

    def checkpoint_payload(self, metrics: IterationMetrics | None = None) -> dict:
        return {
            "config": self.config.to_dict(),
            "game_config": self.game_config.to_snapshot(),
            "resume_semantics": "networks_optimizers_iteration_only",
            "iteration": self.iteration,
            "input_dim": self.input_dim,
            "action_size": self.action_size,
            "advantage_networks": [network.state_dict() for network in self.advantage_networks],
            "strategy_network": self.strategy_network.state_dict(),
            "self_play_league_snapshots": self.self_play_league_snapshots,
            "advantage_optimizers": [
                optimizer.state_dict() for optimizer in self.advantage_optimizers
            ],
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "metrics": None if metrics is None else metrics.to_dict(),
        }

    def save_checkpoint(self, path: str | Path, metrics: IterationMetrics | None = None) -> Path:
        return save_checkpoint(path, self.checkpoint_payload(metrics))

    def load_checkpoint(self, path: str | Path) -> None:
        if self.config.checkpoint.exact_resume:
            # TODO: Implement exact resume by checkpointing reservoir memories, RNG state,
            # and any worker/traversal sampling state needed for deterministic continuation.
            raise NotImplementedError("checkpoint.exact_resume is not implemented yet")
        payload = load_checkpoint(path, device=self.device)
        self.iteration = int(payload.get("iteration", 0))
        self.tracker.log_event(
            f"Resuming from {path} with resume_semantics="
            f"{payload.get('resume_semantics', 'networks_optimizers_iteration_only')}; "
            "reservoir memories and RNG state are not restored"
        )
        for network, state_dict in zip(
            self.advantage_networks, payload["advantage_networks"], strict=True
        ):
            network.load_state_dict(state_dict)
        self.strategy_network.load_state_dict(payload["strategy_network"])
        for optimizer, state_dict in zip(
            self.advantage_optimizers, payload.get("advantage_optimizers", []), strict=False
        ):
            optimizer.load_state_dict(state_dict)
        if "strategy_optimizer" in payload:
            self.strategy_optimizer.load_state_dict(payload["strategy_optimizer"])
        self.self_play_league_snapshots = payload.get("self_play_league_snapshots", [])

    def _advantage_memory_size(self) -> int:
        return sum(len(memory) for memory in self.advantage_memories)

    def _add_advantage_samples(self, samples: list[TrainingSample]) -> None:
        for sample in samples:
            self.advantage_memories[sample.player].add(sample, self.rng)

    def run_iteration(self, iteration: int) -> IterationMetrics:
        self.iteration = iteration
        self._runtime_metrics = {}
        if self.config.traversal.inference_backend == "server":
            self._ensure_inference_server()
            self._maybe_sync_inference_server(iteration)
        traversal_started = time.perf_counter()
        if (
            self.config.traversal.resolved_num_workers() > 1
            or self.config.traversal.inference_backend == "server"
        ):
            total_stats = self._run_traversals_parallel(iteration)
        else:
            total_stats = self._run_traversals_single_process(iteration)
        self._runtime_metrics["time/traversal_seconds"] = time.perf_counter() - traversal_started

        advantage_started = time.perf_counter()
        advantage_loss = self._train_advantage_networks()
        self._runtime_metrics["time/advantage_train_seconds"] = (
            time.perf_counter() - advantage_started
        )

        strategy_started = time.perf_counter()
        strategy_loss = self._train_strategy_network()
        self._runtime_metrics["time/strategy_train_seconds"] = (
            time.perf_counter() - strategy_started
        )

        eval_started = time.perf_counter()
        eval_metrics = self._evaluate(iteration)
        if eval_metrics:
            self._runtime_metrics["time/evaluation_seconds"] = time.perf_counter() - eval_started
        self._runtime_metrics["memory/advantage"] = self._advantage_memory_size()
        for player, memory in enumerate(self.advantage_memories):
            self._runtime_metrics[f"memory/advantage_player_{player}"] = len(memory)
        self._runtime_metrics["memory/strategy"] = len(self.strategy_memory)
        for key, value in total_stats.to_dict().items():
            if key.startswith("traversal_regret_"):
                self._runtime_metrics["traversal/" + key[len("traversal_") :]] = value
            elif key == "traversal_sampled_actions":
                self._runtime_metrics["traversal/sampled_actions"] = value
        return IterationMetrics(
            iteration=iteration,
            advantage_samples=self._advantage_memory_size(),
            strategy_samples=len(self.strategy_memory),
            advantage_loss=advantage_loss,
            strategy_loss=strategy_loss,
            traversal_nodes=total_stats.nodes,
            traversal_terminals=total_stats.terminals,
            traversal_depth_cutoffs=total_stats.depth_cutoffs,
            traversal_node_limit_cutoffs=total_stats.node_limit_cutoffs,
            traversal_max_depth_reached=total_stats.max_depth_reached,
            traversal_endpoint_depth_sum=total_stats.endpoint_depth_sum,
            traversal_endpoints=total_stats.endpoints,
            traversal_avg_endpoint_depth=total_stats.avg_endpoint_depth,
            traversal_endpoint_depth_buckets=dict(total_stats.endpoint_depth_buckets),
            runtime_metrics=dict(self._runtime_metrics),
            eval_metrics=eval_metrics,
        )

    def _run_traversals_single_process(self, iteration: int) -> TraversalStats:
        total_stats = TraversalStats()
        for network in self.advantage_networks:
            network.eval()
        league_networks = self._materialize_league_networks()
        progress_every = int(self.config.traversal.progress_every_traversals)
        completed = 0
        progress_started = time.perf_counter()
        traversals_per_player = self.config.traversal.traversals_per_player
        for player in range(2):
            seeds = [
                self.config.run.seed + iteration * 10_000 + traversal_index * 10 + player
                for traversal_index in range(traversals_per_player)
            ]
            stats, advantage_samples, strategy_samples = run_cython_traversal_batch(
                self.advantage_networks,
                self.game_config,
                seeds,
                player,
                iteration,
                device=self.device,
                strategy_network=(
                    self.strategy_network
                    if self.config.traversal.opponent_policy == "average_strategy"
                    else None
                ),
                action_size=self.action_size,
                encoding=self.config.encoding,
                epsilon=self.config.traversal.regret_matching_epsilon,
                strategy_sample_interval=self.config.traversal.strategy_sample_interval,
                store_strategy_on_traverser_nodes=(
                    self.config.traversal.store_strategy_on_traverser_nodes
                ),
                store_strategy_on_opponent_nodes=(
                    self.config.traversal.store_strategy_on_opponent_nodes
                ),
                max_depth=self.config.traversal.max_depth,
                max_nodes=self.config.traversal.max_nodes_per_traversal,
                sampling_mode=self.config.traversal.sampling_mode,
                outcome_sampling_epsilon=self.config.traversal.outcome_sampling_epsilon,
                outcome_sampling_value_clip=self.config.traversal.outcome_sampling_value_clip,
                outcome_unsampled_regret=self.config.traversal.outcome_unsampled_regret,
                cutoff_value_mode=self.config.traversal.cutoff_value_mode,
                cutoff_rollouts=self.config.traversal.cutoff_rollouts,
                cutoff_rollout_policy=self.config.traversal.cutoff_rollout_policy,
                cutoff_rollout_max_steps=self.config.traversal.cutoff_rollout_max_steps,
                opponent_policy=self.config.traversal.opponent_policy,
                all_negative_fallback=self.config.regret_matching.all_negative_fallback,
                league_advantage_networks=league_networks,
                self_play_anchor_probability=self.config.self_play.anchor_probability,
                self_play_current_weight=self.config.self_play.current_weight,
                self_play_recent_weight=self.config.self_play.recent_weight,
                self_play_older_weight=self.config.self_play.older_weight,
                self_play_anchor_weight=self.config.self_play.anchor_weight,
                self_play_recent_window=self.config.self_play.recent_window,
                endpoint_depth_bucket_width=self.config.traversal.endpoint_depth_bucket_width,
                endpoint_depth_bucket_max=self.config.traversal.endpoint_depth_bucket_max,
                seed=self.config.run.seed + iteration * 1_000_003 + player,
            )
            total_stats.accumulate(stats)
            memory_add_started = time.perf_counter()
            self._add_advantage_samples(advantage_samples)
            self.strategy_memory.add_many(strategy_samples, self.rng)
            self._runtime_metrics["time/memory_add_seconds"] = (
                float(self._runtime_metrics.get("time/memory_add_seconds", 0.0))
                + time.perf_counter()
                - memory_add_started
            )
            completed += len(seeds)
            if progress_every > 0 and completed >= progress_every:
                elapsed = time.perf_counter() - progress_started
                self.tracker.log_event(
                    f"[i={iteration}] Traversal progress completed={completed} "
                    f"elapsed_seconds={elapsed:.2f} total_nodes={total_stats.nodes} "
                    f"nodes_per_second={total_stats.nodes / max(elapsed, 1.0e-12):.1f}"
                )
        return total_stats

    def _run_traversals_parallel(self, iteration: int) -> TraversalStats:
        batches = self._worker_batches(iteration)
        total_stats = TraversalStats()
        if not batches:
            return total_stats
        requested_workers = self.config.traversal.resolved_num_workers()
        max_workers = self.config.traversal.resolved_num_workers(len(batches))
        self.tracker.log_event(
            f"[i={iteration}] Traversal multiprocessing enabled "
            f"requested_workers={requested_workers} effective_workers={max_workers} "
            f"batches={len(batches)} chunk_size={self.config.traversal.worker_chunk_size}"
        )
        if max_workers < requested_workers:
            self.tracker.log_event(
                f"[i={iteration}] Traversal worker count capped "
                f"requested_workers={requested_workers} effective_workers={max_workers} "
                f"available_batches={len(batches)}"
            )
        progress_every = int(self.config.traversal.progress_every_traversals)
        next_progress_at = progress_every if progress_every > 0 else None
        progress_nodes = 0
        progress_traversals = 0
        progress_started = time.perf_counter()
        executor_kwargs: dict[str, object] = {
            "max_workers": max_workers,
            "mp_context": mp.get_context("spawn"),
        }
        if self.config.traversal.inference_backend == "server":
            if self._inference_server is None:
                raise RuntimeError("inference server is not initialized")
            self._record_inference_batch_stats(self._inference_server.drain_batch_stats())
            executor_kwargs["initializer"] = initialize_traversal_worker
            executor_kwargs["initargs"] = (self._inference_server.handles,)
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            total_batches = len(batches)
            in_flight_limit = min(total_batches, max(1, max_workers * 2))
            batch_iter = iter(batches)
            futures = {
                executor.submit(run_traversal_worker_batch, batch)
                for _, batch in zip(range(in_flight_limit), batch_iter, strict=False)
            }
            completed_batches = 0
            while futures:
                done, futures = wait(futures, timeout=5.0, return_when=FIRST_COMPLETED)
                if not done:
                    if (
                        self.config.traversal.inference_backend == "server"
                        and self._inference_server is not None
                        and not self._inference_server.is_alive
                    ):
                        raise RuntimeError("inference server process exited during traversal")
                    continue
                for future in done:
                    result = future.result()
                    completed_batches += 1
                    total_stats.accumulate(result.stats)
                    memory_add_started = time.perf_counter()
                    self._add_advantage_samples(result.advantage_samples)
                    self.strategy_memory.add_many(result.strategy_samples, self.rng)
                    self._runtime_metrics["time/memory_add_seconds"] = (
                        float(self._runtime_metrics.get("time/memory_add_seconds", 0.0))
                        + time.perf_counter()
                        - memory_add_started
                    )
                    progress_nodes += result.stats.nodes
                    progress_traversals += result.traversals
                    if next_progress_at is not None and progress_traversals >= next_progress_at:
                        elapsed = time.perf_counter() - progress_started
                        self.tracker.log_event(
                            f"[i={iteration}] Traversal multiprocessing progress "
                            f"completed_batches={completed_batches}/{total_batches} "
                            f"completed_traversals={progress_traversals} "
                            f"elapsed_seconds={elapsed:.2f} "
                            f"total_nodes={progress_nodes} "
                            f"nodes_per_second={progress_nodes / max(elapsed, 1.0e-12):.1f}"
                        )
                        while (
                            next_progress_at is not None and next_progress_at <= progress_traversals
                        ):
                            next_progress_at += progress_every
                    next_batch = next(batch_iter, None)
                    if next_batch is not None:
                        futures.add(executor.submit(run_traversal_worker_batch, next_batch))
        if (
            self.config.traversal.inference_backend == "server"
            and self._inference_server is not None
        ):
            self._record_inference_batch_stats(self._inference_server.drain_batch_stats())
        return total_stats

    def _worker_batches(self, iteration: int) -> list[TraversalWorkerBatch]:
        batches: list[TraversalWorkerBatch] = []
        if self.config.traversal.inference_backend == "server":
            network_payloads: list[dict] = []
            league_payloads = [[{}, {}] for _snapshot in self.self_play_league_snapshots]
            strategy_payload: dict | None = (
                {} if self.config.traversal.opponent_policy == "average_strategy" else None
            )
        else:
            network_payloads = [
                {name: value.detach().cpu() for name, value in network.state_dict().items()}
                for network in self.advantage_networks
            ]
            league_payloads = self._league_payloads()
            strategy_payload = None
            if self.config.traversal.opponent_policy == "average_strategy":
                strategy_payload = {
                    name: value.detach().cpu()
                    for name, value in self.strategy_network.state_dict().items()
                }
        chunk_size = self.config.traversal.worker_chunk_size
        batch_index = 0
        for player in range(2):
            seeds = [
                self.config.run.seed + iteration * 10_000 + index * 10 + player
                for index in range(self.config.traversal.traversals_per_player)
            ]
            for start in range(0, len(seeds), chunk_size):
                chunk = seeds[start : start + chunk_size]
                batches.append(
                    TraversalWorkerBatch(
                        player=player,
                        iteration=iteration,
                        seeds=chunk,
                        config=self.config.to_dict(),
                        game_config=self.game_config.to_snapshot(),
                        input_dim=self.input_dim,
                        action_size=self.action_size,
                        advantage_networks=network_payloads,
                        league_advantage_networks=league_payloads,
                        worker_seed=self.config.run.seed + iteration * 1_000_003 + batch_index,
                        strategy_network=strategy_payload,
                        inference_handles=None,
                    )
                )
                batch_index += 1
        return batches

    def _frozen_advantage_state_dicts(self) -> list[dict]:
        return [
            {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
            for network in self.advantage_networks
        ]

    def _maybe_record_self_play_snapshot(self, iteration: int) -> None:
        if self.config.traversal.opponent_policy != "self_play_league":
            return
        if self.config.self_play.max_snapshots <= 0:
            return
        if iteration % max(1, self.config.self_play.snapshot_every) != 0:
            return
        self.self_play_league_snapshots.append(self._frozen_advantage_state_dicts())
        overflow = len(self.self_play_league_snapshots) - self.config.self_play.max_snapshots
        if overflow > 0:
            del self.self_play_league_snapshots[:overflow]

    def _materialize_league_networks(self) -> list[list[nn.Module]]:
        league: list[list[nn.Module]] = []
        for snapshot in self.self_play_league_snapshots:
            networks = [
                DeepCFRMLP.from_config(self.input_dim, self.action_size, self.config.network).to(
                    self.device
                )
                for _ in range(2)
            ]
            for network, state_dict in zip(networks, snapshot, strict=True):
                network.load_state_dict(state_dict)
                network.eval()
            league.append(networks)
        return league

    def _league_payloads(self) -> list[list[dict]]:
        return self.self_play_league_snapshots

    def _record_inference_batch_stats(self, stats: list[BatchStatsMessage]) -> None:
        if not stats:
            return
        batch_sizes = [int(item.batch_size) for item in stats]
        group_counts = [int(item.group_count) for item in stats]
        request_count = sum(batch_sizes)
        batch_count = len(batch_sizes)
        self._runtime_metrics["inference_server/batches"] = (
            int(self._runtime_metrics.get("inference_server/batches", 0)) + batch_count
        )
        self._runtime_metrics["inference_server/requests"] = (
            int(self._runtime_metrics.get("inference_server/requests", 0)) + request_count
        )
        self._runtime_metrics["inference_server/groups"] = int(
            self._runtime_metrics.get("inference_server/groups", 0)
        ) + sum(group_counts)
        total_batches = int(self._runtime_metrics["inference_server/batches"])
        total_requests = int(self._runtime_metrics["inference_server/requests"])
        total_groups = int(self._runtime_metrics["inference_server/groups"])
        self._runtime_metrics["inference_server/avg_batch_size"] = total_requests / max(
            total_batches, 1
        )
        self._runtime_metrics["inference_server/avg_groups_per_batch"] = total_groups / max(
            total_batches, 1
        )
        self._runtime_metrics["inference_server/max_batch_size"] = max(
            int(self._runtime_metrics.get("inference_server/max_batch_size", 0)),
            max(batch_sizes),
        )
        current_min = self._runtime_metrics.get("inference_server/min_batch_size")
        self._runtime_metrics["inference_server/min_batch_size"] = (
            min(int(current_min), min(batch_sizes)) if current_min is not None else min(batch_sizes)
        )

    def _inference_num_slots(self) -> int:
        configured = self.config.inference_server.num_slots
        if configured is not None:
            return int(configured)
        workers = max(1, self.config.traversal.resolved_num_workers())
        return max(64, 4 * workers * int(self.config.traversal.worker_chunk_size))

    def _ensure_inference_server(self) -> None:
        if self.config.traversal.resolved_num_workers() < 1:
            raise ValueError(
                "traversal.inference_backend='server' requires traversal.num_workers >= 1"
            )
        if self._inference_server is not None:
            if not self._inference_server.is_alive:
                raise RuntimeError("inference server process is not alive")
            return
        self._inference_server = InferenceServerController(
            input_dim=self.input_dim,
            action_size=self.action_size,
            num_slots=self._inference_num_slots(),
            network_config=self.config.network,
            server_config=self.config.inference_server,
        )

    def _maybe_sync_inference_server(self, iteration: int) -> None:
        if self._inference_server is None:
            return
        interval = max(1, int(self.config.inference_server.weight_sync_every))
        if self._last_inference_weight_sync_iteration is not None and iteration % interval != 0:
            return
        advantage_payloads = [
            {name: value.detach().cpu() for name, value in network.state_dict().items()}
            for network in self.advantage_networks
        ]
        strategy_payload = {
            name: value.detach().cpu() for name, value in self.strategy_network.state_dict().items()
        }
        self._inference_server.push_weights(
            advantage_networks=advantage_payloads,
            strategy_network=strategy_payload,
            league_advantage_networks=self._league_payloads(),
        )
        self._last_inference_weight_sync_iteration = iteration

    def _shutdown_inference_server(self) -> None:
        if self._inference_server is None:
            return
        self._inference_server.shutdown()
        self._inference_server = None

    def train(self) -> list[IterationMetrics]:
        self._start_run_logging()
        metrics: list[IterationMetrics] = []
        start = self.iteration + 1
        stop = self._stop_iteration()
        run_started = time.perf_counter()
        iteration = start
        try:
            while iteration <= stop:
                started = time.perf_counter()
                item = self.run_iteration(iteration)
                metrics.append(item)
                self._maybe_record_self_play_snapshot(iteration)
                checkpoint_started = time.perf_counter()
                self._save_iteration_checkpoints(iteration, item)
                item.runtime_metrics["time/checkpoint_seconds"] = (
                    time.perf_counter() - checkpoint_started
                )
                elapsed = time.perf_counter() - started
                self._append_metrics(item, elapsed)
                if self._time_limit_reached(run_started):
                    break
                iteration += 1
        finally:
            self._shutdown_inference_server()
            self.tracker.close()
        return metrics

    def _stop_iteration(self) -> int:
        if self.config.run.max_iterations is not None:
            return max(self.iteration, int(self.config.run.max_iterations))
        return 2**31 - 1

    def _time_limit_reached(self, run_started: float) -> bool:
        if self.config.run.max_minutes is None:
            return False
        elapsed_minutes = (time.perf_counter() - run_started) / 60.0
        return elapsed_minutes >= self.config.run.max_minutes

    def _should_save_iteration(self, iteration: int) -> bool:
        interval = int(self.config.checkpoint.save_every)
        return interval > 0 and iteration % interval == 0

    def _save_iteration_checkpoints(self, iteration: int, item: IterationMetrics) -> None:
        checkpoint_dir = self.run_dir
        if self._should_save_iteration(iteration):
            self.save_checkpoint(checkpoint_dir / f"iteration_{iteration:05d}.pt", item)
        if self.config.checkpoint.save_latest:
            self.save_checkpoint(checkpoint_dir / "latest.pt", item)

    def _start_run_logging(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(
                json.dumps(self.config.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if self.iteration == 0 and self.metrics_path.exists():
            self.metrics_path.unlink()
        self.tracker.log_event(
            f"Deep CFR run start iteration={self.iteration} seed={self.config.run.seed}"
        )
        warning = eval_skipped_warning(
            self.iteration,
            self.config.run.max_iterations,
            self.config.evaluation.eval_every,
        )
        if warning is not None:
            self.tracker.log_event(warning)

    def _append_metrics(self, metrics: IterationMetrics, iteration_seconds: float) -> None:
        data = metrics.to_dict()
        data["time/iteration_seconds"] = iteration_seconds
        data["time/nodes_per_second"] = metrics.traversal_nodes / max(iteration_seconds, 1.0e-12)
        self.tracker.log_metrics(data, step=metrics.iteration)
        self.tracker.log_event(_format_iteration_summary(metrics, data))

    def _evaluate(self, iteration: int) -> dict[str, float | int]:
        if (
            self.config.evaluation.eval_every <= 0
            or iteration % self.config.evaluation.eval_every != 0
        ):
            return {}
        results: dict[str, float | int] = {}
        eval_device = self._evaluation_device()
        if self.config.evaluation.resolved_num_workers(len(self.config.evaluation.opponents)) > 1:
            return self._evaluate_parallel(iteration, eval_device)
        eval_network = self._evaluation_network(eval_device)
        for opponent in self.config.evaluation.opponents:
            result = evaluate_strategy_network(
                eval_network,
                self.game_config,
                games=self.config.evaluation.games,
                seed=self.config.run.seed + iteration * 1000,
                opponent=opponent,
                device=eval_device,
                max_steps=self.config.evaluation.max_steps,
                encoding=self.config.encoding,
                batch_size=self.config.evaluation.batch_size,
            )
            for key, value in result.items():
                results[f"eval/{opponent}/{key}"] = value
        return results

    def _evaluate_parallel(
        self,
        iteration: int,
        eval_device: torch.device,
    ) -> dict[str, float | int]:
        opponents = self.config.evaluation.opponents
        max_workers = self.config.evaluation.resolved_num_workers(len(opponents))
        self.tracker.log_event(
            f"Evaluation multiprocessing enabled iteration={iteration} "
            f"effective_workers={max_workers} opponents={len(opponents)} "
            f"batch_size={self.config.evaluation.batch_size} device={eval_device}"
        )
        state_dict = self._strategy_state_dict_cpu()
        jobs = [
            EvaluationWorkerJob(
                opponent=opponent,
                strategy_state_dict=state_dict,
                game_config=self.game_config.to_snapshot(),
                network_config=self.config.network.model_dump(mode="python"),
                encoding_config=self.config.encoding.model_dump(mode="python"),
                input_dim=self.input_dim,
                action_size=self.action_size,
                games=self.config.evaluation.games,
                seed=self.config.run.seed + iteration * 1000,
                device=str(eval_device),
                max_steps=self.config.evaluation.max_steps,
                batch_size=self.config.evaluation.batch_size,
            )
            for opponent in opponents
        ]
        results: dict[str, float | int] = {}
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(run_evaluation_worker, job) for job in jobs]
            for future in as_completed(futures):
                opponent, result = future.result()
                for key, value in result.items():
                    results[f"eval/{opponent}/{key}"] = value
        return results

    def _evaluation_device(self) -> torch.device:
        token = self.config.evaluation.device
        if token == "trainer":
            return self.device
        if token == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if token == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("evaluation.device=cuda requested but CUDA is unavailable")
        return torch.device(token)

    def _evaluation_network(self, device: torch.device) -> torch.nn.Module:
        if device == self.device:
            return self.strategy_network
        return copy.deepcopy(self.strategy_network).to(device).eval()

    def _strategy_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu() for key, value in self.strategy_network.state_dict().items()
        }

    def _train_advantage_networks(self) -> float:
        losses: list[float] = []
        for player, (network, memory) in enumerate(
            zip(self.advantage_networks, self.advantage_memories, strict=True)
        ):
            self._runtime_metrics[f"samples/advantage_player_{player}"] = len(memory)
            if len(memory) == 0:
                continue
            losses.append(self._train_advantage(player, network, self.advantage_optimizers[player]))
        return float(np.mean(losses)) if losses else 0.0

    def _train_strategy_network(self) -> float:
        self._runtime_metrics["samples/strategy"] = len(self.strategy_memory)
        if len(self.strategy_memory) == 0:
            return 0.0
        return self._train_strategy(self.strategy_network, self.strategy_optimizer)

    def _batch_tensors(
        self,
        batch: list[TrainingSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        x = torch.as_tensor(
            np.stack([sample.info_state for sample in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.as_tensor(
            np.stack([sample.target for sample in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        legal = torch.as_tensor(
            np.stack([sample.legal_mask for sample in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        iterations = torch.as_tensor(
            [sample.iteration for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        self._runtime_metrics["time/batch_tensor_seconds"] = (
            float(self._runtime_metrics.get("time/batch_tensor_seconds", 0.0))
            + time.perf_counter()
            - started
        )
        return x, y, legal, iterations

    def _iteration_weights(self, iterations: torch.Tensor, exponent: float) -> torch.Tensor:
        if exponent == 0.0:
            return torch.ones_like(iterations, dtype=torch.float32, device=self.device)
        current = max(1, int(self.iteration))
        relative = (iterations / float(current)).clamp(min=0.0, max=1.0)
        return relative.pow(float(exponent))

    def _train_advantage(
        self,
        player: int,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        losses: list[float] = []
        network.train()
        for _step in range(self.config.optimization.advantage_updates_per_iteration):
            sample_started = time.perf_counter()
            batch = self.advantage_memories[player].sample(
                self.config.optimization.advantage_batch_size,
                self.rng,
            )
            self._runtime_metrics[f"time/advantage_player_{player}_sample_seconds"] = (
                float(
                    self._runtime_metrics.get(f"time/advantage_player_{player}_sample_seconds", 0.0)
                )
                + time.perf_counter()
                - sample_started
            )
            x, y, legal, sample_iterations = self._batch_tensors(batch)
            pred = network(x)
            diff = (pred - y).masked_fill(~legal, 0.0)
            if self.config.training_weighting.mode == "none":
                loss = diff.square().sum() / legal.sum().clamp_min(1)
            elif self.config.training_weighting.mode == "lcfr":
                sample_weights = self._iteration_weights(
                    sample_iterations, self.config.training_weighting.lcfr_alpha
                )
                action_weights = sample_weights[:, None] * legal.float()
                loss = (diff.square() * action_weights).sum() / action_weights.sum().clamp_min(
                    1.0e-12
                )
            else:
                positive_weights = self._iteration_weights(
                    sample_iterations, self.config.training_weighting.dcfr_alpha
                )
                negative_weights = self._iteration_weights(
                    sample_iterations, self.config.training_weighting.dcfr_beta
                )
                target_weights = torch.where(
                    y >= 0.0, positive_weights[:, None], negative_weights[:, None]
                )
                action_weights = target_weights * legal.float()
                loss = (diff.square() * action_weights).sum() / action_weights.sum().clamp_min(
                    1.0e-12
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optimization.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.config.optimization.grad_clip
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def _train_strategy(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        last_loss = 0.0
        network.train()
        for _step in range(self.config.optimization.strategy_updates_per_iteration):
            sample_started = time.perf_counter()
            batch = self.strategy_memory.sample(
                self.config.optimization.strategy_batch_size, self.rng
            )
            self._runtime_metrics["time/strategy_sample_seconds"] = (
                float(self._runtime_metrics.get("time/strategy_sample_seconds", 0.0))
                + time.perf_counter()
                - sample_started
            )
            x, y, legal, sample_iterations = self._batch_tensors(batch)
            logits = network(x).masked_fill(~legal, torch.finfo(torch.float32).min)
            log_probs = nn.functional.log_softmax(logits, dim=-1).masked_fill(~legal, 0.0)
            per_sample_loss = -(y * log_probs).sum(dim=-1)
            if self.config.training_weighting.mode == "none":
                loss = per_sample_loss.mean()
            elif self.config.training_weighting.mode == "lcfr":
                sample_weights = self._iteration_weights(
                    sample_iterations, self.config.training_weighting.lcfr_alpha
                )
                loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(
                    1.0e-12
                )
            else:
                sample_weights = self._iteration_weights(
                    sample_iterations, self.config.training_weighting.dcfr_gamma
                )
                loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(
                    1.0e-12
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optimization.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.config.optimization.grad_clip
                )
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss
