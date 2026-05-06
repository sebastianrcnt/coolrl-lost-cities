from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import evaluate_strategy_network
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.traverser import DeepCFRTraverser, TraversalStats
from coolrl_lost_cities.games.classic.deep_cfr.workers import (
    TraversalWorkerBatch,
    run_traversal_worker_batch,
)
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig


def _resolve_torch_device(device: str) -> torch.device:
    token = device.strip().lower()
    if token == "auto":
        token = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(token)


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
    eval_metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, float | int]:
        data = {
            "iteration": self.iteration,
            "advantage_samples": self.advantage_samples,
            "strategy_samples": self.strategy_samples,
            "advantage_loss": self.advantage_loss,
            "strategy_loss": self.strategy_loss,
            "traversal_nodes": self.traversal_nodes,
            "traversal_terminals": self.traversal_terminals,
            "traversal_depth_cutoffs": self.traversal_depth_cutoffs,
            "traversal_node_limit_cutoffs": self.traversal_node_limit_cutoffs,
            "traversal_max_depth_reached": self.traversal_max_depth_reached,
            "traversal_endpoint_depth_sum": self.traversal_endpoint_depth_sum,
            "traversal_endpoints": self.traversal_endpoints,
            "traversal_avg_endpoint_depth": self.traversal_avg_endpoint_depth,
            **{
                f"traversal_endpoint_depth_bucket_{key}": value
                for key, value in self.traversal_endpoint_depth_buckets.items()
            },
        }
        data.update(self.eval_metrics)
        return data


class DeepCFRTrainer:
    def __init__(
        self,
        config: DeepCFRConfig | None = None,
        game_config: LostCitiesConfig | None = None,
        *,
        device: str = "cpu",
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
            torch.optim.Adam(
                network.parameters(),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay,
            )
            for network in self.advantage_networks
        ]
        self.strategy_optimizer = torch.optim.Adam(
            self.strategy_network.parameters(),
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
        )
        self.advantage_memory = ReservoirMemory(self.config.memory.advantage_capacity)
        self.strategy_memory = ReservoirMemory(self.config.memory.strategy_capacity)
        self.rng = np.random.default_rng(self.config.run.seed + 101)
        self.iteration = 0
        self.run_dir = self.config.checkpoint_path
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.progress_path = self.run_dir / "runtime_progress.json"
        self.log_path = self.run_dir / "train.log"
        self.self_play_league_snapshots: list[list[dict]] = []

    def checkpoint_payload(self, metrics: IterationMetrics | None = None) -> dict:
        return {
            "config": self.config.to_dict(),
            "game_config": self.game_config.to_snapshot(),
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
        payload = load_checkpoint(path, device=self.device)
        self.iteration = int(payload.get("iteration", 0))
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

    def run_iteration(self, iteration: int) -> IterationMetrics:
        self.iteration = iteration
        if self.config.traversal.resolved_num_workers() > 1:
            total_stats = self._run_traversals_parallel(iteration)
        else:
            total_stats = self._run_traversals_single_process(iteration)

        advantage_loss = self._train_advantage_networks()
        strategy_loss = self._train_strategy_network()
        eval_metrics = self._evaluate(iteration)
        return IterationMetrics(
            iteration=iteration,
            advantage_samples=len(self.advantage_memory),
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
            eval_metrics=eval_metrics,
        )

    def _run_traversals_single_process(self, iteration: int) -> TraversalStats:
        total_stats = TraversalStats()
        traverser = DeepCFRTraverser(
            self.advantage_networks,
            self.advantage_memory,
            self.strategy_memory,
            device=self.device,
            action_size=self.action_size,
            epsilon=self.config.traversal.regret_matching_epsilon,
            strategy_sample_interval=self.config.traversal.strategy_sample_interval,
            store_strategy_on_traverser_nodes=self.config.traversal.store_strategy_on_traverser_nodes,
            store_strategy_on_opponent_nodes=self.config.traversal.store_strategy_on_opponent_nodes,
            max_depth=self.config.traversal.max_depth,
            max_nodes=self.config.traversal.resolved_max_nodes(),
            outcome_sampling_epsilon=self.config.traversal.outcome_sampling_epsilon,
            outcome_sampling_value_clip=self.config.traversal.outcome_sampling_value_clip,
            outcome_unsampled_regret=self.config.traversal.outcome_unsampled_regret,
            cutoff_value_mode=self.config.traversal.cutoff_value_mode,
            cutoff_rollouts=self.config.traversal.cutoff_rollouts,
            cutoff_rollout_policy=self.config.traversal.cutoff_rollout_policy,
            cutoff_rollout_max_steps=self.config.traversal.cutoff_rollout_max_steps,
            opponent_policy=self.config.traversal.opponent_policy,
            league_advantage_networks=self._materialize_league_networks(),
            self_play_anchor_probability=self.config.self_play.anchor_probability,
            self_play_current_weight=self.config.self_play.current_weight,
            self_play_recent_weight=self.config.self_play.recent_weight,
            self_play_older_weight=self.config.self_play.older_weight,
            self_play_anchor_weight=self.config.self_play.anchor_weight,
            self_play_recent_window=self.config.self_play.recent_window,
            endpoint_depth_bucket_width=self.config.traversal.endpoint_depth_bucket_width,
            endpoint_depth_bucket_max=self.config.traversal.endpoint_depth_bucket_max,
            encoding=self.config.encoding,
            rng=self.rng,
        )
        for network in self.advantage_networks:
            network.eval()
        for traversal_index in range(self.config.traversal.resolved_traversals_per_player()):
            for player in range(2):
                seed = self.config.run.seed + iteration * 10_000 + traversal_index * 10 + player
                state = GameState.new_game(self.game_config, seed=seed)
                _, stats = traverser.traverse(state, player, iteration)
                total_stats.accumulate(stats)
        return total_stats

    def _run_traversals_parallel(self, iteration: int) -> TraversalStats:
        batches = self._worker_batches(iteration)
        total_stats = TraversalStats()
        if not batches:
            return total_stats
        max_workers = self.config.traversal.resolved_num_workers(len(batches))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(run_traversal_worker_batch, batch) for batch in batches]
            for future in as_completed(futures):
                result = future.result()
                total_stats.accumulate(result.stats)
                self.advantage_memory.extend(result.advantage_samples, self.rng)
                self.strategy_memory.extend(result.strategy_samples, self.rng)
        return total_stats

    def _worker_batches(self, iteration: int) -> list[TraversalWorkerBatch]:
        batches: list[TraversalWorkerBatch] = []
        network_payloads = [
            {name: value.detach().cpu() for name, value in network.state_dict().items()}
            for network in self.advantage_networks
        ]
        chunk_size = self.config.traversal.resolved_worker_chunk_size()
        batch_index = 0
        for player in range(2):
            seeds = [
                self.config.run.seed + iteration * 10_000 + index * 10 + player
                for index in range(self.config.traversal.resolved_traversals_per_player())
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
                        league_advantage_networks=self._league_payloads(),
                        worker_seed=self.config.run.seed + iteration * 1_000_003 + batch_index,
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

    def train(self) -> list[IterationMetrics]:
        self._start_run_logging()
        metrics: list[IterationMetrics] = []
        start = self.iteration + 1
        stop = self._stop_iteration()
        run_started = time.perf_counter()
        iteration = start
        while iteration <= stop:
            started = time.perf_counter()
            item = self.run_iteration(iteration)
            elapsed = time.perf_counter() - started
            metrics.append(item)
            self._append_metrics(item, elapsed)
            self._maybe_record_self_play_snapshot(iteration)
            if self._should_save_iteration(iteration):
                self._save_iteration_checkpoints(iteration, item)
            if self._time_limit_reached(run_started):
                break
            iteration += 1
        return metrics

    def _stop_iteration(self) -> int:
        if self.config.run.max_iterations is not None:
            return max(self.iteration, int(self.config.run.max_iterations))
        if self.config.run.max_hours is not None:
            return 2**31 - 1
        return self.iteration + self.config.run.iterations

    def _time_limit_reached(self, run_started: float) -> bool:
        if self.config.run.max_hours is None:
            return False
        elapsed_hours = (time.perf_counter() - run_started) / 3600.0
        return elapsed_hours >= self.config.run.max_hours

    def _should_save_iteration(self, iteration: int) -> bool:
        if self.config.checkpoint.save_every_iteration:
            return True
        interval = int(self.config.checkpoint.save_iteration_interval)
        return interval > 0 and iteration % interval == 0

    def _save_iteration_checkpoints(self, iteration: int, item: IterationMetrics) -> None:
        checkpoint_dir = self.run_dir
        if not self.config.checkpoint.save_latest_only:
            self.save_checkpoint(checkpoint_dir / f"iteration_{iteration:05d}.pt", item)
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
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"Deep CFR run start iteration={self.iteration} seed={self.config.run.seed}\n"
            )

    def _append_metrics(self, metrics: IterationMetrics, iteration_seconds: float) -> None:
        data = metrics.to_dict()
        data["iteration_seconds"] = iteration_seconds
        data["nodes_per_second"] = metrics.traversal_nodes / max(iteration_seconds, 1.0e-12)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, sort_keys=True) + "\n")
        self.progress_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"iteration={metrics.iteration} nodes={metrics.traversal_nodes} adv_loss={metrics.advantage_loss:.6f} "
                f"strategy_loss={metrics.strategy_loss:.6f} seconds={iteration_seconds:.3f}\n"
            )

    def _evaluate(self, iteration: int) -> dict[str, float | int]:
        if (
            self.config.evaluation.eval_every <= 0
            or iteration % self.config.evaluation.eval_every != 0
        ):
            return {}
        results: dict[str, float | int] = {}
        for opponent in self.config.evaluation.opponents:
            result = evaluate_strategy_network(
                self.strategy_network,
                self.game_config,
                games=self.config.evaluation.games,
                seed=self.config.run.seed + iteration * 1000,
                opponent=opponent,
                device=self.device,
                max_steps=self.config.evaluation.max_steps,
                encoding=self.config.encoding,
            )
            for key, value in result.items():
                results[f"eval_{opponent}_{key}"] = value
        return results

    def _train_advantage_networks(self) -> float:
        losses: list[float] = []
        for player, network in enumerate(self.advantage_networks):
            samples = [sample for sample in self.advantage_memory.all() if sample.player == player]
            if not samples:
                continue
            losses.append(self._train_advantage(player, network, self.advantage_optimizers[player]))
        return float(np.mean(losses)) if losses else 0.0

    def _train_strategy_network(self) -> float:
        samples = self.strategy_memory.all()
        if not samples:
            return 0.0
        return self._train_strategy(self.strategy_network, self.strategy_optimizer, samples)

    def _batch_tensors(
        self,
        batch: list[TrainingSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return x, y, legal

    def _train_advantage(
        self,
        player: int,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        last_loss = 0.0
        network.train()
        for _step in range(self.config.optimization.resolved_advantage_train_steps()):
            x, y, legal = self._batch_tensors(
                self.advantage_memory.sample(
                    self.config.optimization.resolved_advantage_batch_size(),
                    self.rng,
                    player=player,
                )
            )
            pred = network(x)
            diff = (pred - y).masked_fill(~legal, 0.0)
            loss = diff.square().sum() / legal.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optimization.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.config.optimization.grad_clip
                )
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    def _train_strategy(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        samples: list[TrainingSample],
    ) -> float:
        last_loss = 0.0
        network.train()
        for _step in range(self.config.optimization.resolved_strategy_train_steps()):
            x, y, legal = self._batch_tensors(
                self.strategy_memory.sample(
                    self.config.optimization.resolved_strategy_batch_size(), self.rng
                )
            )
            logits = network(x).masked_fill(~legal, torch.finfo(torch.float32).min)
            log_probs = nn.functional.log_softmax(logits, dim=-1).masked_fill(~legal, 0.0)
            loss = -(y * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optimization.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.config.optimization.grad_clip
                )
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss
