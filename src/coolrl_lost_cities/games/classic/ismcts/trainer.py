from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import evaluate_strategy_network
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import IsMctsConfig
from .evaluate import evaluate_with_mcts
from .interleaved_self_play import play_self_play_iteration
from .network import AlphaZeroLogitsView, AlphaZeroNet
from .replay_buffer import ReplayBuffer, ReplaySample
from .workers import SelfPlayWorkerBatch, run_self_play_worker


@dataclass
class IterationMetrics:
    iteration: int
    samples_added: int
    replay_size: int
    policy_loss: float
    value_loss: float
    total_loss: float
    self_play_seconds: float
    train_seconds: float
    eval_metrics: dict[str, float | int]
    mcts_metrics: dict[str, float]

    def to_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = {
            "iteration": self.iteration,
            "samples/added": self.samples_added,
            "memory/replay": self.replay_size,
            "loss/policy": self.policy_loss,
            "loss/value": self.value_loss,
            "loss/total": self.total_loss,
            "time/self_play_seconds": self.self_play_seconds,
            "time/train_seconds": self.train_seconds,
        }
        data.update(self.mcts_metrics)
        data.update(self.eval_metrics)
        return data


class IsMctsTrainer:
    def __init__(
        self,
        config: IsMctsConfig,
        game_config: LostCitiesConfig,
        *,
        run_dir: str | Path,
        device: torch.device | str = "cpu",
        tracker: object | None = None,
    ) -> None:
        self.config = config
        self.game_config = game_config
        self.run_dir = Path(run_dir)
        self.device = self._resolve_device(device)
        self.tracker = tracker
        probe = GameState.new_game(game_config, seed=config.run.seed)
        self.input_dim = input_dim(probe, config.encoding)
        self.action_size = probe.action_size
        self.network = AlphaZeroNet.from_config(self.input_dim, self.action_size, config).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=max(float(config.optimization.weight_decay), 1.0e-4),
        )
        self.buffer = ReplayBuffer(config.training.replay_capacity, seed=config.run.seed)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.rng = random.Random(config.run.seed)

    def _resolve_device(self, device: torch.device | str) -> torch.device:
        token = str(device)
        if token == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(token)

    def train(self) -> list[IterationMetrics]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if self.metrics_path.exists():
            self.metrics_path.unlink()
        metrics: list[IterationMetrics] = []
        max_iterations = self.config.run.max_iterations or 1
        started = time.perf_counter()
        for iteration in range(1, max_iterations + 1):
            if self._time_limit_reached(started):
                break
            item = self.run_iteration(iteration)
            metrics.append(item)
            self._append_metrics(item)
            self._save_checkpoints(iteration, item)
            if self.tracker is not None:
                try:
                    self.tracker.log_metrics(item.to_dict(), step=iteration)
                except Exception as exc:  # pragma: no cover
                    print(f"tracker.log_metrics failed: {exc}", flush=True)
            print(json.dumps(item.to_dict(), sort_keys=True))
        return metrics

    def run_iteration(self, iteration: int) -> IterationMetrics:
        print(
            f"[iter {iteration}] self-play start (workers={self.config.training.num_workers})",
            flush=True,
        )
        self.network.eval()
        sp_started = time.perf_counter()
        if self.config.training.num_workers > 1:
            iteration_samples = self._run_self_play_parallel(iteration)
        else:
            iteration_samples = play_self_play_iteration(
                self.network,
                self.config.mcts,
                self.config.training,
                self.game_config,
                self.rng,
                device=self.device,
                encoding=self.config.encoding,
                temperature=self.config.temperature.training,
                max_steps=self.config.evaluation.max_steps,
            )
        self.buffer.add(iteration_samples)
        added = len(iteration_samples)
        self_play_seconds = time.perf_counter() - sp_started
        print(
            f"[iter {iteration}] self-play done in {self_play_seconds:.1f}s, {added} samples",
            flush=True,
        )
        mcts_metrics = self._compute_mcts_metrics(iteration_samples)

        train_started = time.perf_counter()
        losses = []
        for _ in range(self.config.training.gradient_steps_per_iter):
            batch = self.buffer.sample(self.config.training.batch_size)
            losses.append(self._train_batch(batch))
        train_seconds = time.perf_counter() - train_started
        loss_arr = np.asarray(losses, dtype=np.float64)
        print(f"[iter {iteration}] train done in {train_seconds:.1f}s, eval starting", flush=True)
        eval_started = time.perf_counter()
        eval_metrics = self._evaluate(iteration)
        print(
            f"[iter {iteration}] eval done in {time.perf_counter() - eval_started:.1f}s", flush=True
        )
        return IterationMetrics(
            iteration=iteration,
            samples_added=added,
            replay_size=len(self.buffer),
            policy_loss=float(loss_arr[:, 0].mean()) if len(loss_arr) else 0.0,
            value_loss=float(loss_arr[:, 1].mean()) if len(loss_arr) else 0.0,
            total_loss=float(loss_arr[:, 2].mean()) if len(loss_arr) else 0.0,
            self_play_seconds=self_play_seconds,
            train_seconds=train_seconds,
            eval_metrics=eval_metrics,
            mcts_metrics=mcts_metrics,
        )

    def _run_self_play_parallel(self, iteration: int) -> list[ReplaySample]:
        training_cfg = self.config.training
        num_workers = max(1, int(training_cfg.num_workers))
        total_games = int(training_cfg.games_per_iter)
        if total_games <= 0:
            return []
        # Split games across workers as evenly as possible.
        effective_workers = min(num_workers, total_games)
        base = total_games // effective_workers
        remainder = total_games % effective_workers
        per_worker = [base + (1 if i < remainder else 0) for i in range(effective_workers)]
        # Move network state dict to CPU for cross-process transfer.
        cpu_state = {
            name: tensor.detach().cpu() for name, tensor in self.network.state_dict().items()
        }
        config_dict = self.config.to_dict()
        game_snapshot = self.game_config.to_snapshot()
        max_steps = self.config.evaluation.max_steps
        temperature = self.config.temperature.training
        worker_device = str(self.config.training.worker_device)
        batches: list[SelfPlayWorkerBatch] = []
        for worker_index in range(effective_workers):
            seed = self.rng.randrange(2**31)
            batches.append(
                SelfPlayWorkerBatch(
                    worker_index=worker_index,
                    games_for_worker=per_worker[worker_index],
                    base_seed=seed,
                    config=config_dict,
                    game_config=game_snapshot,
                    network_state=cpu_state,
                    temperature=temperature,
                    max_steps=max_steps,
                    device=worker_device,
                )
            )
        samples: list[ReplaySample] = []
        ctx = mp.get_context("spawn")
        print(
            f"  spawning {effective_workers} workers for {total_games} games "
            f"(per_worker={per_worker})...",
            flush=True,
        )
        spawn_started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
            futures = [executor.submit(run_self_play_worker, batch) for batch in batches]
            print(
                f"  workers submitted in {time.perf_counter() - spawn_started:.1f}s, waiting for results...",
                flush=True,
            )
            results = []
            for future in futures:
                res = future.result()
                results.append(res)
                print(
                    f"  worker {res.worker_index} done ({len(res.samples)} samples, "
                    f"elapsed {time.perf_counter() - spawn_started:.1f}s)",
                    flush=True,
                )
        for result in sorted(results, key=lambda item: item.worker_index):
            samples.extend(result.samples)
        return samples

    def _train_batch(self, batch: list[ReplaySample]) -> tuple[float, float, float]:
        self.network.train()
        info = torch.as_tensor(
            np.stack([sample.info_state for sample in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        legal = torch.as_tensor(
            np.stack([sample.legal_mask for sample in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        pi = torch.as_tensor(
            np.stack([sample.pi_target for sample in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        value_target = torch.as_tensor(
            [sample.v_target for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        logits, value_pred = self.network(info, legal)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pi * log_probs).sum(dim=-1).mean()
        v_scale = float(self.network.value_scale)
        value_loss = nn.functional.mse_loss(value_pred / v_scale, value_target / v_scale)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.optimization.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.optimization.grad_clip,
            )
        self.optimizer.step()
        return float(policy_loss.item()), float(value_loss.item()), float(loss.item())

    def _evaluate(self, iteration: int) -> dict[str, float | int]:
        opponents = self.config.evaluation.opponents_for_iteration(iteration)
        if (
            not opponents
            and self.config.evaluation.eval_every > 0
            and self.config.run.max_iterations is not None
            and iteration >= self.config.run.max_iterations
        ):
            opponents = self.config.evaluation.opponents
        if not opponents:
            return {}
        self.network.eval()
        results: dict[str, float | int] = {}
        for opponent in opponents:
            print(f"  eval vs {opponent}...", flush=True)
            opp_started = time.perf_counter()
            if self.config.mcts.eval_with_mcts:
                eval_mcts_cfg = self.config.mcts.model_copy()
                if self.config.mcts.eval_n_simulations > 0:
                    eval_mcts_cfg = eval_mcts_cfg.model_copy(
                        update={"n_simulations": self.config.mcts.eval_n_simulations}
                    )
                result = evaluate_with_mcts(
                    self.network,
                    self.game_config,
                    eval_mcts_cfg,
                    games=self.config.evaluation.games,
                    seed=self.config.run.seed + iteration * 1000,
                    opponent=opponent,
                    device=self.device,
                    encoding=self.config.encoding,
                    max_steps=self.config.evaluation.max_steps,
                    config=self.config,
                    num_workers=self.config.training.num_workers,
                )
            else:
                logits_view = AlphaZeroLogitsView(self.network)
                result = evaluate_strategy_network(
                    logits_view,
                    self.game_config,
                    games=self.config.evaluation.games,
                    seed=self.config.run.seed + iteration * 1000,
                    opponent=opponent,
                    device=self.device,
                    encoding=self.config.encoding,
                    max_steps=self.config.evaluation.max_steps,
                    batch_size=self.config.evaluation.batch_size,
                )
            key = opponent.replace("-", "_")
            for metric_key, value in result.items():
                results[f"eval/{key}/{metric_key}"] = value
            par = result.get("play_action_rate", 0.0)
            sd = result.get("avg_score_diff0", 0.0)
            wr = result.get("win_rate0", 0.0)
            print(
                f"  eval vs {opponent} done in {time.perf_counter() - opp_started:.1f}s "
                f"PA={par:.2f} W={wr:.2f} S={sd:.1f}",
                flush=True,
            )
        return results

    def _compute_mcts_metrics(self, samples: list[ReplaySample]) -> dict[str, float]:
        if not samples:
            return {
                "mcts/avg_visit_entropy": 0.0,
                "mcts/value_prediction_error": 0.0,
                "mcts/policy_mcts_kl": 0.0,
            }
        entropies = [_entropy(sample.pi_target) for sample in samples]
        policy_kls = [
            _kl_divergence(sample.pi_target, sample.prior)
            for sample in samples
            if sample.prior is not None
        ]
        info = torch.as_tensor(
            np.stack([sample.info_state for sample in samples]),
            dtype=torch.float32,
            device=self.device,
        )
        legal = torch.as_tensor(
            np.stack([sample.legal_mask for sample in samples]),
            dtype=torch.bool,
            device=self.device,
        )
        target = torch.as_tensor(
            [sample.v_target for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            _logits, value_pred = self.network(info, legal)
            value_error = nn.functional.mse_loss(value_pred, target)
        return {
            "mcts/avg_visit_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mcts/value_prediction_error": float(value_error.item()),
            "mcts/policy_mcts_kl": float(np.mean(policy_kls)) if policy_kls else 0.0,
        }

    def _append_metrics(self, metrics: IterationMetrics) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics.to_dict(), sort_keys=True) + "\n")

    def _save_checkpoints(self, iteration: int, metrics: IterationMetrics) -> None:
        payload = {
            "config": self.config.to_dict(),
            "game_config": self.game_config.to_snapshot(),
            "iteration": iteration,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics.to_dict(),
        }
        if self.config.checkpoint.save_latest:
            torch.save(payload, self.run_dir / "latest.pt")
        if (
            self.config.checkpoint.save_every > 0
            and iteration % self.config.checkpoint.save_every == 0
        ):
            torch.save(payload, self.run_dir / f"iteration_{iteration:05d}.pt")

    def _time_limit_reached(self, started: float) -> bool:
        if self.config.run.max_minutes is None:
            return False
        return (time.perf_counter() - started) / 60.0 >= self.config.run.max_minutes


def _entropy(distribution: np.ndarray) -> float:
    probs = np.asarray(distribution, dtype=np.float64)
    probs = probs[probs > 0.0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _kl_divergence(target: np.ndarray, prior: np.ndarray | None) -> float:
    if prior is None:
        return 0.0
    pi = np.asarray(target, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    mask = pi > 0.0
    if not np.any(mask):
        return 0.0
    return float((pi[mask] * (np.log(pi[mask]) - np.log(np.clip(p[mask], 1.0e-12, 1.0)))).sum())
