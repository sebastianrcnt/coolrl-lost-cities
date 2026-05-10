from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import IsMctsConfig
from .mcts import IsMctsSearcher
from .network import AlphaZeroNet
from .replay_buffer import ReplayBuffer, ReplaySample
from .self_play import play_self_play_game


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
    ) -> None:
        self.config = config
        self.game_config = game_config
        self.run_dir = Path(run_dir)
        self.device = self._resolve_device(device)
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
            print(json.dumps(item.to_dict(), sort_keys=True))
        return metrics

    def run_iteration(self, iteration: int) -> IterationMetrics:
        self.network.eval()
        sp_started = time.perf_counter()
        added = 0
        for _ in range(self.config.training.games_per_iter):
            samples = play_self_play_game(
                self.network,
                self.config.mcts,
                self.game_config,
                self.rng,
                device=self.device,
                encoding=self.config.encoding,
                temperature=self.config.temperature.training,
                max_steps=self.config.evaluation.max_steps,
            )
            self.buffer.add(samples)
            added += len(samples)
        self_play_seconds = time.perf_counter() - sp_started

        train_started = time.perf_counter()
        losses = []
        for _ in range(self.config.training.gradient_steps_per_iter):
            batch = self.buffer.sample(self.config.training.batch_size)
            losses.append(self._train_batch(batch))
        train_seconds = time.perf_counter() - train_started
        loss_arr = np.asarray(losses, dtype=np.float64)
        eval_metrics = self._evaluate(iteration)
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
        )

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
        value_loss = nn.functional.mse_loss(value_pred, value_target)
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
        if not opponents:
            return {}
        self.network.eval()
        results: dict[str, float | int] = {}
        for opponent in opponents:
            result = evaluate_policy(
                self.network,
                self.game_config,
                opponent=opponent,
                games=self.config.evaluation.games,
                seed=self.config.run.seed + iteration * 1000,
                device=self.device,
                encoding=self.config.encoding,
                max_steps=self.config.evaluation.max_steps,
                mcts_config=self.config.mcts,
            )
            key = opponent.replace("-", "_")
            for metric_key, value in result.items():
                results[f"eval/{key}/{metric_key}"] = value
        return results

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


def evaluate_policy(
    network: AlphaZeroNet,
    config: LostCitiesConfig,
    *,
    opponent: str,
    games: int,
    seed: int,
    device: torch.device | str,
    encoding=None,
    max_steps: int = 10_000,
    mcts_config=None,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    score_diffs: list[int] = []
    wins = losses = draws = 0
    policy_actions = play_actions = 0
    for game_index in range(games):
        policy_player = game_index % 2
        policies = [
            build_bot(opponent, seed=seed + game_index),
            build_bot(opponent, seed=seed + game_index),
        ]
        state = GameState.new_game(config, seed=seed + game_index)
        for _ in range(max_steps):
            if state.terminal:
                break
            current = int(state.current_player)
            if current == policy_player:
                searcher = IsMctsSearcher(
                    network,
                    mcts_config or IsMctsConfig().mcts,
                    device=device,
                    encoding=encoding,
                    rng=random.Random(rng.randrange(2**31)),
                )
                visits = searcher.search(state, current)
                unified = max(visits, key=visits.get)
                action = state.from_unified_action(unified)
            else:
                action = policies[current].act(state)
            if current == policy_player and state.phase == "card":
                policy_actions += 1
                if action % 2 == 0:
                    play_actions += 1
            state.apply_action(action)
        diff = state.score_diff(policy_player)
        score_diffs.append(diff)
        wins += int(diff > 0)
        losses += int(diff < 0)
        draws += int(diff == 0)
    return {
        "games": games,
        "wins0": wins,
        "wins1": losses,
        "draws": draws,
        "win_rate0": wins / max(1, games),
        "avg_score_diff0": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "play_action_rate": play_actions / max(1, policy_actions),
    }
