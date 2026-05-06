from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.cfr_math import regret_matching
from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.traversal import root_action_values
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig


@dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    advantage_samples: int
    strategy_samples: int
    advantage_loss: float
    strategy_loss: float


class DeepCFRTrainer:
    def __init__(
        self,
        config: DeepCFRConfig | None = None,
        game_config: LostCitiesConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self.config = config or DeepCFRConfig()
        self.game_config = game_config or LostCitiesConfig(seed=self.config.seed)
        self.device = torch.device(device)

        probe = GameState.new_game(self.game_config, seed=self.config.seed)
        self.input_dim = input_dim(probe)
        self.action_size = 2 * probe.config.hand_size + 1 + probe.config.n_colors

        torch.manual_seed(self.config.seed)
        self.advantage_networks = [
            DeepCFRMLP(self.input_dim, self.action_size, self.config.hidden_size).to(self.device)
            for _ in range(2)
        ]
        self.strategy_network = DeepCFRMLP(
            self.input_dim, self.action_size, self.config.hidden_size
        ).to(self.device)
        self.advantage_optimizers = [
            torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
            for network in self.advantage_networks
        ]
        self.strategy_optimizer = torch.optim.Adam(
            self.strategy_network.parameters(), lr=self.config.learning_rate
        )
        self.advantage_memory = ReservoirMemory()
        self.strategy_memory = ReservoirMemory()

    def run_iteration(self, iteration: int) -> IterationMetrics:
        for traversal_index in range(self.config.traversals_per_iteration):
            for player in range(2):
                seed = self.config.seed + iteration * 10_000 + traversal_index * 10 + player
                state = GameState.new_game(self.game_config, seed=seed)
                info_state = encode_info_state(state, player)
                values, legal = root_action_values(
                    state,
                    player,
                    seed=seed,
                    rollouts_per_action=self.config.rollouts_per_action,
                    max_steps=self.config.max_rollout_steps,
                )
                legal_bool = legal.astype(bool)
                advantages = np.zeros_like(values, dtype=np.float32)
                if np.any(legal_bool):
                    baseline = float(np.mean(values[legal_bool]))
                    advantages[legal_bool] = values[legal_bool] - baseline
                policy = regret_matching(advantages, legal)

                self.advantage_memory.add(
                    TrainingSample(
                        info_state=info_state, target=advantages, iteration=iteration, player=player
                    )
                )
                self.strategy_memory.add(
                    TrainingSample(
                        info_state=info_state, target=policy, iteration=iteration, player=player
                    )
                )

        advantage_loss = self._train_advantage_networks()
        strategy_loss = self._train_strategy_network()
        return IterationMetrics(
            iteration=iteration,
            advantage_samples=len(self.advantage_memory),
            strategy_samples=len(self.strategy_memory),
            advantage_loss=advantage_loss,
            strategy_loss=strategy_loss,
        )

    def train(self) -> list[IterationMetrics]:
        return [self.run_iteration(iteration) for iteration in range(1, self.config.iterations + 1)]

    def _train_advantage_networks(self) -> float:
        losses: list[float] = []
        for player, network in enumerate(self.advantage_networks):
            samples = [sample for sample in self.advantage_memory.all() if sample.player == player]
            if not samples:
                continue
            losses.append(
                self._train_supervised(
                    network,
                    self.advantage_optimizers[player],
                    samples,
                    self.config.advantage_train_steps,
                )
            )
        return float(np.mean(losses)) if losses else 0.0

    def _train_strategy_network(self) -> float:
        samples = self.strategy_memory.all()
        if not samples:
            return 0.0
        return self._train_supervised(
            self.strategy_network,
            self.strategy_optimizer,
            samples,
            self.config.strategy_train_steps,
        )

    def _train_supervised(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        samples: list[TrainingSample],
        steps: int,
    ) -> float:
        last_loss = 0.0
        batch_size = min(self.config.batch_size, len(samples))
        for step in range(max(steps, 0)):
            offset = (step * batch_size) % len(samples)
            batch = samples[offset : offset + batch_size]
            if len(batch) < batch_size:
                batch = batch + samples[0 : batch_size - len(batch)]
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
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(network(x), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss
