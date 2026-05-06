from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.deep_cfr.memory import ReservoirMemory, TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.traverser import DeepCFRTraverser, TraversalStats
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig


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
        self.rng = np.random.default_rng(self.config.seed + 101)

    def run_iteration(self, iteration: int) -> IterationMetrics:
        total_stats = TraversalStats()
        traverser = DeepCFRTraverser(
            self.advantage_networks,
            self.advantage_memory,
            self.strategy_memory,
            device=self.device,
            action_size=self.action_size,
            epsilon=self.config.regret_matching_epsilon,
            strategy_sample_interval=self.config.strategy_sample_interval,
            store_strategy_on_traverser_nodes=self.config.store_strategy_on_traverser_nodes,
            store_strategy_on_opponent_nodes=self.config.store_strategy_on_opponent_nodes,
            max_depth=self.config.max_traversal_depth,
            max_nodes=self.config.max_nodes_per_traversal,
            rng=self.rng,
        )
        for network in self.advantage_networks:
            network.eval()
        for traversal_index in range(self.config.traversals_per_iteration):
            for player in range(2):
                seed = self.config.seed + iteration * 10_000 + traversal_index * 10 + player
                state = GameState.new_game(self.game_config, seed=seed)
                _, stats = traverser.traverse(state, player, iteration)
                total_stats.accumulate(stats)

        advantage_loss = self._train_advantage_networks()
        strategy_loss = self._train_strategy_network()
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
                self._train_advantage(network, self.advantage_optimizers[player], samples)
            )
        return float(np.mean(losses)) if losses else 0.0

    def _train_strategy_network(self) -> float:
        samples = self.strategy_memory.all()
        if not samples:
            return 0.0
        return self._train_strategy(self.strategy_network, self.strategy_optimizer, samples)

    def _batch(self, samples: list[TrainingSample], step: int) -> list[TrainingSample]:
        batch_size = min(self.config.batch_size, len(samples))
        offset = (step * batch_size) % len(samples)
        batch = samples[offset : offset + batch_size]
        if len(batch) < batch_size:
            batch = batch + samples[0 : batch_size - len(batch)]
        return batch

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
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        samples: list[TrainingSample],
    ) -> float:
        last_loss = 0.0
        network.train()
        for step in range(max(self.config.advantage_train_steps, 0)):
            x, y, legal = self._batch_tensors(self._batch(samples, step))
            pred = network(x)
            diff = (pred - y).masked_fill(~legal, 0.0)
            loss = diff.square().sum() / legal.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
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
        for step in range(max(self.config.strategy_train_steps, 0)):
            x, y, legal = self._batch_tensors(self._batch(samples, step))
            logits = network(x).masked_fill(~legal, torch.finfo(torch.float32).min)
            log_probs = nn.functional.log_softmax(logits, dim=-1).masked_fill(~legal, 0.0)
            loss = -(y * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss
