from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.bots import SafeHeuristicBot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig, classic_config


@dataclass(frozen=True)
class ImitationMetrics:
    samples: int
    loss: float


def collect_safe_heuristic_samples(
    config: LostCitiesConfig | None = None,
    *,
    games: int = 4,
    seed: int = 1,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    game_config = config or classic_config(seed=seed)
    probe = GameState.new_game(game_config, seed=seed)
    action_size = 2 * probe.config.hand_size + 1 + probe.config.n_colors
    bot = SafeHeuristicBot()
    infos: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for game_index in range(games):
        state = GameState.new_game(game_config, seed=seed + game_index)
        for _ in range(max_steps):
            if state.terminal:
                break
            info = encode_info_state(state, state.current_player)
            legal = np.asarray(state.unified_legal_mask(), dtype=bool)
            action = bot.act(state)
            unified = state.to_unified_action(action)
            target = np.zeros(action_size, dtype=np.float32)
            target[unified] = 1.0
            infos.append(info)
            targets.append(target)
            masks.append(legal)
            state.apply_action(action)
    if not infos:
        raise RuntimeError("no imitation samples collected")
    return (
        np.stack(infos).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.stack(masks).astype(bool),
    )


def pretrain_strategy_network(
    strategy_network: nn.Module,
    config: LostCitiesConfig | None = None,
    *,
    games: int = 4,
    seed: int = 1,
    steps: int = 32,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    device: torch.device | str = "cpu",
) -> ImitationMetrics:
    x_np, y_np, legal_np = collect_safe_heuristic_samples(config, games=games, seed=seed)
    device = torch.device(device)
    strategy_network.to(device)
    strategy_network.train()
    optimizer = torch.optim.Adam(strategy_network.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed + 17)
    last_loss = 0.0
    for _ in range(max(0, steps)):
        indices = rng.choice(
            len(x_np), size=min(batch_size, len(x_np)), replace=len(x_np) < batch_size
        )
        x = torch.as_tensor(x_np[indices], dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np[indices], dtype=torch.float32, device=device)
        legal = torch.as_tensor(legal_np[indices], dtype=torch.bool, device=device)
        logits = strategy_network(x).masked_fill(~legal, torch.finfo(torch.float32).min)
        log_probs = nn.functional.log_softmax(logits, dim=-1).masked_fill(~legal, 0.0)
        loss = -(y * log_probs).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return ImitationMetrics(samples=len(x_np), loss=last_loss)


def new_pretrained_strategy_network(
    config: LostCitiesConfig | None = None,
    *,
    hidden_size: int = 64,
    games: int = 4,
    seed: int = 1,
    steps: int = 32,
) -> tuple[DeepCFRMLP, ImitationMetrics]:
    game_config = config or classic_config(seed=seed)
    probe = GameState.new_game(game_config, seed=seed)
    network = DeepCFRMLP(
        input_dim(probe), 2 * probe.config.hand_size + 1 + probe.config.n_colors, hidden_size
    )
    metrics = pretrain_strategy_network(network, game_config, games=games, seed=seed, steps=steps)
    return network, metrics
