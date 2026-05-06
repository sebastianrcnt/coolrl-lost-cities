from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig, classic_config


@dataclass(frozen=True)
class PolicyGradientMetrics:
    episodes: int
    avg_reward: float
    loss: float


def fine_tune_strategy_policy_gradient(
    strategy_network: nn.Module,
    config: LostCitiesConfig | None = None,
    *,
    episodes: int = 2,
    seed: int = 1,
    opponent: str = "random",
    learning_rate: float = 1.0e-4,
    max_steps: int = 10_000,
    device: torch.device | str = "cpu",
) -> PolicyGradientMetrics:
    game_config = config or classic_config(seed=seed)
    device = torch.device(device)
    strategy_network.to(device)
    strategy_network.train()
    optimizer = torch.optim.Adam(strategy_network.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed + 31)
    rewards: list[float] = []
    losses: list[torch.Tensor] = []
    for episode in range(episodes):
        state = GameState.new_game(game_config, seed=seed + episode)
        opponent_policy = build_bot(opponent, seed=seed + episode + 1000)
        log_probs: list[torch.Tensor] = []
        for _ in range(max_steps):
            if state.terminal:
                break
            if state.current_player == 0:
                info = encode_info_state(state, 0)
                legal = torch.as_tensor(state.unified_legal_mask(), dtype=torch.bool, device=device)
                x = torch.as_tensor(info, dtype=torch.float32, device=device).unsqueeze(0)
                logits = (
                    strategy_network(x)
                    .squeeze(0)
                    .masked_fill(~legal, torch.finfo(torch.float32).min)
                )
                probs = torch.softmax(logits, dim=-1)
                action_index = int(rng.choice(len(probs), p=probs.detach().cpu().numpy()))
                log_probs.append(torch.log(probs[action_index].clamp_min(1.0e-12)))
                action = state.from_unified_action(action_index)
            else:
                action = opponent_policy.act(state)
            state.apply_action(action)
        reward = float(state.score_diff(0))
        rewards.append(reward)
        if log_probs:
            losses.append(-torch.stack(log_probs).sum() * reward)
    if losses:
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    else:
        loss_value = 0.0
    return PolicyGradientMetrics(
        episodes=episodes,
        avg_reward=float(np.mean(rewards)) if rewards else 0.0,
        loss=loss_value,
    )
