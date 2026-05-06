from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.evaluation import evaluate_policy
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig
from coolrl_lost_cities.games.classic.policy import LostCitiesPolicy, PolicyInput


class StrategyNetPolicy(LostCitiesPolicy):
    def __init__(
        self,
        strategy_network: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.strategy_network = strategy_network
        self.device = torch.device(device)
        self.sample = sample
        self.rng = np.random.default_rng(seed)

    def act(self, obs_or_state: PolicyInput) -> int:
        if not isinstance(obs_or_state, GameState):
            legal = np.asarray(obs_or_state["legal_mask"], dtype=bool)
            legal_actions = np.flatnonzero(legal)
            if len(legal_actions) == 0:
                raise RuntimeError("no legal action available")
            return int(legal_actions[0])
        state = obs_or_state
        legal = np.asarray(state.unified_legal_mask(), dtype=bool)
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            raise RuntimeError("no legal action available")
        info = encode_info_state(state, state.current_player)
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_network(x).squeeze(0).detach().cpu().numpy()
        masked = np.where(legal, logits, -np.inf)
        if self.sample:
            stable = masked[legal_actions] - np.max(masked[legal_actions])
            probs = np.exp(stable)
            probs = probs / probs.sum()
            unified = int(self.rng.choice(legal_actions, p=probs))
        else:
            unified = int(np.argmax(masked))
        return state.from_unified_action(unified)


def evaluate_strategy_network(
    strategy_network: torch.nn.Module,
    config: LostCitiesConfig,
    *,
    games: int,
    seed: int,
    opponent: str = "random",
    device: torch.device | str = "cpu",
    max_steps: int = 10_000,
) -> dict[str, float | int]:
    strategy_network.eval()

    def make_strategy(seed_value: int | None = None) -> StrategyNetPolicy:
        return StrategyNetPolicy(strategy_network, device=device, seed=seed_value)

    def opponent_factory(seed_value=None):
        return build_bot(opponent, seed=seed_value)

    result = evaluate_policy(
        make_strategy,
        opponent_factory,
        config,
        games=games,
        seed=seed,
        max_steps=max_steps,
    )
    return result.to_dict()


def load_strategy_policy_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    sample: bool = False,
    seed: int | None = None,
) -> tuple[StrategyNetPolicy, LostCitiesConfig]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    game_config = LostCitiesConfig(**payload["game_config"])
    network = DeepCFRMLP.from_config(
        int(payload["input_dim"]),
        int(payload["action_size"]),
        cfg.network,
    ).to(device)
    network.load_state_dict(payload["strategy_network"])
    network.eval()
    return StrategyNetPolicy(network, device=device, sample=sample, seed=seed), game_config
