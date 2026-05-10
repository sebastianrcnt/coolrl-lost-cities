from __future__ import annotations

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState
from coolrl_lost_cities.games.classic.policy import LostCitiesPolicy, PolicyInput

from .network import AlphaZeroNet


class AlphaZeroPolicy(LostCitiesPolicy):
    def __init__(
        self,
        network: AlphaZeroNet,
        *,
        device: torch.device | str = "cpu",
        encoding=None,
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.network = network
        self.device = torch.device(device)
        self.encoding = encoding
        self.sample = sample
        self.rng = np.random.default_rng(seed)

    def act(self, obs_or_state: PolicyInput) -> int:
        if not isinstance(obs_or_state, GameState):
            legal = np.asarray(obs_or_state["legal_mask"], dtype=bool)
            return int(np.flatnonzero(legal)[0])
        state = obs_or_state
        legal = np.asarray(state.unified_legal_mask(), dtype=bool)
        info = encode_info_state(state, state.current_player, self.encoding)
        with torch.inference_mode():
            x = torch.as_tensor(info[None, :], dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(legal[None, :], dtype=torch.bool, device=self.device)
            probs = self.network.policy_distribution(x, mask).squeeze(0).cpu().numpy()
        legal_actions = np.flatnonzero(legal)
        if self.sample:
            unified = int(self.rng.choice(legal_actions, p=probs[legal_actions]))
        else:
            unified = int(legal_actions[int(np.argmax(probs[legal_actions]))])
        return state.from_unified_action(unified)
