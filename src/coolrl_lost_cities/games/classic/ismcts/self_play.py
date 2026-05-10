from __future__ import annotations

import random

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import MctsConfig
from .mcts import IsMctsSearcher
from .network import AlphaZeroNet
from .replay_buffer import ReplaySample


def visit_distribution(
    visit_counts: dict[int, int],
    action_size: int,
    *,
    temperature: float,
) -> np.ndarray:
    pi = np.zeros(action_size, dtype=np.float32)
    if not visit_counts:
        return pi
    actions = np.asarray(list(visit_counts), dtype=np.int64)
    counts = np.asarray([visit_counts[int(action)] for action in actions], dtype=np.float64)
    if temperature <= 0.0:
        best = int(actions[int(np.argmax(counts))])
        pi[best] = 1.0
        return pi
    adjusted = np.power(np.maximum(counts, 1.0e-12), 1.0 / temperature)
    adjusted /= adjusted.sum()
    pi[actions] = adjusted.astype(np.float32)
    return pi


def select_from_distribution(pi: np.ndarray, rng: random.Random) -> int:
    total = float(pi.sum())
    if total <= 0.0:
        raise RuntimeError("empty action distribution")
    threshold = rng.random() * total
    cumsum = 0.0
    for action, prob in enumerate(pi):
        cumsum += float(prob)
        if cumsum >= threshold:
            return action
    return int(len(pi) - 1)


def play_self_play_game(
    network: AlphaZeroNet,
    mcts_config: MctsConfig,
    game_config: LostCitiesConfig,
    rng: random.Random,
    *,
    device: torch.device | str = "cpu",
    encoding=None,
    temperature: float = 1.0,
    max_steps: int = 10_000,
) -> list[ReplaySample]:
    state = GameState.new_game(game_config, seed=rng.randrange(2**31))
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]] = []
    steps = 0
    while not state.terminal and steps < max_steps:
        player = int(state.current_player)
        legal_mask = np.asarray(state.unified_legal_mask(), dtype=bool)
        info = encode_info_state(state, player, encoding)
        with torch.inference_mode():
            x = torch.as_tensor(info[None, :], dtype=torch.float32, device=device)
            mask = torch.as_tensor(legal_mask[None, :], dtype=torch.bool, device=device)
            prior = (
                network.policy_distribution(x, mask)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        searcher = IsMctsSearcher(
            network,
            mcts_config,
            device=device,
            encoding=encoding,
            rng=random.Random(rng.randrange(2**31)),
        )
        visits = searcher.search(state, player)
        pi = visit_distribution(visits, state.action_size, temperature=temperature)
        if pi.sum() <= 0:
            legal_actions = np.flatnonzero(legal_mask)
            pi[legal_actions] = 1.0 / len(legal_actions)
        pending.append((info.astype(np.float32), legal_mask, pi, player, prior))
        action = select_from_distribution(pi, rng)
        state.apply_unified_action(action)
        steps += 1

    final_diff0 = float(state.score_diff(0))
    samples: list[ReplaySample] = []
    for info, legal_mask, pi, player, prior in pending:
        value = final_diff0 if player == 0 else -final_diff0
        samples.append(
            ReplaySample(
                info_state=info,
                legal_mask=legal_mask.astype(bool),
                pi_target=pi.astype(np.float32),
                v_target=value,
                player=player,
                prior=prior,
            )
        )
    return samples
