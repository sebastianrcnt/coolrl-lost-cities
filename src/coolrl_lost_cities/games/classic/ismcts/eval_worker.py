"""Multi-process eval workers for ISMCTS — slice games across processes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from coolrl_lost_cities.games.classic.bots.registry import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import IsMctsConfig, config_from_dict
from .mcts import IsMctsSearcher
from .network import AlphaZeroNet


@dataclass(frozen=True)
class EvalWorkerBatch:
    worker_index: int
    config: dict[str, Any]
    game_config: dict[str, Any]
    network_state: dict[str, Any]
    mcts_config: dict[str, Any]
    opponent: str
    game_indices: list[int]
    seed: int
    device: str
    max_steps: int


@dataclass
class EvalWorkerResult:
    worker_index: int
    score_diffs: list[float]
    wins0: int
    wins1: int
    draws: int
    policy_turns: int
    play_actions: int
    timeouts: int


def run_eval_worker(batch: EvalWorkerBatch) -> EvalWorkerResult:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    cfg: IsMctsConfig = config_from_dict(batch.config)
    game_config = LostCitiesConfig(**batch.game_config)
    device = torch.device(batch.device)
    probe = GameState.new_game(game_config, seed=batch.seed)
    in_dim = input_dim(probe, cfg.encoding)
    network = AlphaZeroNet.from_config(in_dim, probe.action_size, cfg).to(device)
    network.load_state_dict(batch.network_state)
    network.eval()
    from .config import MctsConfig

    mcts_config = MctsConfig.model_validate(batch.mcts_config)

    rng = random.Random(batch.seed + batch.worker_index * 7919)
    score_diffs: list[float] = []
    wins0 = wins1 = draws = 0
    policy_turns = 0
    play_actions = 0
    timeouts = 0
    for game_index in batch.game_indices:
        policy_player = game_index % 2
        opponents = [
            build_bot(batch.opponent, seed=batch.seed + game_index),
            build_bot(batch.opponent, seed=batch.seed + game_index + 1),
        ]
        state = GameState.new_game(game_config, seed=batch.seed + game_index)
        steps = 0
        terminated = False
        while steps < batch.max_steps:
            if state.terminal:
                terminated = True
                break
            current = int(state.current_player)
            if current == policy_player:
                searcher = IsMctsSearcher(
                    network,
                    mcts_config,
                    device=device,
                    encoding=cfg.encoding,
                    rng=random.Random(rng.randrange(2**31)),
                )
                visits = searcher.search(state, current)
                if visits:
                    unified = max(visits, key=visits.get)
                else:
                    unified = state.unified_legal_actions()[0]
                if state.phase == "card":
                    policy_turns += 1
                    if unified % 2 == 0:
                        play_actions += 1
                state.apply_unified_action(unified)
            else:
                action = opponents[current].act(state)
                state.apply_action(action)
            steps += 1
        if not terminated:
            timeouts += 1
        diff = float(state.score_diff(policy_player))
        score_diffs.append(diff)
        if diff > 0:
            wins0 += 1
        elif diff < 0:
            wins1 += 1
        else:
            draws += 1
    return EvalWorkerResult(
        worker_index=batch.worker_index,
        score_diffs=score_diffs,
        wins0=wins0,
        wins1=wins1,
        draws=draws,
        policy_turns=policy_turns,
        play_actions=play_actions,
        timeouts=timeouts,
    )
