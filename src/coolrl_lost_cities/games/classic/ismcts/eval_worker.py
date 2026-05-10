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
from .inference_server import InferenceClient
from .info_set import canonical_info_set_key
from .mcts import IsMctsSearcher
from .network import AlphaZeroNet

_INFERENCE_REQUEST_QUEUE: Any | None = None
_INFERENCE_RESPONSE_QUEUES: list[Any] | None = None


def init_eval_inference_queues(request_queue: Any, response_queues: list[Any]) -> None:
    global _INFERENCE_REQUEST_QUEUE, _INFERENCE_RESPONSE_QUEUES
    _INFERENCE_REQUEST_QUEUE = request_queue
    _INFERENCE_RESPONSE_QUEUES = response_queues


@dataclass(frozen=True)
class EvalWorkerBatch:
    worker_index: int
    config: dict[str, Any]
    game_config: dict[str, Any]
    network_state: dict[str, Any] | None
    mcts_config: dict[str, Any]
    opponent: str
    game_indices: list[int]
    seed: int
    device: str
    max_steps: int
    tasks: list[tuple[str, int]] | None = None
    use_inference_server: bool = False
    request_queue: Any | None = None
    response_queue: Any | None = None


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
    by_opponent: dict[str, dict[str, Any]] | None = None


def run_eval_worker(batch: EvalWorkerBatch) -> EvalWorkerResult:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    cfg: IsMctsConfig = config_from_dict(batch.config)
    game_config = LostCitiesConfig(**batch.game_config)
    probe = GameState.new_game(game_config, seed=batch.seed)
    in_dim = input_dim(probe, cfg.encoding)
    if batch.use_inference_server:
        device = torch.device("cpu")
        network = _NetworkShape(probe.action_size)
        request_queue = batch.request_queue or _INFERENCE_REQUEST_QUEUE
        response_queue = batch.response_queue
        if response_queue is None and _INFERENCE_RESPONSE_QUEUES is not None:
            response_queue = _INFERENCE_RESPONSE_QUEUES[batch.worker_index]
        if request_queue is None or response_queue is None:
            raise RuntimeError("inference server queues are required")
        inference_client = InferenceClient(
            batch.worker_index,
            request_queue,
            response_queue,
        )
    else:
        device = torch.device(batch.device)
        network = AlphaZeroNet.from_config(in_dim, probe.action_size, cfg).to(device)
        if batch.network_state is None:
            raise RuntimeError("network_state is required without inference server")
        network.load_state_dict(batch.network_state)
        network.eval()
        inference_client = None
    from .config import MctsConfig

    mcts_config = MctsConfig.model_validate(batch.mcts_config)

    rng = random.Random(batch.seed + batch.worker_index * 7919)
    score_diffs: list[float] = []
    wins0 = wins1 = draws = 0
    policy_turns = 0
    play_actions = 0
    timeouts = 0
    tasks = batch.tasks or [(batch.opponent, game_index) for game_index in batch.game_indices]
    by_opponent: dict[str, dict[str, Any]] = {}
    for opponent_name, game_index in tasks:
        bucket = by_opponent.setdefault(
            opponent_name,
            {
                "score_diffs": [],
                "wins0": 0,
                "wins1": 0,
                "draws": 0,
                "policy_turns": 0,
                "play_actions": 0,
                "timeouts": 0,
            },
        )
        game_policy_turns = 0
        game_play_actions = 0
        policy_player = game_index % 2
        opponents = [
            build_bot(opponent_name, seed=batch.seed + game_index),
            build_bot(opponent_name, seed=batch.seed + game_index + 1),
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
                visits = (
                    _search_with_inference_server(searcher, state, current, inference_client)
                    if inference_client is not None
                    else searcher.search(state, current)
                )
                if visits:
                    unified = max(visits, key=visits.get)
                else:
                    unified = state.unified_legal_actions()[0]
                if state.phase == "card":
                    policy_turns += 1
                    game_policy_turns += 1
                    if unified % 2 == 0:
                        play_actions += 1
                        game_play_actions += 1
                state.apply_unified_action(unified)
            else:
                action = opponents[current].act(state)
                state.apply_action(action)
            steps += 1
        if not terminated:
            timeouts += 1
            bucket["timeouts"] += 1
        diff = float(state.score_diff(policy_player))
        score_diffs.append(diff)
        bucket["score_diffs"].append(diff)
        if diff > 0:
            wins0 += 1
            bucket["wins0"] += 1
        elif diff < 0:
            wins1 += 1
            bucket["wins1"] += 1
        else:
            draws += 1
            bucket["draws"] += 1
        bucket["policy_turns"] += game_policy_turns
        bucket["play_actions"] += game_play_actions
    return EvalWorkerResult(
        worker_index=batch.worker_index,
        score_diffs=score_diffs,
        wins0=wins0,
        wins1=wins1,
        draws=draws,
        policy_turns=policy_turns,
        play_actions=play_actions,
        timeouts=timeouts,
        by_opponent=by_opponent,
    )


def _search_with_inference_server(
    searcher: IsMctsSearcher,
    state: GameState,
    traverser: int,
    inference_client: InferenceClient,
) -> dict[int, int]:
    from .interleaved_self_play import _evaluate_global_batch

    root_key = canonical_info_set_key(state, state.current_player)
    root = searcher.tree.get_or_create(
        root_key,
        player=state.current_player,
        terminal=state.terminal,
    )
    completed = 0
    sims = int(searcher.config.n_simulations)
    while completed < sims:
        quota = min(int(searcher.config.parallel_simulations), sims - completed)
        pending = searcher.prepare_simulation_batch(state, traverser, quota)
        if not pending:
            break
        jobs = [(_SearchProxy(searcher), item) for item in pending]
        _evaluate_global_batch(
            searcher.network,
            jobs,
            searcher.device,
            inference_client=inference_client,
        )
        completed += len(pending)
    return {action: root.visits.get(action, 0) for action in state.unified_legal_actions()}


@dataclass
class _SearchProxy:
    searcher: IsMctsSearcher


class _NetworkShape:
    def __init__(self, action_size: int) -> None:
        self.action_size = int(action_size)
