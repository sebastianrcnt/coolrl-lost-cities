from __future__ import annotations

import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from coolrl_lost_cities.games.classic.bots.registry import build_bot
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import IsMctsConfig, MctsConfig
from .eval_worker import EvalWorkerBatch, init_eval_inference_queues, run_eval_worker
from .mcts import IsMctsSearcher
from .network import AlphaZeroNet


def evaluate_with_mcts(
    network: AlphaZeroNet,
    game_config: LostCitiesConfig,
    mcts_config: MctsConfig,
    *,
    games: int,
    seed: int,
    opponent: str,
    device: torch.device | str = "cpu",
    encoding=None,
    max_steps: int = 10_000,
    config: IsMctsConfig | None = None,
    num_workers: int = 1,
) -> dict[str, float | int]:
    started = time.perf_counter()
    if num_workers > 1 and config is not None and games > 1:
        return _evaluate_parallel(
            network,
            game_config,
            mcts_config,
            config=config,
            games=games,
            seed=seed,
            opponent=opponent,
            num_workers=num_workers,
            max_steps=max_steps,
            started=started,
        )
    rng = random.Random(seed)
    score_diffs: list[float] = []
    wins0 = wins1 = draws = 0
    policy_turns = 0
    play_actions = 0
    timeouts = 0

    network.eval()
    for game_index in range(games):
        policy_player = game_index % 2
        opponents = [
            build_bot(opponent, seed=seed + game_index),
            build_bot(opponent, seed=seed + game_index + 1),
        ]
        state = GameState.new_game(game_config, seed=seed + game_index)
        steps = 0
        terminated = False
        while steps < max_steps:
            if state.terminal:
                terminated = True
                break
            current = int(state.current_player)
            if current == policy_player:
                searcher = IsMctsSearcher(
                    network,
                    mcts_config,
                    device=device,
                    encoding=encoding,
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

    n = len(score_diffs)
    avg_diff = sum(score_diffs) / n if n else 0.0
    return {
        "games": n,
        "win_rate0": wins0 / n if n else 0.0,
        "win_rate1": wins1 / n if n else 0.0,
        "wins0": wins0,
        "wins1": wins1,
        "draws": draws,
        "avg_score_diff0": avg_diff,
        "policy_turns": policy_turns,
        "play_action_rate": play_actions / policy_turns if policy_turns else 0.0,
        "max_step_timeouts": timeouts,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_parallel(
    network: AlphaZeroNet,
    game_config: LostCitiesConfig,
    mcts_config: MctsConfig,
    *,
    config: IsMctsConfig,
    games: int,
    seed: int,
    opponent: str,
    num_workers: int,
    max_steps: int,
    started: float,
) -> dict[str, float | int]:
    effective_workers = min(num_workers, games)
    base = games // effective_workers
    rem = games % effective_workers
    counts = [base + (1 if i < rem else 0) for i in range(effective_workers)]
    indices_per_worker: list[list[int]] = []
    cursor = 0
    for c in counts:
        indices_per_worker.append(list(range(cursor, cursor + c)))
        cursor += c
    cpu_state = {name: tensor.detach().cpu() for name, tensor in network.state_dict().items()}
    config_dict = config.to_dict()
    game_snapshot = game_config.to_snapshot()
    mcts_dict = mcts_config.model_dump(mode="json")
    worker_device = str(config.training.worker_device)
    batches = [
        EvalWorkerBatch(
            worker_index=i,
            config=config_dict,
            game_config=game_snapshot,
            network_state=cpu_state,
            mcts_config=mcts_dict,
            opponent=opponent,
            game_indices=indices_per_worker[i],
            seed=seed,
            device=worker_device,
            max_steps=max_steps,
        )
        for i in range(effective_workers)
    ]
    ctx = mp.get_context("spawn")
    score_diffs: list[float] = []
    wins0 = wins1 = draws = 0
    policy_turns = 0
    play_actions = 0
    timeouts = 0
    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
        for res in executor.map(run_eval_worker, batches):
            score_diffs.extend(res.score_diffs)
            wins0 += res.wins0
            wins1 += res.wins1
            draws += res.draws
            policy_turns += res.policy_turns
            play_actions += res.play_actions
            timeouts += res.timeouts
    n = len(score_diffs)
    return _evaluation_metrics(
        score_diffs=score_diffs,
        wins0=wins0,
        wins1=wins1,
        draws=draws,
        policy_turns=policy_turns,
        play_actions=play_actions,
        timeouts=timeouts,
        elapsed_seconds=time.perf_counter() - started,
        n=n,
    )


def evaluate_opponents_with_mcts_parallel(
    network: AlphaZeroNet,
    game_config: LostCitiesConfig,
    mcts_config: MctsConfig,
    *,
    config: IsMctsConfig,
    opponents: tuple[str, ...],
    games: int,
    seed: int,
    num_workers: int,
    max_steps: int,
    request_queue=None,
    response_queues=None,
) -> dict[str, dict[str, float | int]]:
    started = time.perf_counter()
    tasks = [(opponent, game_index) for opponent in opponents for game_index in range(games)]
    if not tasks:
        return {}
    effective_workers = min(max(1, int(num_workers)), len(tasks))
    tasks_per_worker = [tasks[i::effective_workers] for i in range(effective_workers)]
    use_inference_server = request_queue is not None and response_queues is not None
    cpu_state = (
        None
        if use_inference_server
        else {name: tensor.detach().cpu() for name, tensor in network.state_dict().items()}
    )
    config_dict = config.to_dict()
    game_snapshot = game_config.to_snapshot()
    mcts_dict = mcts_config.model_dump(mode="json")
    worker_device = str(config.training.worker_device)
    batches = [
        EvalWorkerBatch(
            worker_index=i,
            config=config_dict,
            game_config=game_snapshot,
            network_state=cpu_state,
            mcts_config=mcts_dict,
            opponent=tasks_per_worker[i][0][0] if tasks_per_worker[i] else "",
            game_indices=[],
            seed=seed,
            device=worker_device,
            max_steps=max_steps,
            tasks=tasks_per_worker[i],
            use_inference_server=use_inference_server,
            request_queue=None,
            response_queue=None,
        )
        for i in range(effective_workers)
    ]
    ctx = mp.get_context("spawn")
    aggregate: dict[str, dict[str, object]] = {
        opponent: {
            "score_diffs": [],
            "wins0": 0,
            "wins1": 0,
            "draws": 0,
            "policy_turns": 0,
            "play_actions": 0,
            "timeouts": 0,
        }
        for opponent in opponents
    }
    executor_kwargs = (
        {
            "initializer": init_eval_inference_queues,
            "initargs": (request_queue, response_queues),
        }
        if use_inference_server
        else {}
    )
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=ctx,
        **executor_kwargs,
    ) as executor:
        for result in executor.map(run_eval_worker, batches):
            for opponent, bucket in (result.by_opponent or {}).items():
                dest = aggregate[opponent]
                dest["score_diffs"].extend(bucket["score_diffs"])
                dest["wins0"] += int(bucket["wins0"])
                dest["wins1"] += int(bucket["wins1"])
                dest["draws"] += int(bucket["draws"])
                dest["policy_turns"] += int(bucket["policy_turns"])
                dest["play_actions"] += int(bucket["play_actions"])
                dest["timeouts"] += int(bucket["timeouts"])
    elapsed = time.perf_counter() - started
    return {
        opponent: _evaluation_metrics(
            score_diffs=list(bucket["score_diffs"]),
            wins0=int(bucket["wins0"]),
            wins1=int(bucket["wins1"]),
            draws=int(bucket["draws"]),
            policy_turns=int(bucket["policy_turns"]),
            play_actions=int(bucket["play_actions"]),
            timeouts=int(bucket["timeouts"]),
            elapsed_seconds=elapsed,
            n=len(bucket["score_diffs"]),
        )
        for opponent, bucket in aggregate.items()
    }


def _evaluation_metrics(
    *,
    score_diffs: list[float],
    wins0: int,
    wins1: int,
    draws: int,
    policy_turns: int,
    play_actions: int,
    timeouts: int,
    elapsed_seconds: float,
    n: int,
) -> dict[str, float | int]:
    avg_diff = sum(score_diffs) / n if n else 0.0
    return {
        "games": n,
        "win_rate0": wins0 / n if n else 0.0,
        "win_rate1": wins1 / n if n else 0.0,
        "wins0": wins0,
        "wins1": wins1,
        "draws": draws,
        "avg_score_diff0": avg_diff,
        "policy_turns": policy_turns,
        "play_action_rate": play_actions / policy_turns if policy_turns else 0.0,
        "max_step_timeouts": timeouts,
        "elapsed_seconds": elapsed_seconds,
    }
