"""Multi-process self-play workers for ISMCTS.

Mirrors the Deep CFR pattern: a ``ProcessPoolExecutor`` (spawn context)
runs N worker processes, each receiving the current network state dict and
a slice of the iteration's self-play games. Workers run network inference
on CPU by default (small policy/value MLP, GPU contention is the bottleneck
when sharing a single device across many workers).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import torch

from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import IsMctsConfig, config_from_dict
from .inference_server import InferenceClient
from .interleaved_self_play import play_self_play_iteration
from .network import AlphaZeroNet
from .replay_buffer import ReplaySample

_TORCH_THREADS_CONFIGURED = False
_INFERENCE_REQUEST_QUEUE: Any | None = None
_INFERENCE_RESPONSE_QUEUES: list[Any] | None = None


def init_inference_queues(request_queue: Any, response_queues: list[Any]) -> None:
    global _INFERENCE_REQUEST_QUEUE, _INFERENCE_RESPONSE_QUEUES
    _INFERENCE_REQUEST_QUEUE = request_queue
    _INFERENCE_RESPONSE_QUEUES = response_queues


def _configure_worker_torch_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    _TORCH_THREADS_CONFIGURED = True


@dataclass(frozen=True)
class SelfPlayWorkerBatch:
    worker_index: int
    games_for_worker: int
    base_seed: int
    config: dict[str, Any]
    game_config: dict[str, Any]
    network_state: dict[str, Any] | None
    temperature: float
    max_steps: int
    device: str
    use_inference_server: bool = False
    request_queue: Any | None = None
    response_queue: Any | None = None


@dataclass
class SelfPlayWorkerResult:
    worker_index: int
    samples: list[ReplaySample]


def run_self_play_worker(batch: SelfPlayWorkerBatch) -> SelfPlayWorkerResult:
    import time as _time

    _t0 = _time.perf_counter()
    print(f"  [worker {batch.worker_index}] starting ({batch.games_for_worker} games)", flush=True)
    _configure_worker_torch_threads()
    cfg: IsMctsConfig = config_from_dict(batch.config)
    game_config = LostCitiesConfig(**batch.game_config)
    probe = GameState.new_game(game_config, seed=batch.base_seed)
    in_dim = input_dim(probe, cfg.encoding)
    action_size = probe.action_size
    if batch.use_inference_server:
        device = torch.device("cpu")
        network = _NetworkShape(action_size)
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
        network = AlphaZeroNet.from_config(in_dim, action_size, cfg).to(device)
        if batch.network_state is None:
            raise RuntimeError("network_state is required without inference server")
        network.load_state_dict(batch.network_state)
        network.eval()
        inference_client = None
    print(
        f"  [worker {batch.worker_index}] init done in {_time.perf_counter() - _t0:.1f}s, self-play start",
        flush=True,
    )
    # Build a per-worker TrainingConfig with the worker's game count.
    worker_training = cfg.training.model_copy(
        update={"games_per_iter": int(batch.games_for_worker)}
    )
    rng = random.Random(batch.base_seed)
    _sp_t0 = _time.perf_counter()
    samples = play_self_play_iteration(
        network,
        cfg.mcts,
        worker_training,
        game_config,
        rng,
        device=device,
        encoding=cfg.encoding,
        temperature=batch.temperature,
        max_steps=batch.max_steps,
        inference_client=inference_client,
    )
    print(
        f"  [worker {batch.worker_index}] self-play done in {_time.perf_counter() - _sp_t0:.1f}s ({len(samples)} samples)",
        flush=True,
    )
    return SelfPlayWorkerResult(worker_index=batch.worker_index, samples=samples)


class _NetworkShape:
    def __init__(self, action_size: int) -> None:
        self.action_size = int(action_size)
