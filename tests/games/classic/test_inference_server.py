from __future__ import annotations

import multiprocessing as mp
import threading

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.inference_buffers import (
    InferenceBuffers,
    InferenceClientHandles,
)
from coolrl_lost_cities.games.classic.deep_cfr.inference_client import (
    NETWORK_KIND_ADVANTAGE,
    InferenceClient,
    RequestMessage,
)
from coolrl_lost_cities.games.classic.deep_cfr.inference_server import (
    InferenceServerController,
)
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer


def _child_write_request_row(handles: InferenceClientHandles) -> None:
    request_shm, requests = InferenceBuffers.attach_requests(handles)
    try:
        requests[1, :] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    finally:
        request_shm.close()


def test_inference_buffers_shared_memory_round_trip() -> None:
    ctx = mp.get_context("spawn")
    buffers = InferenceBuffers(num_slots=2, input_dim=3, action_size=4, mp_context=ctx)
    try:
        process = ctx.Process(target=_child_write_request_row, args=(buffers.handles(),))
        process.start()
        process.join(timeout=10.0)

        assert process.exitcode == 0
        np.testing.assert_allclose(buffers.requests[1], np.array([1.0, 2.0, 3.0]))
    finally:
        buffers.release()


def test_inference_client_forwards_through_request_queue() -> None:
    buffers = InferenceBuffers(num_slots=2, input_dim=3, action_size=3)
    handles = buffers.handles()
    client = InferenceClient(handles)

    def serve_one() -> None:
        request = handles.request_queue.get(timeout=5.0)
        assert isinstance(request, RequestMessage)
        buffers.responses[request.slot_id, :] = buffers.requests[request.slot_id, :] * 2.0
        handles.ready_events[request.slot_id].set()

    thread = threading.Thread(target=serve_one)
    thread.start()
    try:
        result = client.forward(
            network_kind=NETWORK_KIND_ADVANTAGE,
            player=0,
            network_index=0,
            state=np.array([2.0, 3.0, 4.0], dtype=np.float32),
        )
        np.testing.assert_allclose(result, np.array([4.0, 6.0, 8.0], dtype=np.float32))
    finally:
        thread.join(timeout=5.0)
        client.close()
        buffers.release()


def test_inference_server_matches_network_forward() -> None:
    config = DeepCFRConfig.model_validate(
        {
            "network": {"hidden_size": 8, "num_layers": 1},
            "inference_server": {"device": "cpu", "num_slots": 4, "max_batch": 4},
        }
    )
    input_dim = 3
    action_size = 2
    network = DeepCFRMLP.from_config(input_dim, action_size, config.network)
    state_dict = network.state_dict()
    controller = InferenceServerController(
        input_dim=input_dim,
        action_size=action_size,
        num_slots=4,
        network_config=config.network,
        server_config=config.inference_server,
    )
    client = InferenceClient(controller.handles)
    try:
        controller.push_weights(
            advantage_networks=[state_dict, state_dict],
            strategy_network=state_dict,
            league_advantage_networks=[],
        )
        state = np.array([0.25, -0.5, 1.5], dtype=np.float32)
        result = client.forward(
            network_kind=NETWORK_KIND_ADVANTAGE,
            player=0,
            network_index=0,
            state=state,
        )
        with torch.inference_mode():
            expected = network(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()
        np.testing.assert_allclose(result, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        client.close()
        controller.shutdown()


def test_deep_cfr_training_smoke_with_cpu_inference_server(tmp_path) -> None:
    config = DeepCFRConfig.model_validate(
        {
            "run": {"max_iterations": 1, "seed": 11, "device": "cpu"},
            "network": {"hidden_size": 8, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 1,
                "max_depth": 2,
                "max_nodes_per_traversal": 64,
                "num_workers": 2,
                "worker_chunk_size": 1,
                "opponent_policy": "average_strategy",
                "inference_backend": "server",
            },
            "optimization": {
                "advantage_batch_size": 2,
                "strategy_batch_size": 2,
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
            },
            "checkpoint": {"save_every": 0, "save_latest": False},
            "evaluation": {"eval_every": 0},
            "inference_server": {"device": "cpu", "num_slots": 8, "max_batch": 4},
        }
    )
    trainer = DeepCFRTrainer(config=config, run_dir=tmp_path, device="cpu")

    metrics = trainer.train()

    assert len(metrics) == 1
    assert metrics[0].traversal_nodes > 0
    assert metrics[0].advantage_samples > 0
