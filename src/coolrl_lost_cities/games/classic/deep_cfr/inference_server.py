from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.config import InferenceServerConfig, NetworkConfig
from coolrl_lost_cities.games.classic.deep_cfr.inference_buffers import (
    InferenceBuffers,
    InferenceClientHandles,
)
from coolrl_lost_cities.games.classic.deep_cfr.inference_client import (
    NETWORK_KIND_ADVANTAGE,
    NETWORK_KIND_LEAGUE,
    NETWORK_KIND_STRATEGY,
    RequestMessage,
)
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP


@dataclass(frozen=True)
class ShutdownMessage:
    pass


@dataclass(frozen=True)
class WeightUpdateMessage:
    advantage_networks: list[dict[str, torch.Tensor]]
    strategy_network: dict[str, torch.Tensor] | None
    league_advantage_networks: list[list[dict[str, torch.Tensor]]]


@dataclass(frozen=True)
class BatchStatsMessage:
    batch_size: int
    group_count: int


def _resolve_server_device(device: str) -> torch.device:
    token = device.strip().lower()
    if token == "auto":
        token = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(token)


def _new_network(
    *,
    input_dim: int,
    action_size: int,
    network_config: NetworkConfig,
    device: torch.device,
) -> torch.nn.Module:
    return DeepCFRMLP.from_config(input_dim, action_size, network_config).to(device).eval()


def _load_state_dict_on_device(
    network: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    network.load_state_dict({name: value.to(device) for name, value in state_dict.items()})
    network.eval()


def run_inference_server(
    handles: InferenceClientHandles,
    *,
    network_config_data: dict[str, Any],
    server_config_data: dict[str, Any],
) -> None:
    network_config = NetworkConfig.model_validate(network_config_data)
    server_config = InferenceServerConfig.model_validate(server_config_data)
    device = _resolve_server_device(server_config.device)
    request_shm, requests = InferenceBuffers.attach_requests(handles)
    response_shm, responses = InferenceBuffers.attach_responses(handles)

    advantage_networks = [
        _new_network(
            input_dim=handles.input_dim,
            action_size=handles.action_size,
            network_config=network_config,
            device=device,
        )
        for _ in range(2)
    ]
    strategy_network = _new_network(
        input_dim=handles.input_dim,
        action_size=handles.action_size,
        network_config=network_config,
        device=device,
    )
    league_advantage_networks: list[list[torch.nn.Module]] = []

    try:
        while True:
            if _apply_pending_weight_updates(
                handles,
                advantage_networks,
                strategy_network,
                league_advantage_networks,
                input_dim=handles.input_dim,
                action_size=handles.action_size,
                network_config=network_config,
                device=device,
            ):
                continue

            try:
                first = handles.request_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            if isinstance(first, ShutdownMessage):
                return
            if not isinstance(first, RequestMessage):
                continue
            batch = [first]
            deadline = time.perf_counter() + max(0, server_config.batch_window_us) / 1_000_000.0
            while len(batch) < server_config.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = handles.request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(item, ShutdownMessage):
                    return
                if isinstance(item, RequestMessage):
                    batch.append(item)
            _serve_request_batch(
                batch,
                requests,
                responses,
                handles,
                advantage_networks,
                strategy_network,
                league_advantage_networks,
                device=device,
                use_amp=server_config.use_amp,
            )
    finally:
        request_shm.close()
        response_shm.close()


def _apply_pending_weight_updates(
    handles: InferenceClientHandles,
    advantage_networks: list[torch.nn.Module],
    strategy_network: torch.nn.Module,
    league_advantage_networks: list[list[torch.nn.Module]],
    *,
    input_dim: int,
    action_size: int,
    network_config: NetworkConfig,
    device: torch.device,
) -> bool:
    applied = False
    while True:
        try:
            item = handles.weight_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, ShutdownMessage):
            handles.request_queue.put(item)
            return True
        if not isinstance(item, WeightUpdateMessage):
            continue
        for network, state_dict in zip(advantage_networks, item.advantage_networks, strict=True):
            _load_state_dict_on_device(network, state_dict, device)
        if item.strategy_network is not None:
            _load_state_dict_on_device(strategy_network, item.strategy_network, device)
        league_advantage_networks[:] = []
        for snapshot in item.league_advantage_networks:
            snapshot_networks = [
                _new_network(
                    input_dim=input_dim,
                    action_size=action_size,
                    network_config=network_config,
                    device=device,
                )
                for _ in range(2)
            ]
            for network, state_dict in zip(snapshot_networks, snapshot, strict=True):
                _load_state_dict_on_device(network, state_dict, device)
            league_advantage_networks.append(snapshot_networks)
        handles.weight_sync_event.set()
        applied = True
    return applied


def _network_for_request(
    request: RequestMessage,
    advantage_networks: list[torch.nn.Module],
    strategy_network: torch.nn.Module,
    league_advantage_networks: list[list[torch.nn.Module]],
) -> torch.nn.Module:
    if request.network_kind == NETWORK_KIND_ADVANTAGE:
        return advantage_networks[request.network_index]
    if request.network_kind == NETWORK_KIND_STRATEGY:
        return strategy_network
    if request.network_kind == NETWORK_KIND_LEAGUE:
        return league_advantage_networks[request.network_index][request.player]
    raise ValueError(f"unknown network kind: {request.network_kind!r}")


def _serve_request_batch(
    batch: list[RequestMessage],
    requests: np.ndarray,
    responses: np.ndarray,
    handles: InferenceClientHandles,
    advantage_networks: list[torch.nn.Module],
    strategy_network: torch.nn.Module,
    league_advantage_networks: list[list[torch.nn.Module]],
    *,
    device: torch.device,
    use_amp: bool,
) -> None:
    groups: dict[tuple[str, int, int], list[RequestMessage]] = {}
    for request in batch:
        key = (request.network_kind, request.player, request.network_index)
        groups.setdefault(key, []).append(request)
    handles.stats_queue.put(BatchStatsMessage(batch_size=len(batch), group_count=len(groups)))

    with torch.inference_mode():
        for requests_for_network in groups.values():
            network = _network_for_request(
                requests_for_network[0],
                advantage_networks,
                strategy_network,
                league_advantage_networks,
            )
            slots = [request.slot_id for request in requests_for_network]
            x = torch.as_tensor(requests[slots, :], dtype=torch.float32, device=device)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    output = network(x)
            else:
                output = network(x)
            values = output.detach().to("cpu", dtype=torch.float32).numpy()
            for row, slot in enumerate(slots):
                responses[slot, :] = values[row]
                handles.ready_events[slot].set()


class InferenceServerController:
    def __init__(
        self,
        *,
        input_dim: int,
        action_size: int,
        num_slots: int,
        network_config: NetworkConfig,
        server_config: InferenceServerConfig,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._buffers = InferenceBuffers(
            num_slots=num_slots,
            input_dim=input_dim,
            action_size=action_size,
            mp_context=self._ctx,
        )
        self.handles = self._buffers.handles()
        self._process = self._ctx.Process(
            target=run_inference_server,
            kwargs={
                "handles": self.handles,
                "network_config_data": network_config.model_dump(mode="json"),
                "server_config_data": server_config.model_dump(mode="json"),
            },
            daemon=True,
        )
        self._process.start()

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def push_weights(
        self,
        *,
        advantage_networks: list[dict[str, torch.Tensor]],
        strategy_network: dict[str, torch.Tensor] | None,
        league_advantage_networks: list[list[dict[str, torch.Tensor]]],
        timeout: float = 60.0,
    ) -> None:
        if not self.is_alive:
            raise RuntimeError("inference server process is not alive")
        self.handles.weight_sync_event.clear()
        self.handles.weight_queue.put(
            WeightUpdateMessage(
                advantage_networks=advantage_networks,
                strategy_network=strategy_network,
                league_advantage_networks=league_advantage_networks,
            )
        )
        if not self.handles.weight_sync_event.wait(timeout=timeout):
            raise TimeoutError("timed out waiting for inference server weight sync")

    def drain_batch_stats(self) -> list[BatchStatsMessage]:
        stats: list[BatchStatsMessage] = []
        while True:
            try:
                item = self.handles.stats_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BatchStatsMessage):
                stats.append(item)
        return stats

    def shutdown(self, timeout: float = 10.0) -> None:
        try:
            self.handles.request_queue.put(ShutdownMessage())
            self.handles.weight_queue.put(ShutdownMessage())
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=timeout)
        finally:
            self._buffers.release()
