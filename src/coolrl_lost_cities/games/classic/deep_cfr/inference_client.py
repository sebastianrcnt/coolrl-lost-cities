from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.inference_buffers import (
    InferenceBuffers,
    InferenceClientHandles,
)

NETWORK_KIND_ADVANTAGE = "advantage"
NETWORK_KIND_STRATEGY = "strategy"
NETWORK_KIND_LEAGUE = "league"


@dataclass(frozen=True)
class RequestMessage:
    slot_id: int
    network_kind: str
    player: int
    network_index: int


class InferenceClient:
    def __init__(self, handles: InferenceClientHandles) -> None:
        self.handles = handles
        self._request_shm, self._requests = InferenceBuffers.attach_requests(handles)
        self._response_shm, self._responses = InferenceBuffers.attach_responses(handles)
        self._slot_id = int(self.handles.free_slots.get())
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.handles.ready_events[self._slot_id].clear()
            self.handles.free_slots.put(self._slot_id)
            self._closed = True
        self._request_shm.close()
        self._response_shm.close()

    def forward(
        self,
        *,
        network_kind: str,
        player: int,
        network_index: int,
        state: np.ndarray,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.handles.input_dim,):
            raise ValueError(
                f"state must have shape {(self.handles.input_dim,)}, got {state.shape}"
            )
        if self._closed:
            raise RuntimeError("InferenceClient is closed")
        slot_id = self._slot_id
        ready = self.handles.ready_events[slot_id]
        ready.clear()
        self._requests[slot_id, :] = state
        self.handles.request_queue.put(
            RequestMessage(
                slot_id=slot_id,
                network_kind=network_kind,
                player=player,
                network_index=network_index,
            )
        )
        ready.wait()
        ready.clear()
        return self._responses[slot_id, :].copy()


class NetworkProxy:
    def __init__(
        self,
        client: InferenceClient,
        *,
        network_kind: str,
        player: int,
        network_index: int,
    ) -> None:
        self.client = client
        self.network_kind = network_kind
        self.player = player
        self.network_index = network_index

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(
                f"NetworkProxy expects a single-row tensor, got shape {tuple(x.shape)}"
            )
        state = x.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        result = self.client.forward(
            network_kind=self.network_kind,
            player=self.player,
            network_index=self.network_index,
            state=state,
        )
        return torch.from_numpy(result).unsqueeze(0)
