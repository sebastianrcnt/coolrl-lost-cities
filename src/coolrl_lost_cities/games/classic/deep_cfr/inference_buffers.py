from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceClientHandles:
    request_shm_name: str
    response_shm_name: str
    num_slots: int
    input_dim: int
    action_size: int
    request_queue: Any
    weight_queue: Any
    free_slots: Any
    ready_events: list[Any]
    weight_sync_event: Any
    stats_queue: Any


class InferenceBuffers:
    def __init__(
        self,
        *,
        num_slots: int,
        input_dim: int,
        action_size: int,
        mp_context: mp.context.BaseContext | None = None,
    ) -> None:
        self.num_slots = int(num_slots)
        self.input_dim = int(input_dim)
        self.action_size = int(action_size)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.action_size <= 0:
            raise ValueError("action_size must be positive")

        self._ctx = mp_context or mp.get_context("spawn")

        request_nbytes = self.num_slots * self.input_dim * np.dtype(np.float32).itemsize
        response_nbytes = self.num_slots * self.action_size * np.dtype(np.float32).itemsize
        self._request_shm = SharedMemory(create=True, size=request_nbytes)
        self._response_shm = SharedMemory(create=True, size=response_nbytes)
        self.requests = np.ndarray(
            (self.num_slots, self.input_dim),
            dtype=np.float32,
            buffer=self._request_shm.buf,
        )
        self.responses = np.ndarray(
            (self.num_slots, self.action_size),
            dtype=np.float32,
            buffer=self._response_shm.buf,
        )
        self.requests.fill(0.0)
        self.responses.fill(0.0)

        self.request_queue = self._ctx.Queue()
        self.weight_queue = self._ctx.Queue()
        self.free_slots = self._ctx.Queue()
        self.ready_events = [self._ctx.Event() for _ in range(self.num_slots)]
        self.weight_sync_event = self._ctx.Event()
        self.stats_queue = self._ctx.Queue()
        for slot in range(self.num_slots):
            self.free_slots.put(slot)

    def handles(self) -> InferenceClientHandles:
        return InferenceClientHandles(
            request_shm_name=self._request_shm.name,
            response_shm_name=self._response_shm.name,
            num_slots=self.num_slots,
            input_dim=self.input_dim,
            action_size=self.action_size,
            request_queue=self.request_queue,
            weight_queue=self.weight_queue,
            free_slots=self.free_slots,
            ready_events=self.ready_events,
            weight_sync_event=self.weight_sync_event,
            stats_queue=self.stats_queue,
        )

    @staticmethod
    def attach_requests(handles: InferenceClientHandles) -> tuple[SharedMemory, np.ndarray]:
        shm = SharedMemory(name=handles.request_shm_name)
        array = np.ndarray(
            (handles.num_slots, handles.input_dim),
            dtype=np.float32,
            buffer=shm.buf,
        )
        return shm, array

    @staticmethod
    def attach_responses(handles: InferenceClientHandles) -> tuple[SharedMemory, np.ndarray]:
        shm = SharedMemory(name=handles.response_shm_name)
        array = np.ndarray(
            (handles.num_slots, handles.action_size),
            dtype=np.float32,
            buffer=shm.buf,
        )
        return shm, array

    def close(self) -> None:
        self._request_shm.close()
        self._response_shm.close()

    def unlink(self) -> None:
        self._request_shm.unlink()
        self._response_shm.unlink()

    def release(self) -> None:
        self.close()
        self.unlink()
