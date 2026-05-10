from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .network import AlphaZeroNet

InferenceRequest = tuple[int, int, np.ndarray, np.ndarray] | None
InferenceResponse = tuple[int, np.ndarray, np.ndarray]


class InferenceClient:
    def __init__(self, worker_id: int, request_queue: Any, response_queue: Any) -> None:
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._ids = itertools.count()

    def infer(self, infos: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        request_id = next(self._ids)
        self.request_queue.put(
            (
                self.worker_id,
                request_id,
                np.asarray(infos, dtype=np.float32),
                np.asarray(masks, dtype=bool),
            )
        )
        while True:
            response_id, priors, values = self.response_queue.get()
            if response_id == request_id:
                return priors, values
            raise RuntimeError(
                f"inference response id mismatch: expected {request_id}, got {response_id}"
            )


@dataclass
class InferenceServer:
    network: AlphaZeroNet
    device: torch.device
    request_queue: Any
    response_queues: list[Any]
    max_batch: int = 64
    batch_timeout_seconds: float = 0.001

    def __post_init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.forward_batches = 0
        self.forward_requests = 0
        self.forward_positions = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self.network.eval()
        self._thread = threading.Thread(target=self._run, name="ismcts-inference-server")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.request_queue.put(None)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self) -> InferenceServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                break
            batch: list[tuple[int, int, np.ndarray, np.ndarray]] = [first]
            rows = _request_rows(first)
            deadline = time.perf_counter() + self.batch_timeout_seconds
            while rows < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                batch.append(item)
                rows += _request_rows(item)
            self._serve(batch)

    def _serve(self, batch: list[tuple[int, int, np.ndarray, np.ndarray]]) -> None:
        infos = np.concatenate([_ensure_2d(item[2]) for item in batch], axis=0)
        masks = np.concatenate([_ensure_2d(item[3]) for item in batch], axis=0)
        with torch.inference_mode():
            x = torch.as_tensor(infos, dtype=torch.float32, device=self.device)
            legal = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            logits, values = self.network(x, legal)
            probs = torch.softmax(logits, dim=-1).masked_fill(~legal, 0.0)
            normalizer = probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            priors = (probs / normalizer).detach().cpu().numpy()
            values_np = values.detach().cpu().numpy()
        cursor = 0
        for worker_id, request_id, request_infos, _request_masks in batch:
            size = _ensure_2d(request_infos).shape[0]
            self.response_queues[worker_id].put(
                (
                    request_id,
                    priors[cursor : cursor + size].astype(np.float32, copy=False),
                    values_np[cursor : cursor + size].astype(np.float32, copy=False),
                )
            )
            cursor += size
        self.forward_batches += 1
        self.forward_requests += len(batch)
        self.forward_positions += int(infos.shape[0])


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[None, :]
    return array


def _request_rows(item: tuple[int, int, np.ndarray, np.ndarray]) -> int:
    return int(_ensure_2d(item[2]).shape[0])
