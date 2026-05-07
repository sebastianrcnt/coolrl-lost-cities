from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from sys import stdout
from typing import Any, Protocol


class RunTracker(Protocol):
    def log_event(self, message: str) -> None:
        """Record a low-frequency training event."""

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        """Record one flat metric payload for a completed training step."""

    def close(self) -> None:
        """Flush and close tracker resources."""


def log_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class NullRunTracker:
    def log_event(self, message: str) -> None:
        pass

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class CompositeRunTracker:
    def __init__(self, trackers: list[RunTracker]):
        self.trackers = trackers

    def log_event(self, message: str) -> None:
        for tracker in self.trackers:
            tracker.log_event(message)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step)

    def close(self) -> None:
        for tracker in self.trackers:
            tracker.close()


class ConsoleRunTracker:
    def log_event(self, message: str) -> None:
        print(f"{log_timestamp()} {message}", file=stdout, flush=True)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class WandbRunTracker:
    def __init__(
        self,
        *,
        project: str,
        run_dir: str | Path,
        name: str | None = None,
        mode: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ):
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "wandb is not installed. Install with: uv sync --extra wandb"
            ) from exc
        run_dir_path = Path(run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=name,
            mode=mode,
            config=config or {},
            dir=str(run_dir_path),
            tags=tags,
            notes=notes,
            reinit=True,
        )

    def log_event(self, message: str) -> None:
        # Human-readable events stay in train.log / console; wandb is purely for metrics.
        pass

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._wandb.finish()


class FileRunTracker:
    def __init__(
        self,
        *,
        log_path: str | Path,
        metrics_path: str | Path,
        progress_path: str | Path,
    ):
        self.log_path = Path(log_path)
        self.metrics_path = Path(metrics_path)
        self.progress_path = Path(progress_path)

    def log_event(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{log_timestamp()} {message}\n")

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.progress_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def close(self) -> None:
        pass
