from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol


class RunLogger(Protocol):
    def info(self, message: str) -> None:
        """Record a low-frequency training log message."""


def log_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class NullRunLogger:
    def info(self, message: str) -> None:
        pass


class FileRunLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def info(self, message: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{log_timestamp()} {message}\n")
