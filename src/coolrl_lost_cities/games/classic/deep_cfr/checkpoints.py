from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def load_checkpoint(path: str | Path, *, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device)
