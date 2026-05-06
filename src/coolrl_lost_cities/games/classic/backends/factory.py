from __future__ import annotations

from ..game import LostCitiesConfig
from ..interfaces import BackendName, LostCitiesBackend
from .python import PythonLostCitiesBackend


def build_backend(
    backend: BackendName,
    config: LostCitiesConfig,
    seed: int | None,
) -> LostCitiesBackend:
    if backend == "python":
        return PythonLostCitiesBackend(config, seed)
    raise ValueError(f"unknown backend: {backend}")
