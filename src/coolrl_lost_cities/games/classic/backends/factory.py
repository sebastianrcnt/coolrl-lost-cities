from __future__ import annotations

from ..game import LostCitiesConfig
from ..interfaces import BackendName, LostCitiesBackend
from .python import PythonLostCitiesBackend
from .rust import RustLostCitiesBackend


def build_lost_cities_backend(
    backend: BackendName,
    config: LostCitiesConfig,
    seed: int | None,
) -> LostCitiesBackend:
    if backend == "python":
        return PythonLostCitiesBackend(config, seed)
    if backend == "rust":
        return RustLostCitiesBackend(config, seed)
    raise ValueError(f"unknown backend: {backend}")
