from __future__ import annotations

from importlib.abc import Traversable
from importlib.resources import files


def asset_path(name: str) -> Traversable:
    return files(__package__).joinpath("assets", name)


def theme_path() -> Traversable:
    return asset_path("pygame_pvp_theme.json")
