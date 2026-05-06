from __future__ import annotations

import json

from coolrl_lost_cities.games.classic.resources import asset_path, theme_path


def test_theme_resource_is_packaged() -> None:
    theme = json.loads(theme_path().read_text())

    assert isinstance(theme, dict)


def test_asset_path_returns_named_asset() -> None:
    assert asset_path("pygame_pvp_theme.json").name == "pygame_pvp_theme.json"
