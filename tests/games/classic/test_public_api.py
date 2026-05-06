from __future__ import annotations

import coolrl_lost_cities.games.classic as classic


def test_classic_package_exports_common_game_api() -> None:
    config = classic.classic_config(seed=1)
    state = classic.GameState.new_game(config)
    bot = classic.build_bot("random", seed=1)

    action = bot.act(state)
    assert state.unified_legal_mask()[state.to_unified_action(action)]
    assert state.config.deck_size == 60


def test_classic_package_exports_backend_alias() -> None:
    backend = classic.build_backend("python", classic.classic_config(), seed=1)

    assert isinstance(backend.snapshot(), classic.Snapshot)


def test_classic_package_exports_bot_registry_helpers() -> None:
    assert "random" in classic.available_bot_names()
    assert isinstance(classic.build_bot("random", seed=1), classic.LostCitiesBot)
