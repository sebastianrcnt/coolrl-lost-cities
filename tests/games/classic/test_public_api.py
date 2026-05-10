from __future__ import annotations

import coolrl_lost_cities.games.classic as classic


def test_classic_package_exports_common_game_api() -> None:
    config = classic.classic_config(seed=1)
    state = classic.GameState.new_game(config)
    bot = classic.build_bot("random", seed=1)

    action = bot.act(state)
    assert state.unified_legal_mask()[state.to_unified_action(action)]
    assert state.config.deck_size == 60


def test_classic_package_exports_snapshot_alias() -> None:
    state = classic.GameState.new_game(classic.classic_config(seed=1))
    snapshot = classic.Snapshot(
        config=state.config,
        deck=list(state.deck),
        hands=[list(hand) for hand in state.hands],
        expeditions=[
            [list(expedition) for expedition in player_expeditions]
            for player_expeditions in state.expeditions
        ],
        discards=[list(discard) for discard in state.discards],
        current_player=state.current_player,
        phase=state.phase,
        pending_discarded_color=state.pending_discarded_color,
        turn_count=state.turn_count,
        terminal=state.terminal,
        legal_mask=state.unified_legal_mask(),
    )

    assert snapshot.score_diff(0) == state.score_diff(0)


def test_classic_package_exports_bot_registry_helpers() -> None:
    assert "random" in classic.available_bot_names()
    assert isinstance(classic.build_bot("random", seed=1), classic.LostCitiesPolicy)


def test_classic_bot_registry_accepts_reproduction_opponent_names() -> None:
    for name in [
        "random",
        "discard_only",
        "heuristic_balanced",
        "heuristic_aggressive",
        "heuristic_cautious",
        "heuristic_noisy",
    ]:
        assert isinstance(classic.build_bot(name, seed=1), classic.LostCitiesPolicy)
