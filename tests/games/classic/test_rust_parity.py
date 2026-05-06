import json
from pathlib import Path
import random
import subprocess

import coolrl_lost_cities.games.classic as classic
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig, build_deck


LOST_CITIES_DIR = Path(classic.__file__).resolve().parent
FIXTURE_DIR = LOST_CITIES_DIR / "fixtures"
RUST_CORE_DIR = LOST_CITIES_DIR / "rust_core"


def _run_probe(*args: str) -> dict:
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--bin", "lost_cities_probe", "--", *args],
        cwd=RUST_CORE_DIR,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def _python_trace(path: Path) -> dict:
    fixture = json.loads(path.read_text())
    config = LostCitiesConfig(**fixture["config"])
    state = GameState.new_game_from_deck(fixture["initial_deck"], config)
    steps = []
    for step in fixture["steps"]:
        action = step["action"]
        if action is not None:
            state.apply_unified_action(action)
        state.validate_invariants()
        snapshot = state.to_snapshot()
        steps.append(
            {
                "action": action,
                "phase": state.phase,
                "current_player": state.current_player,
                "turn_count": state.turn_count,
                "terminal": state.terminal,
                "pending_discarded_color": state.pending_discarded_color,
                "score_diff_player0": state.score_diff(0),
                "legal_mask": state.unified_legal_mask(),
                "deck": snapshot["deck"],
                "hands": snapshot["hands"],
                "expeditions": snapshot["expeditions"],
                "discards": snapshot["discards"],
            }
        )
    return {"config": config.to_snapshot(), "steps": steps}


def test_rust_default_config_matches_python_default() -> None:
    assert _run_probe("defaults") == LostCitiesConfig().to_snapshot()


def test_rust_fixture_trace_matches_python_core() -> None:
    fixture_path = FIXTURE_DIR / "canonical_small.json"
    assert _run_probe("trace", str(fixture_path)) == _python_trace(fixture_path)


def test_rust_randomized_fixture_traces_match_python_core(tmp_path: Path) -> None:
    config = LostCitiesConfig(
        n_colors=3,
        n_ranks=5,
        min_rank=2,
        n_handshakes=1,
        hand_size=5,
    )

    for seed in range(12):
        deck = build_deck(config)
        rng = random.Random(seed)
        rng.shuffle(deck)
        state = GameState.new_game_from_deck(deck, config)
        steps = [{"action": None}]

        for _ in range(64):
            if state.terminal:
                break
            legal = [
                index
                for index, is_legal in enumerate(state.unified_legal_mask())
                if is_legal
            ]
            action = rng.choice(legal)
            state.apply_unified_action(action)
            steps.append({"action": action})

        fixture_path = tmp_path / f"parity_{seed}.json"
        fixture_path.write_text(
            json.dumps(
                {
                    "config": config.to_snapshot(),
                    "initial_deck": [card.to_snapshot() for card in deck],
                    "steps": steps,
                }
            )
        )
        assert _run_probe("trace", str(fixture_path)) == _python_trace(fixture_path)


def test_rust_engine_contract_is_checked_from_python() -> None:
    result = _run_probe("engine")

    assert result["duplicate_kind"] == "AlreadyExists"
    assert result["missing_config_kind"] == "InvalidArgument"
    assert result["empty_session_kind"] == "InvalidArgument"
    assert result["unknown_session_kind"] == "NotFound"
    assert result["invalid_observer_kind"] == "InvalidArgument"
    assert result["invalid_observer_state_unchanged"] is True
    assert result["invalid_observer_action_still_applies"] is True
    assert result["phase_flow"][:2] == [
        {
            "state_version": 0,
            "current_player": 0,
            "observer_player": 0,
            "phase": "card",
            "terminal": False,
        },
        {
            "state_version": 1,
            "current_player": 0,
            "observer_player": 0,
            "phase": "draw",
            "terminal": False,
        },
    ]
    assert result["phase_flow"][2]["state_version"] == 2
    assert result["stale_kind"] == "FailedPrecondition"
    assert result["end_session_counts"] == [1, 0]
    assert result["off_turn_legal_empty"] is True
    assert result["full_session_terminal_reward_matches"] is True
    assert result["full_session_final_scores_match"] is True
    assert result["terminal_reject_kind"] == "FailedPrecondition"
    assert result["deterministic_match"] is True


def test_rust_grpc_contract_is_checked_from_python() -> None:
    result = _run_probe("grpc")

    assert result == {
        "round_trip_phase": "card",
        "opponent_legal_empty": True,
        "stale_code": "FailedPrecondition",
        "invalid_observer_code": "InvalidArgument",
        "invalid_observer_state_unchanged": True,
        "ended_session_code": "NotFound",
    }


def test_rust_core_has_no_native_tests_left() -> None:
    rust_files = [
        path
        for path in (RUST_CORE_DIR / "src").rglob("*.rs")
        if "target" not in path.parts
    ]
    for path in rust_files:
        text = path.read_text()
        assert "#[test]" not in text
        assert "#[tokio::test" not in text

    tests_dir = RUST_CORE_DIR / "tests"
    assert not list(tests_dir.glob("*.rs"))
