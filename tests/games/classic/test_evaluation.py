from coolrl_lost_cities.games.classic import (
    LostCitiesConfig,
    build_bot,
    make_bot_factory,
    play_game_for_evaluation,
    play_match,
)
from coolrl_lost_cities.games.classic.evaluation import main


def test_play_game_for_evaluation_finishes_small_match() -> None:
    config = LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=1, hand_size=5)

    state, result = play_game_for_evaluation(
        build_bot("random", seed=1),
        build_bot("passive-discard", seed=2),
        config,
        seed=3,
        max_steps=200,
    )

    assert state.terminal is True
    assert result.timed_out is False
    assert result.steps > 0
    assert result.score_diff0 == result.score0 - result.score1


def test_play_match_alternates_seats_and_reports_rates() -> None:
    config = LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=1, hand_size=5)

    result = play_match(
        make_bot_factory("random"),
        make_bot_factory("passive-discard"),
        config,
        games=4,
        seed=10,
        max_steps=200,
    )

    assert result.games == 4
    assert result.wins0 + result.wins1 + result.draws == 4
    assert result.avg_game_length > 0.0
    assert result.games_per_second > 0.0
    assert result.steps_per_second > 0.0


def test_evaluation_cli_smoke_json(capsys) -> None:
    main(
        [
            "--bot0",
            "random",
            "--bot1",
            "passive-discard",
            "--games",
            "2",
            "--seed",
            "20",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert '"games": 2' in captured.out
    assert '"win_rate0"' in captured.out
