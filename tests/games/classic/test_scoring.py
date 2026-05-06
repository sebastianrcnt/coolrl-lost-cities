from coolrl_lost_cities.games.classic.game import Card, LostCitiesConfig, score_expedition


def test_empty_expedition_scores_zero() -> None:
    assert score_expedition([], LostCitiesConfig()) == 0


def test_handshake_only_deepens_negative_score() -> None:
    config = LostCitiesConfig(n_handshakes=3)
    assert score_expedition([Card(0, 0), Card(0, 0)], config) == -60


def test_numbers_only_score() -> None:
    config = LostCitiesConfig()
    expedition = [Card(0, 1), Card(0, 3), Card(0, 5)]
    assert score_expedition(expedition, config) == (2 + 4 + 6 - 20)


def test_two_handshakes_and_three_numbers() -> None:
    config = LostCitiesConfig(n_handshakes=3)
    expedition = [Card(0, 0), Card(0, 0), Card(0, 1), Card(0, 2), Card(0, 3)]
    assert score_expedition(expedition, config) == (2 + 3 + 4 - 20) * 3


def test_bonus_threshold_adds_bonus() -> None:
    config = LostCitiesConfig(n_ranks=9, n_handshakes=3, bonus_threshold=4, bonus_amount=20)
    expedition = [Card(0, 1), Card(0, 2), Card(0, 3), Card(0, 4)]
    assert score_expedition(expedition, config) == (2 + 3 + 4 + 5 - 20) + 20


def test_manual_multi_color_examples() -> None:
    config = LostCitiesConfig(n_handshakes=3)
    assert score_expedition([Card(1, 0), Card(1, 5)], config) == (6 - 20) * 2
    assert score_expedition([Card(2, 4), Card(2, 5)], config) == 5 + 6 - 20
    assert score_expedition([Card(0, 0), Card(0, 0), Card(0, 0)], config) == -80
