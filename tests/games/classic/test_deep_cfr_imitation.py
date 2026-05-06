from __future__ import annotations

from coolrl_lost_cities.games.classic.game import LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.imitation import (
    collect_safe_heuristic_samples,
    new_pretrained_strategy_network,
)


def test_collect_safe_heuristic_samples_shapes() -> None:
    x, y, legal = collect_safe_heuristic_samples(LostCitiesConfig(seed=61), games=1, seed=61)

    assert len(x) == len(y) == len(legal)
    assert x.ndim == 2
    assert y.shape == legal.shape
    assert y.sum(axis=1).min() == 1.0


def test_pretrain_strategy_network_smoke() -> None:
    network, metrics = new_pretrained_strategy_network(
        LostCitiesConfig(seed=67),
        hidden_size=16,
        games=1,
        seed=67,
        steps=1,
    )

    assert metrics.samples > 0
    assert metrics.loss >= 0.0
    assert network is not None
