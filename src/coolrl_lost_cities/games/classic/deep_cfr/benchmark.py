from __future__ import annotations

import time
from dataclasses import replace

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer
from coolrl_lost_cities.games.classic.game import classic_config


def benchmark_traversal(
    config: DeepCFRConfig | None = None,
    *,
    num_workers: int = 0,
) -> dict[str, float | int]:
    base = config or DeepCFRConfig(
        iterations=1,
        traversals_per_iteration=8,
        max_traversal_depth=4,
        save_every_iteration=False,
    )
    cfg = replace(base, num_workers=num_workers, save_every_iteration=False, eval_every=0)
    trainer = DeepCFRTrainer(cfg, classic_config(seed=cfg.seed))
    started = time.perf_counter()
    metrics = trainer.run_iteration(1)
    elapsed = time.perf_counter() - started
    return {
        "num_workers": num_workers,
        "elapsed_seconds": elapsed,
        "traversal_nodes": metrics.traversal_nodes,
        "nodes_per_second": metrics.traversal_nodes / max(elapsed, 1.0e-12),
        "advantage_samples": metrics.advantage_samples,
        "strategy_samples": metrics.strategy_samples,
    }
