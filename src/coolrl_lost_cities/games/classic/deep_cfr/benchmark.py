from __future__ import annotations

import time

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer


def benchmark_traversal(
    config: DeepCFRConfig | None = None,
    *,
    num_workers: int = 0,
) -> dict[str, float | int]:
    base = config or DeepCFRConfig.model_validate(
        {
            "run": {"iterations": 1},
            "traversal": {"traversals_per_iteration": 8, "max_depth": 4},
            "checkpoint": {"save_every_iteration": False},
        }
    )
    data = base.model_dump(mode="python")
    data["traversal"]["num_workers"] = num_workers
    data["checkpoint"]["save_every_iteration"] = False
    data["evaluation"]["eval_every"] = 0
    cfg = DeepCFRConfig.model_validate(data)
    trainer = DeepCFRTrainer(cfg, cfg.rules.to_lost_cities_config(seed=cfg.run.seed))
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


def benchmark_traversal_modes(
    config: DeepCFRConfig | None = None,
) -> dict[str, dict[str, float | int]]:
    single = benchmark_traversal(config, num_workers=0)
    multi = benchmark_traversal(config, num_workers=2)
    speedup = float(multi["nodes_per_second"]) / max(float(single["nodes_per_second"]), 1.0e-12)
    return {
        "single": single,
        "multi": multi,
        "summary": {"speedup": speedup},
    }
