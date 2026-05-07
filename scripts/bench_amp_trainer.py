from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.cli import _with_overrides
from coolrl_lost_cities.games.classic.deep_cfr.config import load_config
from coolrl_lost_cities.games.classic.deep_cfr.memory import TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.tracking import NullRunTracker
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "deep_cfr" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trainer-side AMP train phases.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _seed_memories(trainer: DeepCFRTrainer) -> None:
    rng = np.random.default_rng(trainer.config.run.seed + 909)
    action_size = trainer.action_size
    input_dim = trainer.input_dim
    advantage_count = (
        trainer.config.optimization.advantage_batch_size
        * trainer.config.optimization.advantage_updates_per_iteration
    )
    strategy_count = (
        trainer.config.optimization.strategy_batch_size
        * trainer.config.optimization.strategy_updates_per_iteration
    )
    for player in range(2):
        for index in range(max(advantage_count, trainer.config.optimization.advantage_batch_size)):
            legal = rng.random(action_size) > 0.25
            legal[int(rng.integers(0, action_size))] = True
            trainer.advantage_memories[player].add(
                TrainingSample(
                    info_state=rng.normal(size=input_dim).astype(np.float32),
                    target=rng.normal(scale=20.0, size=action_size).astype(np.float32),
                    legal_mask=legal,
                    iteration=index + 1,
                    player=player,
                ),
                rng,
            )
    for index in range(max(strategy_count, trainer.config.optimization.strategy_batch_size)):
        legal = rng.random(action_size) > 0.25
        legal[int(rng.integers(0, action_size))] = True
        target = np.zeros(action_size, dtype=np.float32)
        weights = rng.random(np.count_nonzero(legal)).astype(np.float32)
        weights /= weights.sum()
        target[legal] = weights
        trainer.strategy_memory.add(
            TrainingSample(
                info_state=rng.normal(size=input_dim).astype(np.float32),
                target=target,
                legal_mask=legal,
                iteration=index + 1,
                player=-1,
            ),
            rng,
        )


def _new_trainer(*, use_amp: bool, config_path: str, device: str) -> DeepCFRTrainer:
    config = _with_overrides(
        load_config(config_path),
        {
            "run": {"use_amp": use_amp, "device": device, "max_iterations": 1},
            "checkpoint": {"save_every": 0, "save_latest": False},
            "evaluation": {"eval_every": 0},
        },
    )
    trainer = DeepCFRTrainer(
        config=config,
        game_config=config.rules.to_lost_cities_config(seed=config.run.seed),
        device=device,
        tracker=NullRunTracker(),
    )
    _seed_memories(trainer)
    return trainer


def _measure(trainer: DeepCFRTrainer, runs: int) -> list[float]:
    durations: list[float] = []
    for iteration in range(runs):
        trainer.iteration = iteration + 1
        trainer._runtime_metrics = {}
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        trainer._train_advantage_networks()
        trainer._train_strategy_network()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return durations


def _summary(values: list[float], warmup: int) -> tuple[float, float, float]:
    measured = values[warmup:]
    if not measured:
        raise ValueError("warmup must be less than runs")
    return (
        statistics.mean(measured),
        statistics.median(measured),
        sorted(measured)[math.ceil(0.95 * (len(measured) - 1))],
    )


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.warmup < 0 or args.warmup >= args.runs:
        raise SystemExit("--warmup must be non-negative and less than --runs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu for CPU fallback smoke.")

    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Runs: {args.runs}  Warmup: {args.warmup}")
    print()
    rows: list[tuple[str, float, float, float]] = []
    for use_amp, label in ((False, "fp32"), (True, "amp")):
        trainer = _new_trainer(use_amp=use_amp, config_path=args.config, device=args.device)
        values = _measure(trainer, args.runs)
        mean, p50, p95 = _summary(values, args.warmup)
        rows.append((label, mean, p50, p95))
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{'mode':<8} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10}")
    for label, mean, p50, p95 in rows:
        print(f"{label:<8} {mean * 1000.0:>10.2f} {p50 * 1000.0:>10.2f} {p95 * 1000.0:>10.2f}")
    fp32_mean = rows[0][1]
    amp_mean = rows[1][1]
    print()
    print(f"speedup: {fp32_mean / amp_mean:.2f}x")


if __name__ == "__main__":
    main()
