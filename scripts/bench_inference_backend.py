from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from coolrl_lost_cities.games.classic.deep_cfr.cli import _with_overrides
from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig, load_config
from coolrl_lost_cities.games.classic.deep_cfr.tracking import FileRunTracker
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer

METRIC_KEYS = (
    "time/iteration_seconds",
    "time/traversal_seconds",
    "time/advantage_train_seconds",
    "time/strategy_train_seconds",
    "time/memory_add_seconds",
    "time/batch_tensor_seconds",
)

TABLE_COLUMNS = (
    ("iter", "time/iteration_seconds"),
    ("traversal", "time/traversal_seconds"),
    ("adv_train", "time/advantage_train_seconds"),
    ("strat_train", "time/strategy_train_seconds"),
    ("mem_add", "time/memory_add_seconds"),
    ("batch_tensor", "time/batch_tensor_seconds"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Deep CFR traversal inference backends.")
    parser.add_argument(
        "--config-local",
        default="configs/deep_cfr/default.yaml",
        help="Config for the local traversal inference backend.",
    )
    parser.add_argument(
        "--config-server",
        default="configs/deep_cfr/default_server.yaml",
        help="Config for the server traversal inference backend.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", help="Override run.device for both backend configs.")
    return parser.parse_args()


def _benchmark_overrides(
    *,
    backend: str,
    iterations: int,
    seed: int,
    device: str | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "run": {
            "max_iterations": iterations,
            "max_minutes": None,
            "seed": seed,
        },
        "traversal": {"inference_backend": backend},
        "checkpoint": {
            "save_latest": False,
            "save_every": 0,
        },
        "evaluation": {"eval_every": 0},
    }
    if device is not None:
        overrides["run"]["device"] = device
    return overrides


def _load_benchmark_config(
    path: str,
    *,
    backend: str,
    iterations: int,
    seed: int,
    device: str | None,
) -> DeepCFRConfig:
    config = load_config(path)
    return _with_overrides(
        config,
        _benchmark_overrides(
            backend=backend,
            iterations=iterations,
            seed=seed,
            device=device,
        ),
    )


def _read_metrics(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(rows: list[dict[str, float | int]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)


def _summarize(rows: list[dict[str, float | int]], *, warmup: int) -> dict[str, float]:
    measured = rows[warmup:]
    return {key: _mean(measured, key) for key in METRIC_KEYS}


def _clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_backend(
    *,
    name: str,
    config: DeepCFRConfig,
    run_dir: Path,
    warmup: int,
) -> dict[str, Any]:
    _clear_gpu_memory()
    tracker = FileRunTracker(
        log_path=run_dir / "train.log",
        metrics_path=run_dir / "metrics.jsonl",
        progress_path=run_dir / "runtime_progress.json",
    )
    trainer: DeepCFRTrainer | None = DeepCFRTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        run_dir=run_dir,
        device=config.run.device,
        tracker=tracker,
    )
    try:
        trainer.train()
        rows = _read_metrics(run_dir / "metrics.jsonl")
    finally:
        trainer = None
        _clear_gpu_memory()
    return {
        "backend": name,
        "run_dir": str(run_dir),
        "raw": rows,
        "mean": _summarize(rows, warmup=warmup),
    }


def _speedups(local: dict[str, float], server: dict[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        server_value = server.get(key, 0.0)
        result[key] = None if server_value <= 0.0 else local.get(key, 0.0) / server_value
    return result


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _format_multiplier(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _print_backend_table(results: dict[str, Any]) -> None:
    local = results["backends"]["local"]["mean"]
    server = results["backends"]["server"]["mean"]
    speedup = results["speedup"]
    rows = [
        ("local", [_format_seconds(local[key]) for _label, key in TABLE_COLUMNS]),
        ("server", [_format_seconds(server[key]) for _label, key in TABLE_COLUMNS]),
        ("speedup", [_format_multiplier(speedup[key]) for _label, key in TABLE_COLUMNS]),
    ]
    headers = ["Backend", *[label for label, _key in TABLE_COLUMNS]]
    widths = [max(len(headers[0]), *(len(row[0]) for row in rows))]
    for index, header in enumerate(headers[1:]):
        widths.append(max(len(header), *(len(row[1][index]) for row in rows)))

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for backend, values in rows:
        cells = [backend.ljust(widths[0])]
        cells.extend(value.rjust(widths[index + 1]) for index, value in enumerate(values))
        print("  ".join(cells))


def _print_projection(results: dict[str, Any]) -> None:
    print()
    print("1000-iteration projection")
    print("Backend  hours")
    for backend in ("local", "server"):
        mean_iter = results["backends"][backend]["mean"]["time/iteration_seconds"]
        hours = mean_iter * 1000.0 / 3600.0
        print(f"{backend:<7}  {hours:.2f}h")


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.warmup >= args.iterations:
        raise SystemExit("--warmup must be less than --iterations")

    local_base = load_config(args.config_local)
    seed = local_base.run.seed
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    bench_dir = Path("runs/bench") / f"{timestamp}_inference_backend"
    bench_dir.mkdir(parents=True, exist_ok=False)

    local_config = _load_benchmark_config(
        args.config_local,
        backend="local",
        iterations=args.iterations,
        seed=seed,
        device=args.device,
    )
    server_config = _load_benchmark_config(
        args.config_server,
        backend="server",
        iterations=args.iterations,
        seed=seed,
        device=args.device,
    )

    local_result = _run_backend(
        name="local",
        config=local_config,
        run_dir=bench_dir / "local",
        warmup=args.warmup,
    )
    server_result = _run_backend(
        name="server",
        config=server_config,
        run_dir=bench_dir / "server",
        warmup=args.warmup,
    )

    results = {
        "config": {
            "config_local": args.config_local,
            "config_server": args.config_server,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": seed,
            "device_override": args.device,
        },
        "backends": {
            "local": local_result,
            "server": server_result,
        },
        "speedup": _speedups(local_result["mean"], server_result["mean"]),
    }
    results_path = bench_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    _print_backend_table(results)
    _print_projection(results)
    print()
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
