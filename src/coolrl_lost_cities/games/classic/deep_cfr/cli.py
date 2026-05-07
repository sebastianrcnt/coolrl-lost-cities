from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from coolrl_lost_cities.games.classic.deep_cfr.analyze import analyze_run
from coolrl_lost_cities.games.classic.deep_cfr.benchmark import (
    benchmark_traversal,
    benchmark_traversal_modes,
)
from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig, load_config
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import (
    evaluate_strategy_network,
    load_strategy_policy_from_checkpoint,
)
from coolrl_lost_cities.games.classic.deep_cfr.imitation import (
    new_pretrained_strategy_network,
)
from coolrl_lost_cities.games.classic.deep_cfr.policy_gradient import (
    fine_tune_strategy_policy_gradient,
)
from coolrl_lost_cities.games.classic.deep_cfr.tracking import RunTracker, WandbRunTracker
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer
from coolrl_lost_cities.games.classic.game import classic_config

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _kebab_slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "run"


def _resolve_run_dir(config: DeepCFRConfig, *, keep: bool) -> Path:
    parent = Path("runs") if keep else Path("runs/tmp")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = _kebab_slug(config.run.experiment_name)
    return parent / f"{timestamp}_{slug}"


def _load_config(path: str | None) -> DeepCFRConfig:
    if path is None:
        return DeepCFRConfig()
    return load_config(path)


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _with_overrides(config: DeepCFRConfig, overrides: dict[str, Any]) -> DeepCFRConfig:
    data = config.model_dump(mode="python")
    _deep_update(data, overrides)
    return DeepCFRConfig.model_validate(data)


def _set_path_override(overrides: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"config override must be PATH=VALUE: {assignment}")
    path, raw_value = assignment.split("=", 1)
    keys = path.split(".")
    if any(not key for key in keys):
        raise ValueError(f"config override path must use non-empty dotted keys: {path}")
    value = yaml.safe_load(raw_value)
    cursor = overrides
    for key in keys[:-1]:
        next_cursor = cursor.setdefault(key, {})
        if not isinstance(next_cursor, dict):
            raise ValueError(f"config override path conflicts with scalar value: {path}")
        cursor = next_cursor
    cursor[keys[-1]] = value


def _train_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for assignment in getattr(args, "config_overrides", None) or ():
        _set_path_override(overrides, assignment)
    return overrides


def train_command(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    overrides = _train_overrides_from_args(args)
    config = _with_overrides(config, overrides)
    if args.resume:
        run_dir = Path(args.resume).parent
    else:
        run_dir = _resolve_run_dir(config, keep=args.keep)
    extra_trackers: list[RunTracker] = []
    if args.wandb:
        extra_trackers.append(
            WandbRunTracker(
                project=args.wandb_project,
                name=args.wandb_name or config.run.experiment_name,
                mode=args.wandb_mode,
                config=config.to_dict(),
                run_dir=str(run_dir),
                tags=list(args.wandb_tag) if args.wandb_tag else None,
                notes=args.wandb_notes,
            )
        )
    trainer = DeepCFRTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        run_dir=run_dir,
        device=config.run.device,
        extra_trackers=extra_trackers or None,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


def eval_command(args: argparse.Namespace) -> None:
    policy, game_config = load_strategy_policy_from_checkpoint(args.checkpoint, device=args.device)
    result = evaluate_strategy_network(
        policy.strategy_network,
        game_config,
        games=args.games,
        seed=args.seed,
        opponent=args.opponent,
        device=args.device,
        max_steps=args.max_steps,
        encoding=policy.encoding,
        save_games_path=args.save_games,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.save_games:
        print(f"Game records saved to: {args.save_games}")


def benchmark_command(args: argparse.Namespace) -> None:
    config = DeepCFRConfig.model_validate(
        {
            "run": {"seed": args.seed},
            "traversal": {
                "traversals_per_player": args.traversals,
                "max_depth": args.depth,
            },
            "checkpoint": {"save_every": 0},
        }
    )
    if args.compare:
        print(json.dumps(benchmark_traversal_modes(config), indent=2, sort_keys=True))
        return
    result = benchmark_traversal(
        config,
        num_workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def pretrain_command(args: argparse.Namespace) -> None:
    network, metrics = new_pretrained_strategy_network(
        classic_config(seed=args.seed),
        hidden_size=args.hidden_size,
        games=args.games,
        seed=args.seed,
        steps=args.steps,
    )
    if args.output:
        import torch

        torch.save(
            {"strategy_network": network.state_dict(), "metrics": metrics.__dict__}, args.output
        )
    print(json.dumps(metrics.__dict__, sort_keys=True))


def policy_gradient_command(args: argparse.Namespace) -> None:
    policy, game_config = load_strategy_policy_from_checkpoint(args.checkpoint, device=args.device)
    metrics = fine_tune_strategy_policy_gradient(
        policy.strategy_network,
        game_config,
        episodes=args.episodes,
        seed=args.seed,
        opponent=args.opponent,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        device=args.device,
    )
    if args.output:
        import torch

        torch.save(
            {"strategy_network": policy.strategy_network.state_dict(), "metrics": metrics.__dict__},
            args.output,
        )
    print(json.dumps(metrics.__dict__, sort_keys=True))


def analyze_command(args: argparse.Namespace) -> None:
    written = analyze_run(args.run, args.output_dir, max_iteration=args.max_iteration)
    for path in written:
        print(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lost Cities classic Deep CFR tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config")
    train.add_argument(
        "--resume",
        help="Path to a checkpoint to resume from (e.g. runs/.../latest.pt).",
    )
    train.add_argument(
        "--keep",
        action="store_true",
        help="Place run under runs/ instead of runs/tmp/ (use for real experiments).",
    )
    train.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="PATH=VALUE",
        help=(
            "Override config fields using dotted paths. Repeatable. VALUE is parsed as "
            "YAML, e.g. --set traversal.num_workers=4 --set run.max_minutes=null."
        ),
    )
    train.add_argument(
        "--wandb",
        action="store_true",
        help="Mirror metrics to Weights & Biases (requires wandb extra).",
    )
    train.add_argument("--wandb-project", default="coolrl-lost-cities")
    train.add_argument(
        "--wandb-name",
        help="W&B run name. Defaults to config.run.experiment_name.",
    )
    train.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    train.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        help="Tag to attach to the W&B run (repeatable).",
    )
    train.add_argument(
        "--wandb-notes",
        help="Free-form note describing this run's purpose (shown on the W&B run page).",
    )
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--opponent", default="random")
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--max-steps", type=int, default=10_000)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--save-games", help="Path to save game records as JSON")
    evaluate.set_defaults(func=eval_command)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--workers", type=int, default=0)
    benchmark.add_argument("--traversals", type=int, default=8)
    benchmark.add_argument("--depth", type=int, default=4)
    benchmark.add_argument("--seed", type=int, default=1)
    benchmark.add_argument("--compare", action="store_true")
    benchmark.set_defaults(func=benchmark_command)

    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--games", type=int, default=4)
    pretrain.add_argument("--steps", type=int, default=32)
    pretrain.add_argument("--hidden-size", type=int, default=64)
    pretrain.add_argument("--seed", type=int, default=1)
    pretrain.add_argument("--output")
    pretrain.set_defaults(func=pretrain_command)

    pg = subparsers.add_parser("policy-gradient")
    pg.add_argument("--checkpoint", required=True)
    pg.add_argument("--episodes", type=int, default=2)
    pg.add_argument("--opponent", default="random")
    pg.add_argument("--learning-rate", type=float, default=1.0e-4)
    pg.add_argument("--max-steps", type=int, default=10_000)
    pg.add_argument("--seed", type=int, default=1)
    pg.add_argument("--device", default="cpu")
    pg.add_argument("--output")
    pg.set_defaults(func=policy_gradient_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run", required=True, type=Path)
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument(
        "--max-iteration",
        type=int,
        help="Only plot metrics up to and including this iteration.",
    )
    analyze.set_defaults(func=analyze_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
