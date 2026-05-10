from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from coolrl_lost_cities.games.classic.deep_cfr.tracking import WandbRunTracker

from .config import IsMctsConfig, load_config
from .trainer import IsMctsTrainer

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _kebab_slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-") or "run"


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _set_path_override(overrides: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"config override must be PATH=VALUE: {assignment}")
    path, raw_value = assignment.split("=", 1)
    keys = path.split(".")
    value = yaml.safe_load(raw_value)
    cursor = overrides
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _with_overrides(config: IsMctsConfig, assignments: list[str]) -> IsMctsConfig:
    overrides: dict[str, Any] = {}
    for assignment in assignments:
        _set_path_override(overrides, assignment)
    data = config.model_dump(mode="python")
    _deep_update(data, overrides)
    return IsMctsConfig.model_validate(data)


def _resolve_run_dir(config: IsMctsConfig, *, keep: bool) -> Path:
    parent = Path("runs") if keep else Path("runs/tmp")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return parent / f"{timestamp}_{_kebab_slug(config.run.experiment_name)}"


def train_command(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else IsMctsConfig()
    config = _with_overrides(config, args.config_overrides)
    run_dir = _resolve_run_dir(config, keep=args.keep)
    tracker = None
    if args.wandb:
        tracker = WandbRunTracker(
            project=args.wandb_project,
            name=args.wandb_name or config.run.experiment_name,
            mode=args.wandb_mode,
            run_dir=run_dir,
            config=config.to_dict() if hasattr(config, "to_dict") else config.model_dump(),
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            tags=list(args.wandb_tag) if args.wandb_tag else None,
            notes=args.wandb_notes,
        )
    trainer = IsMctsTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        run_dir=run_dir,
        device=config.run.device,
        tracker=tracker,
    )
    try:
        trainer.train()
    finally:
        if tracker is not None:
            tracker.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lost Cities SO-ISMCTS tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--keep", action="store_true")
    train.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="PATH=VALUE",
    )
    train.add_argument(
        "--wandb",
        action="store_true",
        help="Mirror metrics to Weights & Biases (requires wandb extra).",
    )
    train.add_argument("--wandb-project", default="coolrl-lost-cities")
    train.add_argument("--wandb-name")
    train.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    train.add_argument("--wandb-group")
    train.add_argument("--wandb-job-type")
    train.add_argument("--wandb-tag", action="append", default=[])
    train.add_argument("--wandb-notes")
    train.set_defaults(func=train_command)

    from .eval_checkpoint import add_eval_args, run_eval

    eval_cmd = subparsers.add_parser(
        "eval",
        help="Evaluate a saved checkpoint vs heuristic bots in parallel.",
    )
    add_eval_args(eval_cmd)
    eval_cmd.set_defaults(func=lambda a: run_eval(a))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
