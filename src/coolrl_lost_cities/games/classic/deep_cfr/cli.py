from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from coolrl_lost_cities.games.classic.deep_cfr.config import DeepCFRConfig, config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import (
    evaluate_strategy_network,
    load_strategy_policy_from_checkpoint,
)
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer
from coolrl_lost_cities.games.classic.game import classic_config


def _load_config(path: str | None) -> DeepCFRConfig:
    if path is None:
        return DeepCFRConfig()
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_command(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    overrides = {}
    for key in (
        "iterations",
        "traversals_per_iteration",
        "checkpoint_dir",
        "eval_every",
        "eval_games",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.no_save:
        overrides["save_every_iteration"] = False
    config = replace(config, **overrides)
    trainer = DeepCFRTrainer(config, classic_config(seed=config.seed), device=args.device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    metrics = trainer.train()
    for item in metrics:
        print(json.dumps(item.to_dict(), sort_keys=True))


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
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lost Cities classic Deep CFR tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--iterations", type=int)
    train.add_argument("--traversals-per-iteration", type=int)
    train.add_argument("--checkpoint-dir")
    train.add_argument("--resume")
    train.add_argument("--device", default="cpu")
    train.add_argument("--eval-every", type=int)
    train.add_argument("--eval-games", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--no-save", action="store_true")
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--opponent", default="random")
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--max-steps", type=int, default=10_000)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(func=eval_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
