from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from coolrl_lost_cities.games.classic.deep_cfr.trainer import DeepCFRTrainer
from coolrl_lost_cities.games.classic.game import classic_config

_RESUME_LATEST = "__latest__"


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


def _resolve_resume_path(config: DeepCFRConfig, resume: str | None) -> str | None:
    if resume != _RESUME_LATEST:
        return resume
    latest_path = config.checkpoint_path / "latest.pt"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"--resume was used without a path, but latest checkpoint does not exist: {latest_path}"
        )
    return str(latest_path)


def _train_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.no_save and (args.save_latest_only or args.save_iteration_interval is not None):
        raise ValueError("--no-save cannot be combined with checkpoint save overrides")
    run_overrides = overrides.setdefault("run", {})
    if args.iterations is not None:
        run_overrides["iterations"] = args.iterations
        if args.max_hours is None and args.max_iterations is None:
            run_overrides["max_hours"] = None
            run_overrides["max_iterations"] = None
    if args.max_hours is not None:
        run_overrides["max_hours"] = args.max_hours
    if args.max_iterations is not None:
        run_overrides["max_iterations"] = args.max_iterations
    if args.seed is not None:
        run_overrides["seed"] = args.seed
    if args.traversals_per_iteration is not None:
        traversal_overrides = overrides.setdefault("traversal", {})
        traversal_overrides["traversals_per_iteration"] = args.traversals_per_iteration
        traversal_overrides["traversals_per_player"] = None
    if args.num_workers is not None:
        overrides.setdefault("traversal", {})["num_workers"] = args.num_workers
    if args.checkpoint_dir is not None:
        overrides.setdefault("checkpoint", {})["directory"] = args.checkpoint_dir
    if args.eval_every is not None:
        overrides.setdefault("evaluation", {})["eval_every"] = args.eval_every
    if args.eval_games is not None:
        overrides.setdefault("evaluation", {})["games"] = args.eval_games
    if args.regret_fallback is not None:
        overrides.setdefault("regret_matching", {})["all_negative_fallback"] = args.regret_fallback
    if args.training_weighting is not None:
        overrides.setdefault("training_weighting", {})["mode"] = args.training_weighting
    if args.no_save:
        checkpoint_overrides = overrides.setdefault("checkpoint", {})
        checkpoint_overrides["save_latest"] = False
        checkpoint_overrides["save_every_iteration"] = False
        checkpoint_overrides["save_iteration_interval"] = 0
    if args.save_latest_only:
        checkpoint_overrides = overrides.setdefault("checkpoint", {})
        checkpoint_overrides["save_latest"] = True
        checkpoint_overrides["save_latest_only"] = True
        checkpoint_overrides["save_every_iteration"] = False
    if args.save_iteration_interval is not None:
        overrides.setdefault("checkpoint", {})["save_iteration_interval"] = (
            args.save_iteration_interval
        )
    if args.exact_resume:
        overrides.setdefault("checkpoint", {})["exact_resume"] = True
    return overrides


def train_command(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    overrides = _train_overrides_from_args(args)
    config = _with_overrides(config, overrides)
    resume_path = _resolve_resume_path(config, args.resume)
    trainer = DeepCFRTrainer(
        config,
        config.rules.to_lost_cities_config(seed=config.run.seed),
        device=args.device or config.run.device,
    )
    if resume_path:
        trainer.load_checkpoint(resume_path)
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
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def benchmark_command(args: argparse.Namespace) -> None:
    config = DeepCFRConfig.model_validate(
        {
            "run": {"seed": args.seed},
            "traversal": {
                "traversals_per_iteration": args.traversals,
                "max_depth": args.depth,
            },
            "checkpoint": {"save_every_iteration": False},
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
    train.add_argument("--iterations", type=int)
    train.add_argument("--max-hours", type=float)
    train.add_argument("--max-iterations", type=int)
    train.add_argument("--traversals-per-iteration", type=int)
    train.add_argument("--num-workers")
    train.add_argument("--checkpoint-dir")
    train.add_argument("--resume", nargs="?", const=_RESUME_LATEST, default=None)
    train.add_argument("--exact-resume", action="store_true")
    train.add_argument("--device")
    train.add_argument("--eval-every", type=int)
    train.add_argument("--eval-games", type=int)
    train.add_argument(
        "--regret-fallback",
        choices=("uniform", "argmax_tiebreak"),
        help="Override regret_matching.all_negative_fallback.",
    )
    train.add_argument(
        "--training-weighting",
        choices=("none", "lcfr", "dcfr"),
        help="Override training_weighting.mode.",
    )
    train.add_argument("--seed", type=int)
    train.add_argument("--no-save", action="store_true")
    train.add_argument("--save-latest-only", action="store_true")
    train.add_argument("--save-iteration-interval", type=int)
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--opponent", default="random")
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--max-steps", type=int, default=10_000)
    evaluate.add_argument("--device", default="cpu")
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
