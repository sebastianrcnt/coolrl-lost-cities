from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bots import available_bot_names, build_bot
from .game import GameState, LostCitiesConfig, classic_config
from .policy import LostCitiesPolicy

PolicyFactory = Callable[[int | None], LostCitiesPolicy]
MATCH_EVAL_RECORD_TYPE = "lost_cities.classic.eval.match.v1"


@dataclass(frozen=True)
class GameResult:
    score0: int
    score1: int
    score_diff0: int
    steps: int
    timed_out: bool


@dataclass(frozen=True)
class MatchResult:
    games: int
    wins0: int
    wins1: int
    draws: int
    avg_score0: float
    avg_score1: float
    avg_score_diff0: float
    avg_game_length: float
    max_step_timeouts: int
    elapsed_seconds: float
    games_per_second: float
    steps_per_second: float

    @property
    def win_rate0(self) -> float:
        return self.wins0 / max(1, self.games)

    @property
    def win_rate1(self) -> float:
        return self.wins1 / max(1, self.games)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "wins0": self.wins0,
            "wins1": self.wins1,
            "draws": self.draws,
            "win_rate0": self.win_rate0,
            "win_rate1": self.win_rate1,
            "avg_score0": self.avg_score0,
            "avg_score1": self.avg_score1,
            "avg_score_diff0": self.avg_score_diff0,
            "avg_game_length": self.avg_game_length,
            "max_step_timeouts": self.max_step_timeouts,
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_second": self.games_per_second,
            "steps_per_second": self.steps_per_second,
        }

    def result_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "wins0": self.wins0,
            "wins1": self.wins1,
            "draws": self.draws,
            "win_rate0": self.win_rate0,
            "win_rate1": self.win_rate1,
            "avg_score0": self.avg_score0,
            "avg_score1": self.avg_score1,
            "avg_score_diff0": self.avg_score_diff0,
            "avg_game_length": self.avg_game_length,
            "max_step_timeouts": self.max_step_timeouts,
        }

    def timing(self) -> TimingResult:
        return TimingResult(
            elapsed_seconds=self.elapsed_seconds,
            games_per_second=self.games_per_second,
            steps_per_second=self.steps_per_second,
        )


@dataclass(frozen=True)
class TimingResult:
    elapsed_seconds: float
    games_per_second: float
    steps_per_second: float

    def to_dict(self) -> dict[str, float]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_second": self.games_per_second,
            "steps_per_second": self.steps_per_second,
        }


@dataclass(frozen=True)
class MatchEvalRecord:
    bot0: str
    bot1: str
    config: LostCitiesConfig
    seed: int
    alternate_seats: bool
    max_steps: int
    result: MatchResult
    record_type: str = MATCH_EVAL_RECORD_TYPE

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.record_type,
            "bots": {
                "bot0": self.bot0,
                "bot1": self.bot1,
            },
            "settings": {
                "games": self.result.games,
                "seed": self.seed,
                "alternate_seats": self.alternate_seats,
                "max_steps": self.max_steps,
            },
            "config": self.config.to_snapshot(),
            "result": self.result.result_dict(),
            "timing": self.result.timing().to_dict(),
        }


def make_policy_factory(name: str) -> PolicyFactory:
    canonical = _canonical_bot_name(name)

    def factory(seed: int | None = None) -> LostCitiesPolicy:
        return build_bot(canonical, seed=seed)

    return factory


def play_game_for_evaluation(
    policy0: LostCitiesPolicy,
    policy1: LostCitiesPolicy,
    config: LostCitiesConfig,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> tuple[GameState, GameResult]:
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    state = GameState.new_game(config, seed=seed)
    policies = [policy0, policy1]
    steps = 0
    for _ in range(max_steps):
        if state.terminal:
            break
        action = policies[state.current_player].act(state)
        state.apply_action(action)
        steps += 1
    timed_out = not state.terminal
    if timed_out:
        steps = max_steps
    score0 = state.total_score(0)
    score1 = state.total_score(1)
    return (
        state,
        GameResult(
            score0=score0,
            score1=score1,
            score_diff0=score0 - score1,
            steps=steps,
            timed_out=timed_out,
        ),
    )


def play_match(
    policy0_factory: PolicyFactory,
    policy1_factory: PolicyFactory,
    config: LostCitiesConfig,
    *,
    games: int,
    seed: int = 1,
    max_steps: int = 10_000,
    alternate_seats: bool = True,
) -> MatchResult:
    if games <= 0:
        raise ValueError(f"games must be positive, got {games}")

    score0: list[int] = []
    score1: list[int] = []
    diffs0: list[int] = []
    lengths: list[int] = []
    wins0 = wins1 = draws = timeouts = 0

    started = time.perf_counter()
    for index in range(games):
        game_seed = seed + index
        swap = alternate_seats and index % 2 == 1
        if swap:
            left = policy1_factory(game_seed * 2)
            right = policy0_factory(game_seed * 2 + 1)
        else:
            left = policy0_factory(game_seed * 2)
            right = policy1_factory(game_seed * 2 + 1)

        _, result = play_game_for_evaluation(
            left,
            right,
            config,
            seed=game_seed,
            max_steps=max_steps,
        )
        if swap:
            bot0_score = result.score1
            bot1_score = result.score0
            diff0 = -result.score_diff0
        else:
            bot0_score = result.score0
            bot1_score = result.score1
            diff0 = result.score_diff0

        score0.append(bot0_score)
        score1.append(bot1_score)
        diffs0.append(diff0)
        lengths.append(result.steps)
        timeouts += int(result.timed_out)
        if diff0 > 0:
            wins0 += 1
        elif diff0 < 0:
            wins1 += 1
        else:
            draws += 1

    elapsed = time.perf_counter() - started
    total_steps = sum(lengths)
    return MatchResult(
        games=games,
        wins0=wins0,
        wins1=wins1,
        draws=draws,
        avg_score0=_mean(score0),
        avg_score1=_mean(score1),
        avg_score_diff0=_mean(diffs0),
        avg_game_length=_mean(lengths),
        max_step_timeouts=timeouts,
        elapsed_seconds=elapsed,
        games_per_second=games / max(elapsed, 1.0e-12),
        steps_per_second=total_steps / max(elapsed, 1.0e-12),
    )


def evaluate_policy(
    policy_factory: PolicyFactory,
    opponent_factory: PolicyFactory,
    config: LostCitiesConfig,
    *,
    games: int,
    seed: int = 1,
    max_steps: int = 10_000,
) -> MatchResult:
    return play_match(
        policy_factory,
        opponent_factory,
        config,
        games=games,
        seed=seed,
        max_steps=max_steps,
        alternate_seats=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lost Cities classic bots.")
    parser.add_argument("--bot0", default="heuristic-balanced", choices=available_bot_names())
    parser.add_argument("--bot1", default="random", choices=available_bot_names())
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--no-alternate-seats", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = classic_config()
    alternate_seats = not args.no_alternate_seats
    result = play_match(
        make_policy_factory(args.bot0),
        make_policy_factory(args.bot1),
        config,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        alternate_seats=alternate_seats,
    )
    record = MatchEvalRecord(
        bot0=args.bot0,
        bot1=args.bot1,
        config=config,
        seed=args.seed,
        alternate_seats=alternate_seats,
        max_steps=args.max_steps,
        result=result,
    )

    if args.json:
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
        return

    print(f"{args.bot0} vs {args.bot1}: {result.games} games")
    print(
        "wins/losses/draws: "
        f"{result.wins0}/{result.wins1}/{result.draws} "
        f"(win_rate0={result.win_rate0:.3f})"
    )
    print(
        f"avg_diff0={result.avg_score_diff0:.2f} "
        f"avg_score0={result.avg_score0:.2f} "
        f"avg_score1={result.avg_score1:.2f} "
        f"avg_len={result.avg_game_length:.1f}"
    )
    if args.benchmark:
        print(
            f"elapsed={result.elapsed_seconds:.3f}s "
            f"games/sec={result.games_per_second:.1f} "
            f"steps/sec={result.steps_per_second:.1f}"
        )
    if result.max_step_timeouts:
        print(f"max_step_timeouts={result.max_step_timeouts}")


def _canonical_bot_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _mean(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0
