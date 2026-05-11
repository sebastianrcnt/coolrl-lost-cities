"""Standalone parallel evaluation of an ISMCTS checkpoint vs heuristic bots.

Reuses the same MCTS / bot stack as the training-loop eval, but does not
interfere with a running training process. Useful for comparing a snapshot
against opponents that are not in `evaluation.opponents` (e.g. heuristic-balanced,
the rollout policy) and for running many more games than per-iter eval typically
allows.

Invoked via ``lost-cities-ismcts eval`` (see ``cli.py``).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class _WorkerJob:
    ckpt_path: str
    opponent: str
    game_indices: tuple[int, ...]
    seed: int
    device: str
    worker_index: int
    verbose: bool
    n_sims_override: int = 0  # 0 means use checkpoint's eval_n_simulations


@dataclass
class _GameResult:
    game_index: int
    policy_player: int
    score_diff: float
    turns: int
    policy_turns: int
    play_actions: int
    timed_out: bool


@dataclass
class _WorkerResult:
    worker_index: int
    opponent: str
    games: list[_GameResult] = field(default_factory=list)
    elapsed: float = 0.0


def _run_games(job: _WorkerJob) -> _WorkerResult:
    # Inside-worker imports + thread-cap to avoid CPU oversubscription
    from coolrl_lost_cities.games.classic.bots.registry import build_bot
    from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
    from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig
    from coolrl_lost_cities.games.classic.ismcts.config import IsMctsConfig
    from coolrl_lost_cities.games.classic.ismcts.mcts import IsMctsSearcher
    from coolrl_lost_cities.games.classic.ismcts.network import AlphaZeroNet

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    started = time.perf_counter()
    print(
        f"  [worker {job.worker_index}] start ({len(job.game_indices)} games "
        f"vs {job.opponent}, device={job.device})",
        flush=True,
    )

    ckpt = torch.load(job.ckpt_path, map_location="cpu", weights_only=False)
    cfg = IsMctsConfig.model_validate(ckpt["config"])
    game_config = LostCitiesConfig(**ckpt["game_config"])
    probe = GameState.new_game(game_config, seed=cfg.run.seed)
    dim = input_dim(probe, cfg.encoding)
    device = torch.device(job.device)
    net = AlphaZeroNet.from_config(dim, probe.action_size, cfg).to(device)
    net.load_state_dict(ckpt["network"])
    net.eval()

    eval_mcts_cfg = cfg.mcts.model_copy()
    if job.n_sims_override > 0:
        eval_mcts_cfg = eval_mcts_cfg.model_copy(update={"n_simulations": job.n_sims_override})
    elif cfg.mcts.eval_n_simulations > 0:
        eval_mcts_cfg = eval_mcts_cfg.model_copy(
            update={"n_simulations": cfg.mcts.eval_n_simulations}
        )

    rng = random.Random(job.seed + job.worker_index * 7919)
    result = _WorkerResult(worker_index=job.worker_index, opponent=job.opponent)

    max_steps = 500
    for game_index in job.game_indices:
        game_started = time.perf_counter()
        policy_player = game_index % 2
        opps = [
            build_bot(job.opponent, seed=job.seed + game_index),
            build_bot(job.opponent, seed=job.seed + game_index + 1),
        ]
        state = GameState.new_game(game_config, seed=job.seed + game_index)
        turns = 0
        policy_turns = 0
        play_actions = 0
        timed_out = False
        while True:
            if state.terminal:
                break
            if turns >= max_steps:
                timed_out = True
                break
            current = int(state.current_player)
            if current == policy_player:
                searcher = IsMctsSearcher(
                    net,
                    eval_mcts_cfg,
                    device=device,
                    encoding=cfg.encoding,
                    rng=random.Random(rng.randrange(2**31)),
                )
                visits = searcher.search(state, current)
                unified = (
                    max(visits, key=visits.get) if visits else state.unified_legal_actions()[0]
                )
                if state.phase == "card":
                    policy_turns += 1
                    if unified % 2 == 0:
                        play_actions += 1
                state.apply_unified_action(unified)
            else:
                state.apply_action(opps[current].act(state))
            turns += 1

        diff = float(state.score_diff(policy_player))
        gr = _GameResult(
            game_index=game_index,
            policy_player=policy_player,
            score_diff=diff,
            turns=turns,
            policy_turns=policy_turns,
            play_actions=play_actions,
            timed_out=timed_out,
        )
        result.games.append(gr)
        if job.verbose:
            elapsed_g = time.perf_counter() - game_started
            pa = play_actions / policy_turns if policy_turns else 0.0
            print(
                f"  [worker {job.worker_index}] game {game_index:3d} "
                f"as P{policy_player} | turns={turns:3d} "
                f"score={diff:+6.1f} PA={pa:.2f}"
                f"{' TIMEOUT' if timed_out else ''} "
                f"({elapsed_g:.1f}s)",
                flush=True,
            )

    result.elapsed = time.perf_counter() - started
    won = sum(1 for g in result.games if g.score_diff > 0)
    print(
        f"  [worker {job.worker_index}] done {len(result.games)} games "
        f"vs {job.opponent} in {result.elapsed:.1f}s "
        f"(W={won}/{len(result.games)})",
        flush=True,
    )
    return result


def _split_games(n_games: int, n_workers: int) -> list[tuple[int, ...]]:
    n_workers = max(1, min(n_workers, n_games))
    base = n_games // n_workers
    rem = n_games % n_workers
    out: list[tuple[int, ...]] = []
    cursor = 0
    for i in range(n_workers):
        count = base + (1 if i < rem else 0)
        out.append(tuple(range(cursor, cursor + count)))
        cursor += count
    return out


def _summarize(games: list[_GameResult]) -> dict[str, float]:
    n = len(games)
    if n == 0:
        return {}
    wins = sum(1 for g in games if g.score_diff > 0)
    losses = sum(1 for g in games if g.score_diff < 0)
    draws = sum(1 for g in games if g.score_diff == 0)
    timeouts = sum(1 for g in games if g.timed_out)
    score_diffs = [g.score_diff for g in games]
    avg = sum(score_diffs) / n
    var = sum((d - avg) ** 2 for d in score_diffs) / n if n > 1 else 0.0
    std = var**0.5
    total_policy_turns = sum(g.policy_turns for g in games)
    total_play_actions = sum(g.play_actions for g in games)
    pa = total_play_actions / total_policy_turns if total_policy_turns else 0.0
    avg_turns = sum(g.turns for g in games) / n
    z = 1.96
    # Wilson CI on win rate (half-width only; report as wr ± half)
    wr = wins / n
    denom = 1 + z * z / n
    half = z / denom * ((wr * (1 - wr) / n + z * z / (4 * n * n)) ** 0.5)
    score_ci = z * std / (n**0.5) if n > 1 else 0.0
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": wr,
        "win_rate_ci_half": half,
        "avg_score_diff": avg,
        "score_std": std,
        "score_ci_half": score_ci,
        "play_action_rate": pa,
        "avg_turns": avg_turns,
    }


def _default_ckpt() -> Path:
    """Find latest 'ismcts-overnight*' run's latest.pt, or fallback to newest run."""
    candidates = sorted(Path("runs").glob("*ismcts-overnight*"))
    if candidates:
        return candidates[-1] / "latest.pt"
    candidates = sorted(Path("runs").iterdir())
    if not candidates:
        raise SystemExit("no runs/ directory entries")
    return candidates[-1] / "latest.pt"


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint .pt. Default: latest overnight run latest.pt.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["heuristic-balanced", "heuristic-aggressive", "heuristic-cautious"],
        help="Opponent bot names from registry.",
    )
    parser.add_argument("--games", type=int, default=50, help="Games per opponent (default: 50).")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Worker device (default: cpu; cuda may compete with running training).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Parallel worker count (default: 8)."
    )
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-game result lines (turns/score/PA) in addition to per-worker summaries.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=0,
        help="Override n_simulations at eval time (0 = use checkpoint's eval_n_simulations).",
    )


def run_eval(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt()
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        raise SystemExit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    iteration = ckpt.get("iteration", "?")
    print(f"checkpoint : {ckpt_path}")
    print(f"iteration  : {iteration}")
    print(f"device     : {args.device}")
    print(f"workers    : {args.num_workers}")
    print(f"games/opp  : {args.games}")
    print(f"opponents  : {args.opponents}")
    print(f"seed       : {args.seed}")
    if args.verbose:
        print("verbose    : True (per-game logging)")
    print()

    ctx = mp.get_context("spawn")
    started_all = time.perf_counter()

    for opponent in args.opponents:
        opp_started = time.perf_counter()
        slices = _split_games(args.games, args.num_workers)
        jobs = [
            _WorkerJob(
                ckpt_path=str(ckpt_path),
                opponent=opponent,
                game_indices=tuple(slices[i]),
                seed=args.seed,
                device=args.device,
                worker_index=i,
                verbose=args.verbose,
                n_sims_override=int(args.n_sims),
            )
            for i in range(len(slices))
        ]
        all_games: list[_GameResult] = []
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as ex:
            futures = [ex.submit(_run_games, j) for j in jobs]
            for f in as_completed(futures):
                res = f.result()
                all_games.extend(res.games)

        elapsed = time.perf_counter() - opp_started
        summary = _summarize(all_games)
        n = int(summary["games"])
        print()
        print(
            f"vs {opponent:22s} | W={summary['wins']}/{n} "
            f"({summary['win_rate']:.2f} ± {summary['win_rate_ci_half']:.2f}) "
            f"| S={summary['avg_score_diff']:+6.1f} ± {summary['score_ci_half']:5.1f} "
            f"(σ={summary['score_std']:.1f}) | PA={summary['play_action_rate']:.2f} "
            f"| turns={summary['avg_turns']:.0f} "
            f"| timeouts={summary['timeouts']} "
            f"| elapsed={elapsed:.1f}s"
        )
        print()

    total = time.perf_counter() - started_all
    print(f"total elapsed: {total:.1f}s")
