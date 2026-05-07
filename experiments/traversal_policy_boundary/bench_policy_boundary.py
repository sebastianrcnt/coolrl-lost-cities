from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.deep_cfr.traversal import run_cython_traversal_batch
from coolrl_lost_cities.games.classic.game import GameState

from coolrl_lost_cities.games.classic.deep_cfr.config import load_config
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP


def _timer(device: torch.device | None = None) -> float:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def _build_networks(cfg: Any, input_dim_value: int, action_size: int, device: torch.device):
    networks = [
        DeepCFRMLP.from_config(input_dim_value, action_size, cfg.network).to(device).eval()
        for _ in range(2)
    ]
    strategy = DeepCFRMLP.from_config(input_dim_value, action_size, cfg.network).to(device).eval()
    return networks, strategy


def _make_corpus(cfg: Any, count: int, seed: int) -> list[GameState]:
    rng = random.Random(seed)
    game_config = cfg.rules.to_lost_cities_config(seed=seed)
    states: list[GameState] = []
    state = GameState.new_game(game_config, seed=seed)
    while len(states) < count:
        if state.terminal:
            state = GameState.new_game(game_config, seed=rng.randrange(1, 2**31))
            continue
        states.append(state.clone())
        legal = state.unified_legal_actions()
        if not legal:
            state = GameState.new_game(game_config, seed=rng.randrange(1, 2**31))
            continue
        state.push_unified_action(int(legal[rng.randrange(len(legal))]))
    return states


def _regret_matching(values: np.ndarray, legal: list[int], epsilon: float) -> np.ndarray:
    policy = np.zeros_like(values, dtype=np.float32)
    positives = [max(float(values[action]), 0.0) for action in legal]
    positive_sum = sum(positives)
    if positive_sum <= epsilon:
        uniform = 1.0 / max(1, len(legal))
        for action in legal:
            policy[action] = uniform
        return policy
    for action, positive in zip(legal, positives, strict=True):
        policy[action] = positive / positive_sum
    return policy


def bench_full_traversal(
    cfg: Any,
    *,
    traversals: int,
    runs: int,
    warmup: int,
    device: torch.device,
    input_dim_value: int,
    action_size: int,
) -> dict[str, Any]:
    timings: list[float] = []
    stat_rows: list[dict[str, Any]] = []
    seed_base = int(cfg.run.seed)
    game_config = cfg.rules.to_lost_cities_config(seed=seed_base)

    for run_idx in range(runs + warmup):
        networks, strategy = _build_networks(cfg, input_dim_value, action_size, device)
        seeds = [seed_base + run_idx * 100_000 + idx for idx in range(traversals)]
        start = _timer(device)
        stats, advantage_samples, strategy_samples = run_cython_traversal_batch(
            networks,
            game_config,
            seeds,
            0,
            1,
            device=device,
            action_size=action_size,
            strategy_network=strategy,
            encoding=cfg.encoding,
            epsilon=cfg.traversal.regret_matching_epsilon,
            strategy_sample_interval=cfg.traversal.strategy_sample_interval,
            store_strategy_on_traverser_nodes=cfg.traversal.store_strategy_on_traverser_nodes,
            store_strategy_on_opponent_nodes=cfg.traversal.store_strategy_on_opponent_nodes,
            max_depth=cfg.traversal.max_depth,
            max_nodes=cfg.traversal.max_nodes_per_traversal,
            sampling_mode=cfg.traversal.sampling_mode,
            outcome_sampling_epsilon=cfg.traversal.outcome_sampling_epsilon,
            outcome_sampling_value_clip=cfg.traversal.outcome_sampling_value_clip,
            outcome_unsampled_regret=cfg.traversal.outcome_unsampled_regret,
            cutoff_value_mode=cfg.traversal.cutoff_value_mode,
            cutoff_rollouts=cfg.traversal.cutoff_rollouts,
            cutoff_rollout_policy=cfg.traversal.cutoff_rollout_policy,
            cutoff_rollout_max_steps=cfg.traversal.cutoff_rollout_max_steps,
            opponent_policy=cfg.traversal.opponent_policy,
            all_negative_fallback=cfg.regret_matching.all_negative_fallback,
            league_advantage_networks=None,
            self_play_anchor_probability=cfg.self_play.anchor_probability,
            self_play_current_weight=cfg.self_play.current_weight,
            self_play_recent_weight=cfg.self_play.recent_weight,
            self_play_older_weight=cfg.self_play.older_weight,
            self_play_anchor_weight=cfg.self_play.anchor_weight,
            self_play_recent_window=cfg.self_play.recent_window,
            endpoint_depth_bucket_width=cfg.traversal.endpoint_depth_bucket_width,
            endpoint_depth_bucket_max=cfg.traversal.endpoint_depth_bucket_max,
            seed=seed_base + run_idx,
        )
        elapsed = _timer(device) - start
        if run_idx >= warmup:
            stats_dict = stats.to_dict()
            stats_dict["advantage_samples_returned"] = len(advantage_samples)
            stats_dict["strategy_samples_returned"] = len(strategy_samples)
            timings.append(elapsed)
            stat_rows.append(stats_dict)
        del networks, strategy

    policy_calls = [int(row["traversal_regret_matching_decisions"]) for row in stat_rows]
    nodes = [int(row["traversal_nodes"]) for row in stat_rows]
    return {
        "seconds": timings,
        "median_seconds": _median(timings),
        "median_nodes": _median([float(value) for value in nodes]),
        "median_policy_calls": _median([float(value) for value in policy_calls]),
        "median_us_per_node": _median(
            [sec * 1_000_000.0 / max(1, node) for sec, node in zip(timings, nodes, strict=True)]
        ),
        "median_us_per_policy_call": _median(
            [
                sec * 1_000_000.0 / max(1, calls)
                for sec, calls in zip(timings, policy_calls, strict=True)
            ]
        ),
        "stats": stat_rows,
    }


def bench_encode_legal(corpus: list[GameState], cfg: Any, repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    counts: list[int] = []
    total = 0.0
    for _ in range(repeats):
        for state in corpus:
            start = time.perf_counter()
            _ = encode_info_state(state, state.current_player, cfg.encoding)
            legal = state.unified_legal_actions()
            total += len(legal)
            durations.append(time.perf_counter() - start)
            counts.append(len(legal))
    return {
        "median_us": _median(durations) * 1_000_000.0,
        "p95_us": _quantile(durations, 0.95) * 1_000_000.0,
        "mean_legal_actions": float(statistics.mean(counts)),
        "checksum": total,
    }


def bench_push_pop(corpus: list[GameState], repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    total = 0
    for _ in range(repeats):
        for state in corpus:
            legal = state.unified_legal_actions()
            if not legal:
                continue
            action = int(legal[total % len(legal)])
            start = time.perf_counter()
            state.push_unified_action(action)
            state.pop_action()
            durations.append(time.perf_counter() - start)
            total += action
    return {
        "median_us": _median(durations) * 1_000_000.0,
        "p95_us": _quantile(durations, 0.95) * 1_000_000.0,
        "checksum": total,
    }


def bench_policy_boundary(
    corpus: list[GameState],
    cfg: Any,
    network: torch.nn.Module,
    device: torch.device,
    action_size: int,
    repeats: int,
) -> dict[str, Any]:
    durations: list[float] = []
    checksum = 0.0
    with torch.inference_mode():
        for _ in range(repeats):
            for state in corpus:
                start = _timer(device)
                info_state = encode_info_state(state, state.current_player, cfg.encoding)
                legal = state.unified_legal_actions()
                x = torch.as_tensor(info_state, dtype=torch.float32, device=device).unsqueeze(0)
                advantages = network(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
                policy = _regret_matching(
                    advantages,
                    legal,
                    float(cfg.traversal.regret_matching_epsilon),
                )
                checksum += float(policy.sum()) + float(advantages[0])
                durations.append(_timer(device) - start)
    return {
        "median_us": _median(durations) * 1_000_000.0,
        "p95_us": _quantile(durations, 0.95) * 1_000_000.0,
        "checksum": checksum,
    }


def bench_torch_forward(
    corpus: list[GameState],
    cfg: Any,
    network: torch.nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    repeats: int,
) -> dict[str, Any]:
    states = np.stack(
        [encode_info_state(state, state.current_player, cfg.encoding) for state in corpus]
    ).astype(np.float32)
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for batch_size in batch_sizes:
            durations: list[float] = []
            checksum = 0.0
            for _ in range(repeats):
                for offset in range(0, len(states), batch_size):
                    chunk = states[offset : offset + batch_size]
                    if len(chunk) != batch_size:
                        continue
                    x = torch.as_tensor(chunk, dtype=torch.float32, device=device)
                    start = _timer(device)
                    out = network(x)
                    checksum += float(out.detach().sum().cpu())
                    durations.append((_timer(device) - start) / batch_size)
            if not durations:
                continue
            results[str(batch_size)] = {
                "median_us_per_call": _median(durations) * 1_000_000.0,
                "p95_us_per_call": _quantile(durations, 0.95) * 1_000_000.0,
                "checksum": checksum,
            }
    return results


def _print_table(result: dict[str, Any]) -> None:
    full = result["full_traversal"]
    encode = result["encode_legal"]
    push_pop = result["push_pop"]
    boundary = result["policy_boundary_bs1"]
    forward = result["torch_forward"]

    print("Traversal policy-boundary microbench")
    print(
        f"config={result['config']} device={result['device']} torch_threads={result['torch_threads']}"
    )
    print(
        f"full traversal: {full['median_seconds']:.3f}s, "
        f"policy_calls={full['median_policy_calls']:.0f}, nodes={full['median_nodes']:.0f}"
    )
    print()
    print("component                 median us/call   p95 us/call")
    print(
        f"encode+legal                     {encode['median_us']:9.2f}     {encode['p95_us']:9.2f}"
    )
    print(
        f"push+pop                         {push_pop['median_us']:9.2f}     {push_pop['p95_us']:9.2f}"
    )
    print(
        f"policy boundary bs=1             {boundary['median_us']:9.2f}     {boundary['p95_us']:9.2f}"
    )
    for batch_size, row in forward.items():
        print(
            f"torch forward bs={batch_size:<3}            "
            f"{row['median_us_per_call']:9.2f}     {row['p95_us_per_call']:9.2f}"
        )
    print()
    print("derived")
    print(f"full traversal us/node              {full['median_us_per_node']:.2f}")
    print(f"full traversal us/policy_call       {full['median_us_per_policy_call']:.2f}")
    if boundary["median_us"] > 0:
        print(
            "full/policy-boundary ratio          "
            f"{full['median_us_per_policy_call'] / boundary['median_us']:.2f}x"
        )
    bs64 = forward.get("64")
    if bs64 and bs64["median_us_per_call"] > 0:
        print(
            "policy bs=1 vs forward bs=64        "
            f"{boundary['median_us'] / bs64['median_us_per_call']:.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deep_cfr/default.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--traversals", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--corpus-size", type=int, default=512)
    parser.add_argument("--component-repeats", type=int, default=4)
    parser.add_argument("--forward-repeats", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    cfg = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    probe = GameState.new_game(
        cfg.rules.to_lost_cities_config(seed=cfg.run.seed), seed=cfg.run.seed
    )
    input_dim_value = input_dim(probe, cfg.encoding)
    action_size = 2 * probe.config.hand_size + 1 + probe.config.n_colors
    networks, _strategy = _build_networks(cfg, input_dim_value, action_size, device)
    corpus = _make_corpus(cfg, args.corpus_size, int(cfg.run.seed))

    # Run a tiny warmup through the component path before timed loops.
    _ = encode_info_state(corpus[0], corpus[0].current_player, cfg.encoding)
    with torch.inference_mode():
        _ = networks[0](torch.as_tensor(_, dtype=torch.float32, device=device).unsqueeze(0))

    result = {
        "config": args.config,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "traversals": args.traversals,
        "runs": args.runs,
        "warmup": args.warmup,
        "corpus_size": args.corpus_size,
        "input_dim": input_dim_value,
        "action_size": action_size,
        "full_traversal": bench_full_traversal(
            cfg,
            traversals=args.traversals,
            runs=args.runs,
            warmup=args.warmup,
            device=device,
            input_dim_value=input_dim_value,
            action_size=action_size,
        ),
        "encode_legal": bench_encode_legal(corpus, cfg, args.component_repeats),
        "push_pop": bench_push_pop(corpus, args.component_repeats),
        "policy_boundary_bs1": bench_policy_boundary(
            corpus,
            cfg,
            networks[0],
            device,
            action_size,
            args.component_repeats,
        ),
        "torch_forward": bench_torch_forward(
            corpus,
            cfg,
            networks[0],
            device,
            [1, 4, 8, 64, 256],
            args.forward_repeats,
        ),
    }
    _print_table(result)

    output = Path(args.output) if args.output else Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
