from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.game import GameState

from coolrl_lost_cities.games.classic.deep_cfr.config import load_config
from coolrl_lost_cities.games.classic.deep_cfr.memory import TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.traversal_stats import TraversalStats


def _next_u32(state: int) -> tuple[int, int]:
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state


def _next_double(state: int) -> tuple[int, float]:
    state, value = _next_u32(state)
    return state, value / 4294967296.0


def _sample_policy(policy: np.ndarray, actions: list[int], random_value: float) -> int:
    fallback = -1
    cumulative = 0.0
    r = min(max(random_value, 0.0), 0.9999999999999999)
    for action in actions:
        if policy[action] > 0.0:
            fallback = action
            cumulative += float(policy[action])
            if r < cumulative:
                return action
    return fallback


def _regret_matching(
    advantages: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, bool, int, bool]:
    policy = np.zeros_like(advantages, dtype=np.float32)
    legal = np.flatnonzero(legal_mask)
    positives = np.maximum(advantages[legal], 0.0)
    positive_sum = float(positives.sum())
    fallback = positive_sum <= epsilon
    tie_size = 0
    full_tie = False
    if not fallback:
        policy[legal] = positives / positive_sum
        return policy, fallback, tie_size, full_tie

    if len(legal) == 0:
        return policy, fallback, tie_size, full_tie
    best = float(np.max(advantages[legal]))
    tied = legal[np.flatnonzero(advantages[legal] == best)]
    tie_size = int(len(tied))
    full_tie = tie_size > 1 and tie_size == len(legal)
    policy[legal] = 1.0 / float(len(legal))
    return policy, fallback, tie_size, full_tie


def _sampling_policy(policy: np.ndarray, legal_mask: np.ndarray, epsilon: float) -> np.ndarray:
    legal = np.flatnonzero(legal_mask)
    out = np.zeros_like(policy, dtype=np.float32)
    if len(legal) == 0:
        return out
    if epsilon <= 0.0:
        out[:] = policy
        return out
    uniform = 1.0 / float(len(legal))
    out[legal] = (1.0 - epsilon) * policy[legal] + epsilon * uniform
    return out


def _record_endpoint(stats: TraversalStats, depth: int, width: int, max_depth: int) -> None:
    stats.endpoint_depth_sum += depth
    start = (depth // width) * width
    key = f"{max_depth}_plus" if start >= max_depth else f"{start}_{start + width - 1}"
    stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1


@dataclass
class PrototypeConfig:
    action_size: int
    encoding: Any
    epsilon: float
    outcome_sampling_epsilon: float
    outcome_sampling_value_clip: float | None
    max_depth: int | None
    max_nodes: int | None
    strategy_sample_interval: int
    store_strategy_on_traverser_nodes: bool
    store_strategy_on_opponent_nodes: bool
    endpoint_depth_bucket_width: int
    endpoint_depth_bucket_max: int


@dataclass
class PolicyResult:
    info_state: np.ndarray
    legal_mask: np.ndarray
    policy: np.ndarray
    fallback: bool
    tie_size: int
    full_tie: bool


@dataclass
class PolicyRequest:
    context_index: int
    player: int
    info_state: np.ndarray
    legal_mask: np.ndarray


@dataclass
class Samples:
    advantage: list[TrainingSample] = field(default_factory=list)
    strategy: list[TrainingSample] = field(default_factory=list)


@dataclass
class AfterChildFrame:
    depth: int
    player: int
    action: int
    action_prob: float
    info_state: np.ndarray
    legal_mask: np.ndarray
    policy: np.ndarray
    sampling_policy: np.ndarray
    fallback: bool
    tie_size: int
    full_tie: bool


@dataclass
class EnterFrame:
    depth: int


Frame = EnterFrame | AfterChildFrame


class BatchedPolicy:
    def __init__(
        self,
        networks: list[torch.nn.Module],
        *,
        device: torch.device,
        epsilon: float,
    ) -> None:
        self.networks = networks
        self.device = device
        self.epsilon = epsilon
        self.batch_sizes: list[int] = []
        self.forward_seconds = 0.0

    def one(self, player: int, info_state: np.ndarray, legal_mask: np.ndarray) -> PolicyResult:
        return self.batch([PolicyRequest(-1, player, info_state, legal_mask)])[0]

    def batch(self, requests: list[PolicyRequest]) -> list[PolicyResult]:
        if not requests:
            return []
        out: list[PolicyResult | None] = [None] * len(requests)
        for player in sorted({request.player for request in requests}):
            indices = [idx for idx, request in enumerate(requests) if request.player == player]
            states = np.stack([requests[idx].info_state for idx in indices]).astype(np.float32)
            x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                values = self.networks[player](x).detach().cpu().numpy().astype(np.float32)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.forward_seconds += time.perf_counter() - start
            self.batch_sizes.append(len(indices))
            for local_idx, request_idx in enumerate(indices):
                request = requests[request_idx]
                policy, fallback, tie_size, full_tie = _regret_matching(
                    values[local_idx], request.legal_mask, self.epsilon
                )
                out[request_idx] = PolicyResult(
                    info_state=request.info_state,
                    legal_mask=request.legal_mask,
                    policy=policy,
                    fallback=fallback,
                    tie_size=tie_size,
                    full_tie=full_tie,
                )
        return [result for result in out if result is not None]


class RecursivePrototype:
    def __init__(self, cfg: PrototypeConfig, policy: BatchedPolicy) -> None:
        self.cfg = cfg
        self.policy = policy

    def traverse(
        self,
        state: GameState,
        *,
        traverser: int,
        iteration: int,
        rng: int,
    ) -> tuple[float, int, TraversalStats, Samples]:
        stats = TraversalStats()
        samples = Samples()
        value, rng = self._traverse(state, traverser, iteration, 0, rng, stats, samples)
        return value, rng, stats, samples

    def _traverse(
        self,
        state: GameState,
        traverser: int,
        iteration: int,
        depth: int,
        rng: int,
        stats: TraversalStats,
        samples: Samples,
    ) -> tuple[float, int]:
        stats.nodes += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        cutoff = self._cutoff(state, traverser, depth, stats)
        if cutoff is not None:
            return cutoff, rng

        player = int(state.current_player)
        result = self._policy_result(state, player)
        self._record_strategy(result, player, traverser, iteration, depth, stats, samples)
        actions = [int(action) for action in np.flatnonzero(result.legal_mask)]
        if not actions:
            stats.terminals += 1
            _record_endpoint(
                stats,
                depth,
                self.cfg.endpoint_depth_bucket_width,
                self.cfg.endpoint_depth_bucket_max,
            )
            return float(state.score_diff(traverser)), rng

        sampling_policy = _sampling_policy(
            result.policy, result.legal_mask, self.cfg.outcome_sampling_epsilon
        )
        rng, random_value = _next_double(rng)
        action = _sample_policy(sampling_policy, actions, random_value)
        action_prob = max(float(sampling_policy[action]), self.cfg.epsilon)
        state.push_unified_action(action)
        try:
            child_value, rng = self._traverse(
                state, traverser, iteration, depth + 1, rng, stats, samples
            )
        finally:
            state.pop_action()

        stats.sampled_actions += 1
        stats.regret_matching_decisions += 1
        sampled_action_value = child_value / action_prob
        if self.cfg.outcome_sampling_value_clip is not None:
            clip = float(self.cfg.outcome_sampling_value_clip)
            sampled_action_value = max(-clip, min(clip, sampled_action_value))
        node_value = float(result.policy[action]) * sampled_action_value
        if player == traverser:
            self._record_advantage(
                result,
                action,
                sampled_action_value,
                node_value,
                iteration,
                player,
                stats,
                samples,
            )
        return node_value, rng

    def _policy_result(self, state: GameState, player: int) -> PolicyResult:
        info_state = encode_info_state(state, player, self.cfg.encoding)
        legal_mask = np.zeros(self.cfg.action_size, dtype=bool)
        legal_mask[state.unified_legal_actions()] = True
        return self.policy.one(player, info_state, legal_mask)

    def _cutoff(
        self,
        state: GameState,
        traverser: int,
        depth: int,
        stats: TraversalStats,
    ) -> float | None:
        if self.cfg.max_nodes is not None and stats.nodes >= self.cfg.max_nodes:
            stats.node_limit_cutoffs += 1
        elif state.terminal:
            stats.terminals += 1
        elif self.cfg.max_depth is not None and depth >= self.cfg.max_depth:
            stats.depth_cutoffs += 1
        else:
            return None
        _record_endpoint(
            stats,
            depth,
            self.cfg.endpoint_depth_bucket_width,
            self.cfg.endpoint_depth_bucket_max,
        )
        return float(state.score_diff(traverser))

    def _record_strategy(
        self,
        result: PolicyResult,
        player: int,
        traverser: int,
        iteration: int,
        depth: int,
        stats: TraversalStats,
        samples: Samples,
    ) -> None:
        if player == traverser:
            if not self.cfg.store_strategy_on_traverser_nodes:
                return
        elif not self.cfg.store_strategy_on_opponent_nodes:
            return
        if depth % self.cfg.strategy_sample_interval != 0:
            return
        samples.strategy.append(
            TrainingSample(
                info_state=result.info_state,
                target=result.policy.copy(),
                legal_mask=result.legal_mask.copy(),
                iteration=iteration,
                player=player,
            )
        )
        stats.strategy_samples += 1

    def _record_advantage(
        self,
        result: PolicyResult,
        action: int,
        sampled_action_value: float,
        node_value: float,
        iteration: int,
        player: int,
        stats: TraversalStats,
        samples: Samples,
    ) -> None:
        target = np.zeros(self.cfg.action_size, dtype=np.float32)
        target[result.legal_mask] = -node_value
        target[action] = sampled_action_value - node_value
        samples.advantage.append(
            TrainingSample(
                info_state=result.info_state,
                target=target,
                legal_mask=result.legal_mask.copy(),
                iteration=iteration,
                player=player,
            )
        )
        stats.advantage_samples += 1


class InterleavedContext:
    def __init__(
        self,
        state: GameState,
        *,
        traverser: int,
        iteration: int,
        rng: int,
        cfg: PrototypeConfig,
    ) -> None:
        self.state = state
        self.traverser = traverser
        self.iteration = iteration
        self.rng = rng
        self.cfg = cfg
        self.stats = TraversalStats()
        self.samples = Samples()
        self.stack: list[Frame] = [EnterFrame(0)]
        self.pending: PolicyRequest | None = None
        self.last_value = 0.0
        self.done = False
        self.value = 0.0

    def advance_until_policy(self, context_index: int) -> None:
        while not self.done and self.pending is None and self.stack:
            frame = self.stack.pop()
            if isinstance(frame, EnterFrame):
                self._enter(frame.depth, context_index)
            else:
                self._after_child(frame)
        if not self.stack and self.pending is None and not self.done:
            self.done = True
            self.value = self.last_value

    def apply_policy(self, result: PolicyResult) -> None:
        if self.pending is None:
            raise RuntimeError("context has no pending policy request")
        self.pending = None
        player = result.player if hasattr(result, "player") else int(self.state.current_player)
        depth = int(getattr(result, "depth", -1))
        if depth < 0:
            raise RuntimeError("policy result is missing depth")
        self._record_strategy(result, player, depth)
        actions = [int(action) for action in np.flatnonzero(result.legal_mask)]
        if not actions:
            self.stats.terminals += 1
            _record_endpoint(
                self.stats,
                depth,
                self.cfg.endpoint_depth_bucket_width,
                self.cfg.endpoint_depth_bucket_max,
            )
            self._return_value(float(self.state.score_diff(self.traverser)))
            return

        sampling_policy = _sampling_policy(
            result.policy, result.legal_mask, self.cfg.outcome_sampling_epsilon
        )
        self.rng, random_value = _next_double(self.rng)
        action = _sample_policy(sampling_policy, actions, random_value)
        action_prob = max(float(sampling_policy[action]), self.cfg.epsilon)
        self.state.push_unified_action(action)
        self.stack.append(
            AfterChildFrame(
                depth=depth,
                player=player,
                action=action,
                action_prob=action_prob,
                info_state=result.info_state,
                legal_mask=result.legal_mask,
                policy=result.policy,
                sampling_policy=sampling_policy,
                fallback=result.fallback,
                tie_size=result.tie_size,
                full_tie=result.full_tie,
            )
        )
        self.stack.append(EnterFrame(depth + 1))

    def _enter(self, depth: int, context_index: int) -> None:
        self.stats.nodes += 1
        self.stats.max_depth_reached = max(self.stats.max_depth_reached, depth)
        cutoff = self._cutoff(depth)
        if cutoff is not None:
            self._return_value(cutoff)
            return
        player = int(self.state.current_player)
        info_state = encode_info_state(self.state, player, self.cfg.encoding)
        legal_mask = np.zeros(self.cfg.action_size, dtype=bool)
        legal_mask[self.state.unified_legal_actions()] = True
        request = PolicyRequest(context_index, player, info_state, legal_mask)
        request.depth = depth  # type: ignore[attr-defined]
        self.pending = request

    def _after_child(self, frame: AfterChildFrame) -> None:
        child_value = self.last_value
        self.state.pop_action()
        self.stats.sampled_actions += 1
        self.stats.regret_matching_decisions += 1
        sampled_action_value = child_value / frame.action_prob
        if self.cfg.outcome_sampling_value_clip is not None:
            clip = float(self.cfg.outcome_sampling_value_clip)
            sampled_action_value = max(-clip, min(clip, sampled_action_value))
        node_value = float(frame.policy[frame.action]) * sampled_action_value
        if frame.player == self.traverser:
            target = np.zeros(self.cfg.action_size, dtype=np.float32)
            target[frame.legal_mask] = -node_value
            target[frame.action] = sampled_action_value - node_value
            self.samples.advantage.append(
                TrainingSample(
                    info_state=frame.info_state,
                    target=target,
                    legal_mask=frame.legal_mask.copy(),
                    iteration=self.iteration,
                    player=frame.player,
                )
            )
            self.stats.advantage_samples += 1
        self._return_value(node_value)

    def _cutoff(self, depth: int) -> float | None:
        if self.cfg.max_nodes is not None and self.stats.nodes >= self.cfg.max_nodes:
            self.stats.node_limit_cutoffs += 1
        elif self.state.terminal:
            self.stats.terminals += 1
        elif self.cfg.max_depth is not None and depth >= self.cfg.max_depth:
            self.stats.depth_cutoffs += 1
        else:
            return None
        _record_endpoint(
            self.stats,
            depth,
            self.cfg.endpoint_depth_bucket_width,
            self.cfg.endpoint_depth_bucket_max,
        )
        return float(self.state.score_diff(self.traverser))

    def _return_value(self, value: float) -> None:
        self.last_value = value
        if not self.stack:
            self.done = True
            self.value = value

    def _record_strategy(self, result: PolicyResult, player: int, depth: int) -> None:
        if player == self.traverser:
            if not self.cfg.store_strategy_on_traverser_nodes:
                return
        elif not self.cfg.store_strategy_on_opponent_nodes:
            return
        if depth % self.cfg.strategy_sample_interval != 0:
            return
        self.samples.strategy.append(
            TrainingSample(
                info_state=result.info_state,
                target=result.policy.copy(),
                legal_mask=result.legal_mask.copy(),
                iteration=self.iteration,
                player=player,
            )
        )
        self.stats.strategy_samples += 1


class InterleavedPrototype:
    def __init__(self, cfg: PrototypeConfig, policy: BatchedPolicy) -> None:
        self.cfg = cfg
        self.policy = policy
        self.scheduler_seconds = 0.0

    def run(
        self,
        states: list[GameState],
        *,
        traverser: int,
        iteration: int,
        rng_seeds: list[int],
        interleave_width: int,
        max_batch: int,
    ) -> tuple[list[float], list[int], list[TraversalStats], list[Samples], list[int]]:
        contexts = [
            InterleavedContext(
                state,
                traverser=traverser,
                iteration=iteration,
                rng=rng,
                cfg=self.cfg,
            )
            for state, rng in zip(states, rng_seeds, strict=True)
        ]
        active = list(range(len(contexts)))
        batch_sizes: list[int] = []
        while active:
            start = time.perf_counter()
            runnable = active[: max(1, interleave_width)]
            for context_index in runnable:
                contexts[context_index].advance_until_policy(context_index)
            requests: list[PolicyRequest] = []
            request_contexts: list[int] = []
            for context_index in runnable:
                request = contexts[context_index].pending
                if request is not None:
                    requests.append(request)
                    request_contexts.append(context_index)
                    if len(requests) >= max_batch:
                        break
            self.scheduler_seconds += time.perf_counter() - start

            if requests:
                results = self.policy.batch(requests)
                batch_sizes.append(len(requests))
                for context_index, request, result in zip(
                    request_contexts, requests, results, strict=True
                ):
                    result.player = request.player  # type: ignore[attr-defined]
                    result.depth = request.depth  # type: ignore[attr-defined]
                    contexts[context_index].apply_policy(result)
                continue

            active = [idx for idx in active if not contexts[idx].done]
        return (
            [context.value for context in contexts],
            [context.rng for context in contexts],
            [context.stats for context in contexts],
            [context.samples for context in contexts],
            batch_sizes,
        )


def _sample_checksum(samples: list[Samples]) -> dict[str, float]:
    adv = [sample for group in samples for sample in group.advantage]
    strat = [sample for group in samples for sample in group.strategy]
    return {
        "advantage_count": len(adv),
        "strategy_count": len(strat),
        "advantage_target_sum": float(sum(float(sample.target.sum()) for sample in adv)),
        "strategy_target_sum": float(sum(float(sample.target.sum()) for sample in strat)),
    }


def _stats_checksum(stats: list[TraversalStats]) -> dict[str, int]:
    keys = [
        "nodes",
        "terminals",
        "depth_cutoffs",
        "node_limit_cutoffs",
        "advantage_samples",
        "strategy_samples",
        "sampled_actions",
        "regret_matching_decisions",
    ]
    return {key: int(sum(getattr(row, key) for row in stats)) for key in keys}


def _assert_close(name: str, left: Any, right: Any, *, atol: float = 1.0e-6) -> None:
    if isinstance(left, float) or isinstance(right, float):
        if abs(float(left) - float(right)) > atol:
            raise AssertionError(f"{name} mismatch: {left!r} != {right!r}")
        return
    if left != right:
        raise AssertionError(f"{name} mismatch: {left!r} != {right!r}")


def _build_proto_config(cfg: Any, max_depth: int | None, max_nodes: int | None) -> PrototypeConfig:
    probe = GameState.new_game(
        cfg.rules.to_lost_cities_config(seed=cfg.run.seed), seed=cfg.run.seed
    )
    return PrototypeConfig(
        action_size=2 * probe.config.hand_size + 1 + probe.config.n_colors,
        encoding=cfg.encoding,
        epsilon=float(cfg.traversal.regret_matching_epsilon),
        outcome_sampling_epsilon=float(cfg.traversal.outcome_sampling_epsilon),
        outcome_sampling_value_clip=cfg.traversal.outcome_sampling_value_clip,
        max_depth=max_depth,
        max_nodes=max_nodes,
        strategy_sample_interval=int(cfg.traversal.strategy_sample_interval),
        store_strategy_on_traverser_nodes=bool(cfg.traversal.store_strategy_on_traverser_nodes),
        store_strategy_on_opponent_nodes=bool(cfg.traversal.store_strategy_on_opponent_nodes),
        endpoint_depth_bucket_width=int(cfg.traversal.endpoint_depth_bucket_width),
        endpoint_depth_bucket_max=int(cfg.traversal.endpoint_depth_bucket_max),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deep_cfr/default.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--traversals", type=int, default=64)
    parser.add_argument("--interleave-width", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    proto_cfg = _build_proto_config(cfg, args.max_depth, args.max_nodes)
    probe = GameState.new_game(
        cfg.rules.to_lost_cities_config(seed=cfg.run.seed), seed=cfg.run.seed
    )
    input_dim_value = input_dim(probe, cfg.encoding)
    networks = [
        DeepCFRMLP.from_config(input_dim_value, proto_cfg.action_size, cfg.network)
        .to(device)
        .eval()
        for _ in range(2)
    ]
    game_config = cfg.rules.to_lost_cities_config(seed=cfg.run.seed)
    states = [
        GameState.new_game(game_config, seed=args.seed + index) for index in range(args.traversals)
    ]
    rng_seeds = [args.seed * 1009 + index * 9176 + 1 for index in range(args.traversals)]

    recursive_policy = BatchedPolicy(networks, device=device, epsilon=proto_cfg.epsilon)
    recursive = RecursivePrototype(proto_cfg, recursive_policy)
    start = time.perf_counter()
    recursive_rows = [
        recursive.traverse(
            state.clone(),
            traverser=0,
            iteration=1,
            rng=rng,
        )
        for state, rng in zip(states, rng_seeds, strict=True)
    ]
    recursive_seconds = time.perf_counter() - start

    interleaved_policy = BatchedPolicy(networks, device=device, epsilon=proto_cfg.epsilon)
    interleaved = InterleavedPrototype(proto_cfg, interleaved_policy)
    start = time.perf_counter()
    values, rng_out, stats, samples, scheduler_batch_sizes = interleaved.run(
        [state.clone() for state in states],
        traverser=0,
        iteration=1,
        rng_seeds=rng_seeds,
        interleave_width=args.interleave_width,
        max_batch=args.max_batch,
    )
    interleaved_seconds = time.perf_counter() - start

    recursive_values = [row[0] for row in recursive_rows]
    recursive_rng = [row[1] for row in recursive_rows]
    recursive_stats = [row[2] for row in recursive_rows]
    recursive_samples = [row[3] for row in recursive_rows]

    for idx, (left, right) in enumerate(zip(recursive_values, values, strict=True)):
        _assert_close(f"value[{idx}]", left, right, atol=1.0e-4)
    _assert_close("rng", recursive_rng, rng_out)
    _assert_close("stats", _stats_checksum(recursive_stats), _stats_checksum(stats))
    recursive_sample_checksum = _sample_checksum(recursive_samples)
    interleaved_sample_checksum = _sample_checksum(samples)
    for key, left in recursive_sample_checksum.items():
        _assert_close(key, left, interleaved_sample_checksum[key], atol=1.0e-2)

    realized = scheduler_batch_sizes
    result = {
        "config": args.config,
        "device": str(device),
        "traversals": args.traversals,
        "interleave_width": args.interleave_width,
        "max_batch": args.max_batch,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "recursive_seconds": recursive_seconds,
        "interleaved_seconds": interleaved_seconds,
        "speedup": recursive_seconds / interleaved_seconds if interleaved_seconds > 0 else 0.0,
        "recursive_policy_batches": recursive_policy.batch_sizes,
        "interleaved_policy_batches": interleaved_policy.batch_sizes,
        "scheduler_batch_sizes": realized,
        "scheduler_batch_mean": float(statistics.mean(realized)) if realized else 0.0,
        "scheduler_batch_max": max(realized) if realized else 0,
        "scheduler_seconds": interleaved.scheduler_seconds,
        "recursive_forward_seconds": recursive_policy.forward_seconds,
        "interleaved_forward_seconds": interleaved_policy.forward_seconds,
        "stats": _stats_checksum(stats),
        "sample_checksum": interleaved_sample_checksum,
    }

    print("Option B interleaved traversal prototype")
    print(f"device={device} traversals={args.traversals} max_depth={args.max_depth}")
    print("mode          total_s  forward_s  sched_s  batch_mean  batch_max")
    print(
        f"recursive     {recursive_seconds:7.3f}  {recursive_policy.forward_seconds:9.3f}"
        f"  {'-':>7}  {1.0:10.1f}  {1:9d}"
    )
    print(
        f"interleaved   {interleaved_seconds:7.3f}  {interleaved_policy.forward_seconds:9.3f}"
        f"  {interleaved.scheduler_seconds:7.3f}  {result['scheduler_batch_mean']:10.1f}"
        f"  {result['scheduler_batch_max']:9d}"
    )
    print(f"speedup       {result['speedup']:.2f}x")
    print("parity        PASS")

    output = Path(args.output) if args.output else Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
