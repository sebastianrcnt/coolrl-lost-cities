from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.deep_cfr.memory import TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.traversal_stats import TraversalStats
from coolrl_lost_cities.games.classic.game import GameState


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


def _masked_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    policy = np.zeros_like(logits, dtype=np.float32)
    legal = np.flatnonzero(legal_mask)
    if len(legal) == 0:
        return policy
    legal_logits = logits[legal].astype(np.float32)
    shifted = legal_logits - float(np.max(legal_logits))
    values = np.exp(shifted, dtype=np.float32)
    total = float(values.sum())
    if total <= 0.0:
        policy[legal] = 1.0 / float(len(legal))
        return policy
    policy[legal] = values / total
    return policy


def _record_endpoint(stats: TraversalStats, depth: int, width: int, max_depth: int) -> None:
    stats.endpoint_depth_sum += depth
    start = (depth // width) * width
    key = f"{max_depth}_plus" if start >= max_depth else f"{start}_{start + width - 1}"
    stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1


@dataclass
class InterleavedTraversalConfig:
    action_size: int
    encoding: Any
    epsilon: float
    outcome_sampling_epsilon: float
    outcome_sampling_value_clip: float | None
    outcome_unsampled_regret: str
    max_depth: int | None
    max_nodes: int | None
    strategy_sample_interval: int
    store_strategy_on_traverser_nodes: bool
    store_strategy_on_opponent_nodes: bool
    opponent_policy: str
    endpoint_depth_bucket_width: int
    endpoint_depth_bucket_max: int
    deterministic: bool = False


@dataclass
class PolicyResult:
    info_state: np.ndarray
    legal_mask: np.ndarray
    policy: np.ndarray
    fallback: bool
    tie_size: int
    full_tie: bool
    player: int = -1
    depth: int = -1
    kind: str = "advantage"


@dataclass
class PolicyRequest:
    context_index: int
    player: int
    info_state: np.ndarray
    legal_mask: np.ndarray
    depth: int
    network_kind: str = "advantage"


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
    swapped_deck_index: int


@dataclass
class FixedActionFrame:
    swapped_deck_index: int


@dataclass
class EnterFrame:
    depth: int


Frame = EnterFrame | AfterChildFrame | FixedActionFrame


class BatchedPolicy:
    def __init__(
        self,
        networks: list[torch.nn.Module],
        *,
        device: torch.device,
        epsilon: float,
        strategy_network: torch.nn.Module | None = None,
        deterministic: bool = False,
    ) -> None:
        self.networks = networks
        self.strategy_network = strategy_network
        self.device = device
        self.epsilon = epsilon
        self.deterministic = deterministic
        self.batch_sizes: list[int] = []
        self.forward_seconds = 0.0

    def batch(self, requests: list[PolicyRequest]) -> list[PolicyResult]:
        if not requests:
            return []
        out: list[PolicyResult | None] = [None] * len(requests)
        if self.deterministic:
            groups = [
                (request.network_kind, request.player, [index])
                for index, request in enumerate(requests)
            ]
        else:
            group_keys = sorted({(request.network_kind, request.player) for request in requests})
            groups = [
                (
                    network_kind,
                    player,
                    [
                        idx
                        for idx, request in enumerate(requests)
                        if (request.network_kind, request.player) == (network_kind, player)
                    ],
                )
                for network_kind, player in group_keys
            ]
        for network_kind, player, indices in groups:
            indices = [idx for idx in indices]
            states = np.stack([requests[idx].info_state for idx in indices]).astype(np.float32)
            x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                network = self._network(network_kind, player)
                values = network(x).detach().cpu().numpy().astype(np.float32)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.forward_seconds += time.perf_counter() - start
            self.batch_sizes.append(len(indices))
            for local_idx, request_idx in enumerate(indices):
                request = requests[request_idx]
                if request.network_kind == "strategy":
                    policy = _masked_softmax(values[local_idx], request.legal_mask)
                    fallback = False
                    tie_size = 0
                    full_tie = False
                else:
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

    def _network(self, network_kind: str, player: int) -> torch.nn.Module:
        if network_kind == "advantage":
            return self.networks[player]
        if network_kind == "strategy" and self.strategy_network is not None:
            return self.strategy_network
        raise ValueError(f"unsupported policy network request: {network_kind!r}")


class InterleavedContext:
    def __init__(
        self,
        state: GameState,
        *,
        traverser: int,
        iteration: int,
        rng: int,
        cfg: InterleavedTraversalConfig,
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
            elif isinstance(frame, AfterChildFrame):
                self._after_child(frame)
            else:
                self._after_fixed_action(frame)
        if not self.stack and self.pending is None and not self.done:
            self.done = True
            self.value = self.last_value

    def apply_policy(self, result: PolicyResult) -> None:
        if self.pending is None:
            raise RuntimeError("context has no pending policy request")
        self.pending = None
        player = result.player
        depth = result.depth
        if depth < 0:
            raise RuntimeError("policy result is missing depth")
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

        if result.kind == "strategy":
            self.rng, random_value = _next_double(self.rng)
            action = _sample_policy(result.policy, actions, random_value)
            swapped_deck_index = self._sample_deck_draw_chance(action)
            self.state.push_unified_action(action)
            self.stack.append(FixedActionFrame(swapped_deck_index=swapped_deck_index))
            self.stack.append(EnterFrame(depth + 1))
            return

        self._record_strategy(result, player, depth)
        sampling_policy = _sampling_policy(
            result.policy, result.legal_mask, self.cfg.outcome_sampling_epsilon
        )
        self.rng, random_value = _next_double(self.rng)
        action = _sample_policy(sampling_policy, actions, random_value)
        action_prob = max(float(sampling_policy[action]), self.cfg.epsilon)
        swapped_deck_index = self._sample_deck_draw_chance(action)
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
                swapped_deck_index=swapped_deck_index,
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
        network_kind = (
            "strategy"
            if player != self.traverser and self.cfg.opponent_policy == "average_strategy"
            else "advantage"
        )
        request = PolicyRequest(context_index, player, info_state, legal_mask, depth, network_kind)
        self.pending = request

    def _after_child(self, frame: AfterChildFrame) -> None:
        child_value = self.last_value
        self.state.pop_action()
        if frame.swapped_deck_index >= 0:
            self.state.swap_deck_cards(frame.swapped_deck_index, len(self.state.deck) - 1)
        self.stats.sampled_actions += 1
        self._record_regret_matching_decision(frame)
        sampled_action_value = child_value / frame.action_prob
        if self.cfg.outcome_sampling_value_clip is not None:
            clip = float(self.cfg.outcome_sampling_value_clip)
            sampled_action_value = max(-clip, min(clip, sampled_action_value))
        node_value = float(frame.policy[frame.action]) * sampled_action_value
        if frame.player == self.traverser:
            target = np.zeros(self.cfg.action_size, dtype=np.float32)
            if self.cfg.outcome_unsampled_regret == "negative_node_value":
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

    def _after_fixed_action(self, frame: FixedActionFrame) -> None:
        child_value = self.last_value
        self.state.pop_action()
        if frame.swapped_deck_index >= 0:
            self.state.swap_deck_cards(frame.swapped_deck_index, len(self.state.deck) - 1)
        self._return_value(child_value)

    def _sample_deck_draw_chance(self, unified_action: int) -> int:
        deck_draw_action = 2 * self.state.config.hand_size
        deck_len = len(self.state.deck)
        if self.state.phase != "draw" or unified_action != deck_draw_action or deck_len <= 1:
            return -1
        self.rng, value = _next_u32(self.rng)
        sampled_index = int(value % deck_len)
        if sampled_index == deck_len - 1:
            return -1
        self.state.swap_deck_cards(sampled_index, deck_len - 1)
        return sampled_index

    def _record_regret_matching_decision(self, frame: AfterChildFrame) -> None:
        self.stats.regret_matching_decisions += 1
        if not frame.fallback:
            return
        self.stats.regret_fallback_count += 1
        self.stats.regret_fallback_depth_sum += frame.depth
        self._record_fallback_depth_bucket(frame.depth)
        expeditions = self.state.expeditions
        opened_colors = sum(1 for cards in expeditions[frame.player] if cards)
        self.stats.regret_fallback_opened_colors_sum += opened_colors
        self.stats.regret_fallback_opened_colors_buckets[opened_colors] = (
            self.stats.regret_fallback_opened_colors_buckets.get(opened_colors, 0) + 1
        )
        if frame.tie_size > 1:
            self.stats.regret_fallback_argmax_tie_count += 1
            self.stats.regret_fallback_argmax_tie_size_sum += frame.tie_size
        if frame.full_tie:
            self.stats.regret_fallback_argmax_full_tie_count += 1

        card_action_size = self.state.card_action_size
        hand = self.state.hand_slots(frame.player)
        legal_actions = self.state.unified_legal_actions()
        self.stats.regret_fallback_legal_actions_sum += len(legal_actions)
        for legal_action in legal_actions:
            if legal_action < card_action_size:
                if legal_action % 2 == 1:
                    self.stats.regret_fallback_legal_discard_sum += 1
                    continue
                card = hand[legal_action // 2]
                color = int(card.color)
                if not expeditions[frame.player][color]:
                    self.stats.regret_fallback_legal_open_new_sum += 1
                    self.stats.regret_fallback_open_new_available_by_color[color] = (
                        self.stats.regret_fallback_open_new_available_by_color.get(color, 0) + 1
                    )
                else:
                    self.stats.regret_fallback_legal_play_existing_sum += 1
                continue
            if legal_action == card_action_size:
                self.stats.regret_fallback_legal_draw_deck_sum += 1
            else:
                self.stats.regret_fallback_legal_draw_pile_sum += 1

        if frame.action < card_action_size:
            if frame.action % 2 == 1:
                self.stats.regret_fallback_action_discard += 1
                return
            card = hand[frame.action // 2]
            color = int(card.color)
            if not expeditions[frame.player][color]:
                self.stats.regret_fallback_action_open_new += 1
                self.stats.regret_fallback_open_new_selected_by_color[color] = (
                    self.stats.regret_fallback_open_new_selected_by_color.get(color, 0) + 1
                )
            else:
                self.stats.regret_fallback_action_play_existing += 1
            return
        if frame.action == card_action_size:
            self.stats.regret_fallback_action_draw_deck += 1
        else:
            self.stats.regret_fallback_action_draw_pile += 1

    def _record_fallback_depth_bucket(self, depth: int) -> None:
        width = 50
        max_depth = 400
        start = (depth // width) * width
        key = f"{max_depth}_plus" if start >= max_depth else f"{start}_{start + width - 1}"
        self.stats.regret_fallback_depth_buckets[key] = (
            self.stats.regret_fallback_depth_buckets.get(key, 0) + 1
        )

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


class InterleavedTraversalScheduler:
    def __init__(self, cfg: InterleavedTraversalConfig, policy: BatchedPolicy) -> None:
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
                    result.kind = request.network_kind
                    result.player = request.player
                    result.depth = request.depth
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


def run_interleaved_traversal_batch(
    advantage_networks: list[torch.nn.Module],
    strategy_network: torch.nn.Module | None,
    game_config: Any,
    seeds: list[int],
    player: int,
    iteration: int,
    *,
    device: torch.device,
    action_size: int,
    encoding: Any,
    epsilon: float,
    strategy_sample_interval: int,
    store_strategy_on_traverser_nodes: bool,
    store_strategy_on_opponent_nodes: bool,
    max_depth: int | None,
    max_nodes: int | None,
    outcome_sampling_epsilon: float,
    outcome_sampling_value_clip: float | None,
    outcome_unsampled_regret: str,
    opponent_policy: str,
    endpoint_depth_bucket_width: int,
    endpoint_depth_bucket_max: int,
    seed: int,
    interleave_width: int,
    interleave_max_batch: int,
    traversal_start_index: int = 0,
    deterministic: bool = False,
) -> tuple[TraversalStats, list[TrainingSample], list[TrainingSample], dict[str, float | int]]:
    cfg = InterleavedTraversalConfig(
        action_size=action_size,
        encoding=encoding,
        epsilon=epsilon,
        outcome_sampling_epsilon=outcome_sampling_epsilon,
        outcome_sampling_value_clip=outcome_sampling_value_clip,
        outcome_unsampled_regret=outcome_unsampled_regret,
        max_depth=max_depth,
        max_nodes=max_nodes,
        strategy_sample_interval=strategy_sample_interval,
        store_strategy_on_traverser_nodes=store_strategy_on_traverser_nodes,
        store_strategy_on_opponent_nodes=store_strategy_on_opponent_nodes,
        opponent_policy=opponent_policy,
        endpoint_depth_bucket_width=endpoint_depth_bucket_width,
        endpoint_depth_bucket_max=endpoint_depth_bucket_max,
        deterministic=deterministic,
    )
    states = [GameState.new_game(game_config, seed=game_seed) for game_seed in seeds]
    rng_seeds = [
        int(seed) + (int(traversal_start_index) + index) * 1_000_003 for index in range(len(seeds))
    ]
    policy = BatchedPolicy(
        advantage_networks,
        device=device,
        epsilon=cfg.epsilon,
        strategy_network=strategy_network,
        deterministic=cfg.deterministic,
    )
    scheduler = InterleavedTraversalScheduler(cfg, policy)
    _values, _rng_out, stats_rows, sample_rows, batch_sizes = scheduler.run(
        states,
        traverser=player,
        iteration=iteration,
        rng_seeds=rng_seeds,
        interleave_width=max(1, int(interleave_width)),
        max_batch=max(1, int(interleave_max_batch)),
    )
    total_stats = TraversalStats()
    advantage_samples: list[TrainingSample] = []
    strategy_samples: list[TrainingSample] = []
    for stats, samples in zip(stats_rows, sample_rows, strict=True):
        total_stats.accumulate(stats)
        advantage_samples.extend(samples.advantage)
        strategy_samples.extend(samples.strategy)
    runtime_stats: dict[str, float | int] = {
        "interleaved/batches": len(batch_sizes),
        "interleaved/requests": sum(batch_sizes),
        "interleaved/max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "interleaved/avg_batch_size": (float(statistics.mean(batch_sizes)) if batch_sizes else 0.0),
        "interleaved/scheduler_seconds": scheduler.scheduler_seconds,
        "interleaved/forward_seconds": policy.forward_seconds,
    }
    return total_stats, advantage_samples, strategy_samples, runtime_stats
