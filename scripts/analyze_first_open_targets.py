#!/usr/bin/env python
"""Audit regenerated first-open advantage targets from Deep CFR checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.interleaved_traversal import (
    AfterChildFrame,
    BatchedPolicy,
    InterleavedContext,
    InterleavedTraversalConfig,
    PolicyRequest,
    Samples,
)
from coolrl_lost_cities.games.classic.deep_cfr.memory import TrainingSample
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.deep_cfr.traversal_stats import TraversalStats


@dataclass
class TargetBucket:
    target_values: list[float] = field(default_factory=list)
    sampled_target_values: list[float] = field(default_factory=list)
    policy_probs: list[float] = field(default_factory=list)
    sampled: int = 0
    legal: int = 0

    def add(
        self,
        *,
        target: float,
        policy_prob: float,
        sampled: bool,
    ) -> None:
        self.target_values.append(float(target))
        if sampled:
            self.sampled_target_values.append(float(target))
        self.policy_probs.append(float(policy_prob))
        self.sampled += int(sampled)
        self.legal += 1

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": len(self.target_values),
            "target_mean": _mean(self.target_values),
            "target_p10": _percentile(self.target_values, 10),
            "target_p25": _percentile(self.target_values, 25),
            "target_p50": _percentile(self.target_values, 50),
            "target_p75": _percentile(self.target_values, 75),
            "target_p90": _percentile(self.target_values, 90),
            "target_positive_rate": _positive_rate(self.target_values),
            "sampled_target_mean": _mean(self.sampled_target_values),
            "sampled_target_p25": _percentile(self.sampled_target_values, 25),
            "sampled_target_p50": _percentile(self.sampled_target_values, 50),
            "sampled_target_p75": _percentile(self.sampled_target_values, 75),
            "sampled_target_positive_rate": _positive_rate(self.sampled_target_values),
            "policy_prob_mean": _mean(self.policy_probs),
            "sampled": self.sampled,
            "sampled_rate": self.sampled / max(1, self.legal),
        }


@dataclass
class TargetAudit:
    buckets: dict[str, TargetBucket] = field(default_factory=lambda: defaultdict(TargetBucket))
    candidate_states: int = 0
    first_open_candidates: int = 0
    sampled_open_actions: int = 0
    traverser_samples: int = 0

    def accumulate(self, other: TargetAudit) -> None:
        self.candidate_states += other.candidate_states
        self.first_open_candidates += other.first_open_candidates
        self.sampled_open_actions += other.sampled_open_actions
        self.traverser_samples += other.traverser_samples
        for label, bucket in other.buckets.items():
            target = self.buckets[label]
            target.target_values.extend(bucket.target_values)
            target.sampled_target_values.extend(bucket.sampled_target_values)
            target.policy_probs.extend(bucket.policy_probs)
            target.sampled += bucket.sampled
            target.legal += bucket.legal

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_states": self.candidate_states,
            "first_open_candidates": self.first_open_candidates,
            "sampled_open_actions": self.sampled_open_actions,
            "traverser_samples": self.traverser_samples,
            "buckets": {key: value.to_dict() for key, value in sorted(self.buckets.items())},
        }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _positive_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value > 0.0) / len(values)


def _numeric_value(card: Any, min_rank: int) -> int:
    if card.rank == 0:
        return 0
    return min_rank + card.rank - 1


def _open_quality(state: GameState, player: int, color: int) -> str:
    expedition = state.expeditions[player][color]
    hand_cards = [
        card for card in state.hand_slots(player) if card is not None and card.color == color
    ]
    last_numeric = state.last_numeric_rank(player, color)
    current_sum = sum(_numeric_value(card, state.config.min_rank) for card in expedition)
    current_wagers = sum(1 for card in expedition if card.rank == 0)
    playable_numeric = [card for card in hand_cards if card.rank > 0 and card.rank > last_numeric]
    playable_wagers = [card for card in hand_cards if card.rank == 0 and last_numeric == 0]
    projected_sum = current_sum + sum(
        _numeric_value(card, state.config.min_rank) for card in playable_numeric
    )
    projected_wagers = current_wagers + len(playable_wagers)
    projected_len = len(expedition) + len(playable_numeric) + len(playable_wagers)
    recoverable_score = (projected_sum + state.config.expedition_penalty) * (projected_wagers + 1)
    if recoverable_score >= 0:
        return "open_good"
    if projected_len >= state.config.bonus_threshold:
        return "open_weak"
    return "open_bad"


def _classify_action(state: GameState, unified_action: int, player: int) -> str:
    card_action_size = state.config.hand_size * 2
    if unified_action >= card_action_size:
        return "draw_deck" if unified_action == card_action_size else "draw_pile"
    if unified_action % 2 == 1:
        return "discard"
    card = state.hand_slots(player)[unified_action // 2]
    if card is None:
        return "invalid_play"
    color = int(card.color)
    if state.expeditions[player][color]:
        return "play_existing"
    return _open_quality(state, player, color)


class AuditedInterleavedContext(InterleavedContext):
    def __init__(
        self,
        state: GameState,
        *,
        traverser: int,
        iteration: int,
        rng: int,
        cfg: InterleavedTraversalConfig,
    ) -> None:
        super().__init__(state, traverser=traverser, iteration=iteration, rng=rng, cfg=cfg)
        self.target_audit = TargetAudit()

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
            self._record_target_audit(frame, target)
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

    def _record_target_audit(self, frame: AfterChildFrame, target: np.ndarray) -> None:
        labels = {
            int(action): _classify_action(self.state, int(action), frame.player)
            for action in np.flatnonzero(frame.legal_mask)
        }
        open_actions = [action for action, label in labels.items() if label.startswith("open_")]
        if not open_actions:
            return
        self.target_audit.candidate_states += 1
        self.target_audit.first_open_candidates += len(open_actions)
        self.target_audit.traverser_samples += 1
        if labels.get(frame.action, "").startswith("open_"):
            self.target_audit.sampled_open_actions += 1
        for action in open_actions:
            self.target_audit.buckets[labels[action]].add(
                target=float(target[action]),
                policy_prob=float(frame.policy[action]),
                sampled=action == frame.action,
            )
        for action, label in labels.items():
            if label.startswith("open_"):
                continue
            self.target_audit.buckets["non_open"].add(
                target=float(target[action]),
                policy_prob=float(frame.policy[action]),
                sampled=action == frame.action,
            )


class AuditedInterleavedTraversalScheduler:
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
    ) -> tuple[list[TraversalStats], list[Samples], TargetAudit, list[int]]:
        contexts = [
            AuditedInterleavedContext(
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

        audit = TargetAudit()
        for context in contexts:
            audit.accumulate(context.target_audit)
        return (
            [context.stats for context in contexts],
            [context.samples for context in contexts],
            audit,
            batch_sizes,
        )


def _load_checkpoint(
    checkpoint: Path, device: torch.device
) -> tuple[Any, LostCitiesConfig, list[torch.nn.Module], torch.nn.Module | None, int, int]:
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    game_config = LostCitiesConfig(**payload["game_config"])
    input_dim = int(payload["input_dim"])
    action_size = int(payload["action_size"])
    advantage_networks = [
        DeepCFRMLP.from_config(input_dim, action_size, cfg.network).to(device) for _ in range(2)
    ]
    for network, state_dict in zip(advantage_networks, payload["advantage_networks"], strict=True):
        network.load_state_dict(state_dict)
        network.eval()
    strategy_network = None
    if payload.get("strategy_network") is not None:
        strategy_network = DeepCFRMLP.from_config(input_dim, action_size, cfg.network).to(device)
        strategy_network.load_state_dict(payload["strategy_network"])
        strategy_network.eval()
    return cfg, game_config, advantage_networks, strategy_network, input_dim, action_size


def analyze_checkpoint(
    checkpoint: Path,
    *,
    traversals_per_player: int,
    seed: int,
    device: torch.device,
    force_interleave_width: int | None,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    iteration = int(payload.get("iteration", -1))
    cfg, game_config, advantage_networks, strategy_network, _input_dim, action_size = (
        _load_checkpoint(checkpoint, device)
    )
    traversal_cfg = InterleavedTraversalConfig(
        action_size=action_size,
        encoding=cfg.encoding,
        epsilon=cfg.traversal.regret_matching_epsilon,
        outcome_sampling_epsilon=cfg.traversal.outcome_sampling_epsilon,
        outcome_sampling_value_clip=cfg.traversal.outcome_sampling_value_clip,
        outcome_unsampled_regret=cfg.traversal.outcome_unsampled_regret,
        outcome_unsampled_first_open_prior_alpha=getattr(
            cfg.traversal, "outcome_unsampled_first_open_prior_alpha", 0.0
        ),
        all_negative_fallback=cfg.regret_matching.all_negative_fallback,
        max_depth=cfg.traversal.max_depth,
        max_nodes=cfg.traversal.max_nodes_per_traversal,
        strategy_sample_interval=cfg.traversal.strategy_sample_interval,
        store_strategy_on_traverser_nodes=cfg.traversal.store_strategy_on_traverser_nodes,
        store_strategy_on_opponent_nodes=cfg.traversal.store_strategy_on_opponent_nodes,
        opponent_policy=cfg.traversal.opponent_policy,
        endpoint_depth_bucket_width=cfg.traversal.endpoint_depth_bucket_width,
        endpoint_depth_bucket_max=cfg.traversal.endpoint_depth_bucket_max,
        deterministic=cfg.run.deterministic,
    )
    total_stats = TraversalStats()
    total_audit = TargetAudit()
    batch_sizes: list[int] = []
    started = time.perf_counter()
    for player in range(2):
        seeds = [
            int(seed) + iteration * 100_000 + player * 10_000 + index
            for index in range(traversals_per_player)
        ]
        states = [GameState.new_game(game_config, seed=game_seed) for game_seed in seeds]
        rng_seeds = [
            int(seed) + 777_777 + player * 1_000_003 + index * 1_000_003
            for index in range(traversals_per_player)
        ]
        policy = BatchedPolicy(
            advantage_networks,
            device=device,
            epsilon=traversal_cfg.epsilon,
            strategy_network=strategy_network,
            deterministic=traversal_cfg.deterministic,
        )
        scheduler = AuditedInterleavedTraversalScheduler(traversal_cfg, policy)
        stats_rows, _samples_rows, audit, player_batch_sizes = scheduler.run(
            states,
            traverser=player,
            iteration=iteration,
            rng_seeds=rng_seeds,
            interleave_width=force_interleave_width or cfg.traversal.interleave_width,
            max_batch=cfg.traversal.interleave_max_batch,
        )
        for stats in stats_rows:
            total_stats.accumulate(stats)
        total_audit.accumulate(audit)
        batch_sizes.extend(player_batch_sizes)
    return {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "traversals_per_player": traversals_per_player,
        "seed": seed,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "traversal_nodes": total_stats.nodes,
        "advantage_samples": total_stats.advantage_samples,
        "interleaved_batches": len(batch_sizes),
        "interleaved_avg_batch_size": float(statistics.mean(batch_sizes)) if batch_sizes else 0.0,
        "audit": total_audit.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--traversals-per-player", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interleave-width", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in args.checkpoints:
        row = analyze_checkpoint(
            checkpoint,
            traversals_per_player=args.traversals_per_player,
            seed=args.seed,
            device=device,
            force_interleave_width=args.interleave_width,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
