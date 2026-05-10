#!/usr/bin/env python
"""Compare first-open heuristic labels against forced-action continuation value."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP


@dataclass
class Bucket:
    deltas: list[float] = field(default_factory=list)
    open_values: list[float] = field(default_factory=list)
    baseline_values: list[float] = field(default_factory=list)
    policy_probs: list[float] = field(default_factory=list)
    selected: int = 0

    def add(
        self,
        *,
        delta: float,
        open_value: float,
        baseline_value: float,
        policy_prob: float,
        selected: bool,
    ) -> None:
        self.deltas.append(float(delta))
        self.open_values.append(float(open_value))
        self.baseline_values.append(float(baseline_value))
        self.policy_probs.append(float(policy_prob))
        self.selected += int(selected)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": len(self.deltas),
            "delta_mean": _mean(self.deltas),
            "delta_p10": _percentile(self.deltas, 10),
            "delta_p25": _percentile(self.deltas, 25),
            "delta_p50": _percentile(self.deltas, 50),
            "delta_p75": _percentile(self.deltas, 75),
            "delta_p90": _percentile(self.deltas, 90),
            "delta_positive_rate": _positive_rate(self.deltas),
            "open_value_mean": _mean(self.open_values),
            "baseline_value_mean": _mean(self.baseline_values),
            "policy_prob_mean": _mean(self.policy_probs),
            "selected": self.selected,
            "selected_rate": self.selected / max(1, len(self.deltas)),
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


def _regret_matching(
    advantages: np.ndarray,
    legal: np.ndarray,
    *,
    epsilon: float,
    fallback: str,
) -> np.ndarray:
    legal_actions = np.flatnonzero(legal)
    policy = np.zeros_like(advantages, dtype=np.float32)
    if len(legal_actions) == 0:
        return policy
    positive = np.where(legal, np.maximum(advantages, 0.0), 0.0).astype(np.float32)
    total = float(positive.sum())
    if total > epsilon:
        return positive / total
    if fallback == "uniform":
        policy[legal_actions] = 1.0 / float(len(legal_actions))
        return policy
    best = float(np.max(advantages[legal_actions]))
    best_actions = legal_actions[advantages[legal_actions] == best]
    policy[int(best_actions[0])] = 1.0
    return policy


class AdvantagePolicy:
    def __init__(
        self,
        networks: list[torch.nn.Module],
        *,
        device: torch.device,
        encoding: Any,
        epsilon: float,
        fallback: str,
    ) -> None:
        self.networks = networks
        self.device = device
        self.encoding = encoding
        self.epsilon = epsilon
        self.fallback = fallback

    def distribution(self, state: GameState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        player = int(state.current_player)
        legal = np.asarray(state.unified_legal_mask(), dtype=bool)
        info = encode_info_state(state, player, self.encoding)
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = self.networks[player](x).squeeze(0).detach().cpu().numpy()
        policy = _regret_matching(
            advantages,
            legal,
            epsilon=self.epsilon,
            fallback=self.fallback,
        )
        return np.flatnonzero(legal), policy, advantages

    def select_unified(self, state: GameState) -> int:
        legal_actions, policy, advantages = self.distribution(state)
        if len(legal_actions) == 0:
            raise RuntimeError("no legal action available")
        probs = policy[legal_actions]
        if float(probs.sum()) > 0.0:
            return int(legal_actions[int(np.argmax(probs))])
        return int(legal_actions[int(np.argmax(advantages[legal_actions]))])


def _load_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[Any, LostCitiesConfig, AdvantagePolicy, int]:
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    game_config = LostCitiesConfig(**payload["game_config"])
    action_size = int(payload["action_size"])
    networks = [
        DeepCFRMLP.from_config(int(payload["input_dim"]), action_size, cfg.network).to(device)
        for _ in range(2)
    ]
    for network, state_dict in zip(networks, payload["advantage_networks"], strict=True):
        network.load_state_dict(state_dict)
        network.eval()
    policy = AdvantagePolicy(
        networks,
        device=device,
        encoding=cfg.encoding,
        epsilon=cfg.traversal.regret_matching_epsilon,
        fallback=cfg.regret_matching.all_negative_fallback,
    )
    return cfg, game_config, policy, int(payload.get("iteration", -1))


def _rollout_value(
    state: GameState,
    *,
    policy_player: int,
    policy: AdvantagePolicy,
    opponent: str,
    seed: int,
    max_steps: int,
    post_policy: str = "model",
) -> float:
    """Roll out from `state` to terminal and return policy_player's score diff.

    When `post_policy == "model"`, policy_player uses the trained advantage
    network policy. Otherwise `post_policy` is treated as a bot name and a
    fresh bot is built for policy_player too — used to diagnose whether the
    self-play rollout itself is poisoning forced-open continuation values.
    """
    rollout = state.clone()
    opponent_policy = build_bot(opponent, seed=seed)
    post_policy_bot = build_bot(post_policy, seed=seed * 7 + 1) if post_policy != "model" else None
    steps = 0
    while not rollout.terminal and steps < max_steps:
        current = int(rollout.current_player)
        if current == policy_player:
            if post_policy_bot is not None:
                action = post_policy_bot.act(rollout)
            else:
                unified = policy.select_unified(rollout)
                action = rollout.from_unified_action(unified)
        else:
            action = opponent_policy.act(rollout)
        rollout.apply_action(action)
        steps += 1
    return float(rollout.score_diff(policy_player))


def _best_non_open_action(
    labels: dict[int, str],
    legal_actions: np.ndarray,
    policy_probs: np.ndarray,
    advantages: np.ndarray,
) -> int | None:
    candidates = [
        int(action) for action in legal_actions if not labels[int(action)].startswith("open_")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda action: (float(policy_probs[action]), float(advantages[action]), -action),
    )


def analyze_checkpoint(
    checkpoint: Path,
    *,
    games: int,
    seed: int,
    opponent: str,
    device: torch.device,
    max_steps: int,
    max_candidates: int,
    post_policy: str = "model",
) -> dict[str, Any]:
    _cfg, game_config, policy, iteration = _load_checkpoint(checkpoint, device)
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    candidate_states = 0
    first_open_candidates = 0
    evaluated_open_candidates = 0
    policy_turns = 0
    started = time.perf_counter()

    for game_index in range(games):
        if evaluated_open_candidates >= max_candidates:
            break
        game_seed = seed + game_index
        swap = game_index % 2 == 1
        policy_player = 1 if swap else 0
        opponent_policy = build_bot(opponent, seed=game_seed * 2 + (1 - policy_player))
        state = GameState.new_game(game_config, seed=game_seed)
        for _step in range(max_steps):
            if state.terminal or evaluated_open_candidates >= max_candidates:
                break
            current = int(state.current_player)
            if current != policy_player:
                state.apply_action(opponent_policy.act(state))
                continue
            policy_turns += 1
            legal_actions, policy_probs, advantages = policy.distribution(state)
            labels = {
                int(action): _classify_action(state, int(action), current)
                for action in legal_actions
            }
            open_actions = [
                int(action) for action, label in labels.items() if label.startswith("open_")
            ]
            best_non_open = _best_non_open_action(labels, legal_actions, policy_probs, advantages)
            selected = policy.select_unified(state)
            if open_actions and best_non_open is not None:
                candidate_states += 1
                baseline_state = state.clone()
                baseline_state.apply_action(baseline_state.from_unified_action(best_non_open))
                baseline_value = _rollout_value(
                    baseline_state,
                    policy_player=policy_player,
                    policy=policy,
                    opponent=opponent,
                    seed=game_seed * 10_000 + candidate_states * 101 + 1,
                    max_steps=max_steps,
                    post_policy=post_policy,
                )
                for open_action in open_actions:
                    if evaluated_open_candidates >= max_candidates:
                        break
                    open_state = state.clone()
                    open_state.apply_action(open_state.from_unified_action(open_action))
                    open_value = _rollout_value(
                        open_state,
                        policy_player=policy_player,
                        policy=policy,
                        opponent=opponent,
                        seed=game_seed * 10_000 + candidate_states * 101 + 2,
                        max_steps=max_steps,
                        post_policy=post_policy,
                    )
                    label = labels[open_action]
                    buckets[label].add(
                        delta=open_value - baseline_value,
                        open_value=open_value,
                        baseline_value=baseline_value,
                        policy_prob=float(policy_probs[open_action]),
                        selected=open_action == selected,
                    )
                    first_open_candidates += 1
                    evaluated_open_candidates += 1
            state.apply_action(state.from_unified_action(selected))

    return {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "opponent": opponent,
        "games": games,
        "seed": seed,
        "device": str(device),
        "max_candidates": max_candidates,
        "elapsed_seconds": time.perf_counter() - started,
        "policy_turns": policy_turns,
        "candidate_states": candidate_states,
        "first_open_candidates": first_open_candidates,
        "post_policy": post_policy,
        "buckets": {key: bucket.to_dict() for key, bucket in sorted(buckets.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--opponent", default="safe_heuristic_strict")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=231_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument(
        "--post-policy",
        default="model",
        help=(
            "Policy used for the policy_player during forced-action rollouts. "
            "'model' uses the trained advantage network; any other value is "
            "treated as a bot name (e.g. 'safe_heuristic_strict')."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in args.checkpoints:
        row = analyze_checkpoint(
            checkpoint,
            games=args.games,
            seed=args.seed,
            opponent=args.opponent,
            device=device,
            max_steps=args.max_steps,
            max_candidates=args.max_candidates,
            post_policy=args.post_policy,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
