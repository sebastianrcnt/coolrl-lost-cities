#!/usr/bin/env python
"""Analyze current-network advantages on first-open action candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP


@dataclass
class BucketStats:
    values: list[float] = field(default_factory=list)
    policy_probs: list[float] = field(default_factory=list)
    selected: int = 0

    def add(self, value: float, policy_prob: float, *, selected: bool) -> None:
        self.values.append(float(value))
        self.policy_probs.append(float(policy_prob))
        self.selected += int(selected)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": len(self.values),
            "adv_mean": _mean(self.values),
            "adv_p25": _percentile(self.values, 25),
            "adv_p50": _percentile(self.values, 50),
            "adv_p75": _percentile(self.values, 75),
            "policy_prob_mean": _mean(self.policy_probs),
            "selected": self.selected,
            "selected_rate": self.selected / max(1, len(self.values)),
        }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _numeric_value(card, min_rank: int) -> int:
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


def _regret_matching(
    advantages: np.ndarray,
    legal: np.ndarray,
    *,
    epsilon: float,
    fallback: str,
) -> np.ndarray:
    legal_actions = np.flatnonzero(legal)
    positive = np.where(legal, np.maximum(advantages, 0.0), 0.0).astype(np.float32)
    total = float(positive.sum())
    if total > epsilon:
        return positive / total
    policy = np.zeros_like(advantages, dtype=np.float32)
    if len(legal_actions) == 0:
        return policy
    if fallback == "uniform":
        policy[legal_actions] = 1.0 / float(len(legal_actions))
        return policy
    best = float(np.max(advantages[legal_actions]))
    best_actions = legal_actions[advantages[legal_actions] == best]
    policy[int(best_actions[0])] = 1.0
    return policy


def _classify_unified_action(state: GameState, unified_action: int, player: int) -> str:
    card_action_size = state.config.hand_size * 2
    if unified_action >= card_action_size:
        return "draw_deck" if unified_action == card_action_size else "draw_pile"
    if unified_action % 2 == 1:
        return "discard"
    hand = state.hand_slots(player)
    card = hand[unified_action // 2]
    if card is None:
        return "invalid_play"
    color = int(card.color)
    if state.expeditions[player][color]:
        return "play_existing"
    return _open_quality(state, player, color)


def _select_current_action(
    advantages: np.ndarray,
    legal: np.ndarray,
    *,
    epsilon: float,
    fallback: str,
) -> tuple[int, np.ndarray]:
    policy = _regret_matching(advantages, legal, epsilon=epsilon, fallback=fallback)
    legal_actions = np.flatnonzero(legal)
    if len(legal_actions) == 0:
        raise RuntimeError("no legal action available")
    return int(legal_actions[int(np.argmax(policy[legal_actions]))]), policy


def load_checkpoint_networks(checkpoint: Path, device: torch.device | str):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    game_config = LostCitiesConfig(**payload["game_config"])
    input_dim = int(payload["input_dim"])
    action_size = int(payload["action_size"])
    networks = [
        DeepCFRMLP.from_config(input_dim, action_size, cfg.network).to(device) for _ in range(2)
    ]
    for network, state_dict in zip(networks, payload["advantage_networks"], strict=True):
        network.load_state_dict(state_dict)
        network.eval()
    return cfg, game_config, networks


def analyze_checkpoint(
    checkpoint: Path,
    *,
    games: int,
    seed: int,
    opponent: str,
    device: torch.device | str,
    max_steps: int,
) -> dict:
    cfg, game_config, networks = load_checkpoint_networks(checkpoint, device)
    buckets: dict[str, BucketStats] = defaultdict(BucketStats)
    selected_buckets: dict[str, int] = defaultdict(int)
    candidate_states = 0
    first_open_candidates = 0
    policy_turns = 0

    for game_index in range(games):
        game_seed = seed + game_index
        swap = game_index % 2 == 1
        policy_player = 1 if swap else 0
        opponent_policy = build_bot(opponent, seed=game_seed * 2 + (1 - policy_player))
        state = GameState.new_game(game_config, seed=game_seed)
        for _step in range(max_steps):
            if state.terminal:
                break
            player = int(state.current_player)
            if player != policy_player:
                state.apply_action(opponent_policy.act(state))
                continue
            policy_turns += 1
            legal = np.asarray(state.unified_legal_mask(), dtype=bool)
            info = encode_info_state(state, player, cfg.encoding)
            with torch.inference_mode():
                x = torch.as_tensor(info, dtype=torch.float32, device=device).unsqueeze(0)
                advantages = networks[player](x).squeeze(0).detach().cpu().numpy()
            selected_action, policy = _select_current_action(
                advantages,
                legal,
                epsilon=cfg.traversal.regret_matching_epsilon,
                fallback=cfg.regret_matching.all_negative_fallback,
            )
            labels = {
                int(action): _classify_unified_action(state, int(action), player)
                for action in np.flatnonzero(legal)
            }
            open_actions = [action for action, label in labels.items() if label.startswith("open_")]
            if open_actions:
                candidate_states += 1
                first_open_candidates += len(open_actions)
                selected_buckets[labels[selected_action]] += 1
                for action in open_actions:
                    label = labels[action]
                    buckets[label].add(
                        float(advantages[action]),
                        float(policy[action]),
                        selected=action == selected_action,
                    )
                non_open_advantages = [
                    float(advantages[action])
                    for action, label in labels.items()
                    if not label.startswith("open_")
                ]
                for value in non_open_advantages:
                    buckets["non_open"].add(value, 0.0, selected=False)
            state.apply_action(state.from_unified_action(selected_action))

    iteration = int(torch.load(checkpoint, map_location="cpu").get("iteration", -1))
    return {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "opponent": opponent,
        "games": games,
        "policy_turns": policy_turns,
        "candidate_states": candidate_states,
        "first_open_candidates": first_open_candidates,
        "selected_buckets": dict(sorted(selected_buckets.items())),
        "buckets": {key: value.to_dict() for key, value in sorted(buckets.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    opponents = args.opponent or ["heuristic_cautious"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in args.checkpoints:
        for opponent in opponents:
            row = analyze_checkpoint(
                checkpoint,
                games=args.games,
                seed=args.seed,
                opponent=opponent,
                device=args.device,
                max_steps=args.max_steps,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
