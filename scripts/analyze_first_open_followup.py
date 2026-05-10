#!/usr/bin/env python
"""Inspect model's post-forced-open behavior over the next K policy_player turns.

Diagnoses *why* forced-open continuation values are poor under self-play
rollouts: does the model play followup cards of the opened color, discard
them, or open another color instead?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_first_open_counterfactual import (  # noqa: E402
    AdvantagePolicy,
    _classify_action,
)


@dataclass
class FollowupBucket:
    candidates: int = 0
    same_color_plays: list[int] = field(default_factory=list)
    same_color_discards: list[int] = field(default_factory=list)
    other_open_new: list[int] = field(default_factory=list)
    other_play_existing: list[int] = field(default_factory=list)
    other_discard: list[int] = field(default_factory=list)
    draw_deck: list[int] = field(default_factory=list)
    draw_pile: list[int] = field(default_factory=list)
    same_color_held_at_force: list[int] = field(default_factory=list)
    same_color_played_in_window: list[int] = field(default_factory=list)
    same_color_discarded_in_window: list[int] = field(default_factory=list)
    final_score_diff: list[float] = field(default_factory=list)
    same_color_held_terminal: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def stats(values: list[float] | list[int]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "median": 0.0}
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }

        return {
            "candidates": self.candidates,
            "same_color_plays": stats(self.same_color_plays),
            "same_color_discards": stats(self.same_color_discards),
            "other_open_new": stats(self.other_open_new),
            "other_play_existing": stats(self.other_play_existing),
            "other_discard": stats(self.other_discard),
            "draw_deck": stats(self.draw_deck),
            "draw_pile": stats(self.draw_pile),
            "same_color_held_at_force": stats(self.same_color_held_at_force),
            "same_color_played_in_window": stats(self.same_color_played_in_window),
            "same_color_discarded_in_window": stats(self.same_color_discarded_in_window),
            "same_color_held_terminal": stats(self.same_color_held_terminal),
            "final_score_diff": stats(self.final_score_diff),
        }


def _load_checkpoint(
    checkpoint: Path, device: torch.device
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


def _count_color_cards_in_hand(state: GameState, player: int, color: int) -> int:
    return sum(
        1 for card in state.hand_slots(player) if card is not None and int(card.color) == color
    )


def _label_followup_action(
    state: GameState, unified_action: int, player: int, forced_color: int
) -> str:
    """Categorise a followup action relative to the previously forced-open color."""
    card_action_size = state.config.hand_size * 2
    if unified_action == card_action_size:
        return "draw_deck"
    if unified_action > card_action_size:
        return "draw_pile"
    if unified_action % 2 == 1:
        slot = unified_action // 2
        card = state.hand_slots(player)[slot]
        if card is None:
            return "discard_invalid"
        return "discard_same" if int(card.color) == forced_color else "discard_other"
    slot = unified_action // 2
    card = state.hand_slots(player)[slot]
    if card is None:
        return "play_invalid"
    color = int(card.color)
    is_open_action = not state.expeditions[player][color]
    if color == forced_color:
        return "play_same"
    return "open_other" if is_open_action else "play_other_existing"


def _force_and_observe(
    base_state: GameState,
    *,
    forced_action: int,
    forced_color: int,
    bucket: FollowupBucket,
    policy: AdvantagePolicy,
    opponent_policy: Any,
    policy_player: int,
    window: int,
    max_steps: int,
) -> None:
    rollout = base_state.clone()
    same_at_force = _count_color_cards_in_hand(rollout, policy_player, forced_color)
    rollout.apply_action(rollout.from_unified_action(forced_action))
    counts: Counter[str] = Counter()
    policy_turns_seen = 0
    same_color_plays = 0
    same_color_discards = 0
    steps = 0
    while not rollout.terminal and steps < max_steps and policy_turns_seen < window:
        current = int(rollout.current_player)
        if current != policy_player:
            rollout.apply_action(opponent_policy.act(rollout))
            steps += 1
            continue
        unified = policy.select_unified(rollout)
        label = _label_followup_action(rollout, unified, policy_player, forced_color)
        counts[label] += 1
        if label == "play_same":
            same_color_plays += 1
        elif label == "discard_same":
            same_color_discards += 1
        rollout.apply_action(rollout.from_unified_action(unified))
        steps += 1
        policy_turns_seen += 1
    while not rollout.terminal and steps < max_steps:
        current = int(rollout.current_player)
        if current == policy_player:
            unified = policy.select_unified(rollout)
            rollout.apply_action(rollout.from_unified_action(unified))
        else:
            rollout.apply_action(opponent_policy.act(rollout))
        steps += 1

    bucket.candidates += 1
    bucket.same_color_plays.append(counts.get("play_same", 0))
    bucket.same_color_discards.append(counts.get("discard_same", 0))
    bucket.other_open_new.append(counts.get("open_other", 0))
    bucket.other_play_existing.append(counts.get("play_other_existing", 0))
    bucket.other_discard.append(counts.get("discard_other", 0))
    bucket.draw_deck.append(counts.get("draw_deck", 0))
    bucket.draw_pile.append(counts.get("draw_pile", 0))
    bucket.same_color_held_at_force.append(same_at_force)
    bucket.same_color_played_in_window.append(same_color_plays)
    bucket.same_color_discarded_in_window.append(same_color_discards)
    bucket.same_color_held_terminal.append(
        _count_color_cards_in_hand(rollout, policy_player, forced_color)
    )
    bucket.final_score_diff.append(float(rollout.score_diff(policy_player)))


def analyze_checkpoint(
    checkpoint: Path,
    *,
    games: int,
    seed: int,
    opponent: str,
    device: torch.device,
    max_steps: int,
    max_candidates: int,
    window: int,
) -> dict[str, Any]:
    _cfg, game_config, policy, iteration = _load_checkpoint(checkpoint, device)
    buckets: dict[str, FollowupBucket] = defaultdict(FollowupBucket)
    candidates_evaluated = 0
    started = time.perf_counter()

    for game_index in range(games):
        if candidates_evaluated >= max_candidates:
            break
        game_seed = seed + game_index
        swap = game_index % 2 == 1
        policy_player = 1 if swap else 0
        opponent_policy = build_bot(opponent, seed=game_seed * 2 + (1 - policy_player))
        state = GameState.new_game(game_config, seed=game_seed)
        for _step in range(max_steps):
            if state.terminal or candidates_evaluated >= max_candidates:
                break
            current = int(state.current_player)
            if current != policy_player:
                state.apply_action(opponent_policy.act(state))
                continue
            legal_actions, _policy_probs, _advantages = policy.distribution(state)
            labels = {
                int(action): _classify_action(state, int(action), current)
                for action in legal_actions
            }
            open_actions = [
                int(action) for action, label in labels.items() if label.startswith("open_")
            ]
            if open_actions:
                for open_action in open_actions:
                    if candidates_evaluated >= max_candidates:
                        break
                    color = int(state.hand_slots(current)[open_action // 2].color)
                    bucket = buckets[labels[open_action]]
                    _force_and_observe(
                        state,
                        forced_action=open_action,
                        forced_color=color,
                        bucket=bucket,
                        policy=policy,
                        opponent_policy=opponent_policy,
                        policy_player=policy_player,
                        window=window,
                        max_steps=max_steps,
                    )
                    candidates_evaluated += 1
            unified = policy.select_unified(state)
            state.apply_action(state.from_unified_action(unified))

    return {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "opponent": opponent,
        "games": games,
        "seed": seed,
        "device": str(device),
        "max_candidates": max_candidates,
        "window": window,
        "elapsed_seconds": time.perf_counter() - started,
        "candidates_evaluated": candidates_evaluated,
        "buckets": {key: bucket.to_dict() for key, bucket in sorted(buckets.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--opponent", default="safe_heuristic_strict")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=232_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Number of policy_player turns to observe after the forced open.",
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
            window=args.window,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
