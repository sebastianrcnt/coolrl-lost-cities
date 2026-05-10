#!/usr/bin/env python
"""Compare Deep CFR current regret-matching policy vs average strategy policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.config import config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.evaluate import (
    EvalRuntimeCounters,
    PolicyEvalDiagnostics,
    StrategyNetPolicy,
    _accumulate_game_diagnostics,
    _advance_opponent_turn,
    _advance_policy_turn,
    _EvalGame,
    _finalize_if_done,
)
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.policy import LostCitiesPolicy, PolicyInput


class CurrentRegretPolicy(LostCitiesPolicy):
    def __init__(
        self,
        advantage_networks: list[torch.nn.Module],
        *,
        device: torch.device | str,
        encoding,
        epsilon: float,
        all_negative_fallback: str,
    ) -> None:
        self.advantage_networks = advantage_networks
        self.device = torch.device(device)
        self.encoding = encoding
        self.epsilon = float(epsilon)
        self.all_negative_fallback = all_negative_fallback
        self.runtime = EvalRuntimeCounters()

    def select_actions_batch(self, states: list[GameState]) -> list[tuple[int, float]]:
        return [self.select_action(state) for state in states]

    def select_action(self, state: GameState) -> tuple[int, float]:
        started = time.perf_counter()
        self.runtime.policy_turns += 1
        legal_actions, distribution = self.action_distribution(state)
        probs = distribution[legal_actions]
        entropy = _entropy(probs)
        unified = int(legal_actions[int(np.argmax(probs))])
        self.runtime.policy_select_seconds += time.perf_counter() - started
        return state.from_unified_action(unified), entropy

    def action_distribution(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        legal = np.asarray(state.unified_legal_mask(), dtype=bool)
        legal_actions = np.flatnonzero(legal)
        self.runtime.policy_legal_mask_seconds += time.perf_counter() - started
        if len(legal_actions) == 0:
            raise RuntimeError("no legal action available")

        started = time.perf_counter()
        info = encode_info_state(state, state.current_player, self.encoding)
        self.runtime.policy_encoding_seconds += time.perf_counter() - started

        started = time.perf_counter()
        network = self.advantage_networks[int(state.current_player)]
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = network(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        self.runtime.policy_network_seconds += time.perf_counter() - started

        started = time.perf_counter()
        distribution = self._regret_matching_distribution(advantages, legal, legal_actions)
        self.runtime.policy_postprocess_seconds += time.perf_counter() - started
        return legal_actions, distribution

    def _regret_matching_distribution(
        self,
        advantages: np.ndarray,
        legal: np.ndarray,
        legal_actions: np.ndarray,
    ) -> np.ndarray:
        positive = np.where(legal, np.maximum(advantages, 0.0), 0.0).astype(np.float32)
        total = float(positive.sum())
        if total > self.epsilon:
            return positive / total
        distribution = np.zeros_like(advantages, dtype=np.float32)
        if self.all_negative_fallback == "uniform":
            distribution[legal_actions] = 1.0 / float(len(legal_actions))
            return distribution
        best = float(np.max(advantages[legal_actions]))
        best_actions = legal_actions[advantages[legal_actions] == best]
        distribution[int(best_actions[0])] = 1.0
        return distribution

    def act(self, obs_or_state: PolicyInput) -> int:
        if not isinstance(obs_or_state, GameState):
            legal = np.asarray(obs_or_state["legal_mask"], dtype=bool)
            legal_actions = np.flatnonzero(legal)
            if len(legal_actions) == 0:
                raise RuntimeError("no legal action available")
            return int(legal_actions[0])
        action, _entropy_value = self.select_action(obs_or_state)
        return action


def evaluate_policy(
    policy: CurrentRegretPolicy | StrategyNetPolicy,
    config: LostCitiesConfig,
    *,
    games: int,
    seed: int,
    opponent: str,
    max_steps: int,
    batch_size: int,
) -> dict[str, float | int]:
    diagnostics = PolicyEvalDiagnostics()
    started = time.perf_counter()
    active_games: list[_EvalGame] = []
    for index in range(games):
        game_seed = seed + index
        swap = index % 2 == 1
        policy_player = 1 if swap else 0
        opponent_policy = build_bot(opponent, seed=game_seed * 2 + (1 - policy_player))
        policies = [opponent_policy, policy] if swap else [policy, opponent_policy]
        active_games.append(
            _EvalGame(
                state=GameState.new_game(config, seed=game_seed),
                policies=policies,
                policy_player=policy_player,
                diagnostics=PolicyEvalDiagnostics(games=1),
                first_open_recoverable_by_color={},
                game_index=index,
                game_seed=game_seed,
            )
        )

    while active_games:
        pending_policy_games: list[_EvalGame] = []
        next_active_games: list[_EvalGame] = []
        for game in active_games:
            if _finalize_if_done(game, max_steps=max_steps):
                _accumulate_game_diagnostics(diagnostics, game.diagnostics)
                continue
            if game.state.current_player == game.policy_player:
                pending_policy_games.append(game)
            else:
                _advance_opponent_turn(game)
                if _finalize_if_done(game, max_steps=max_steps):
                    _accumulate_game_diagnostics(diagnostics, game.diagnostics)
                else:
                    next_active_games.append(game)

        chunk_size = max(1, int(batch_size))
        for start in range(0, len(pending_policy_games), chunk_size):
            chunk = pending_policy_games[start : start + chunk_size]
            actions = policy.select_actions_batch([game.state for game in chunk])
            for game, (action, entropy) in zip(chunk, actions, strict=True):
                _advance_policy_turn(game, action, entropy)
                if _finalize_if_done(game, max_steps=max_steps):
                    _accumulate_game_diagnostics(diagnostics, game.diagnostics)
                else:
                    next_active_games.append(game)

        active_games = next_active_games

    diagnostics.runtime.accumulate(policy.runtime)
    return diagnostics.to_dict(time.perf_counter() - started)


def load_policies(
    checkpoint: Path,
    *,
    device: torch.device | str,
) -> tuple[StrategyNetPolicy, CurrentRegretPolicy, LostCitiesConfig, dict]:
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    game_config = LostCitiesConfig(**payload["game_config"])
    input_dim = int(payload["input_dim"])
    action_size = int(payload["action_size"])

    strategy_network = DeepCFRMLP.from_config(input_dim, action_size, cfg.network).to(device)
    strategy_network.load_state_dict(payload["strategy_network"])
    strategy_network.eval()

    advantage_networks = [
        DeepCFRMLP.from_config(input_dim, action_size, cfg.network).to(device) for _ in range(2)
    ]
    for network, state_dict in zip(advantage_networks, payload["advantage_networks"], strict=True):
        network.load_state_dict(state_dict)
        network.eval()

    average_policy = StrategyNetPolicy(
        strategy_network,
        device=device,
        seed=cfg.run.seed * 2,
        encoding=cfg.encoding,
    )
    current_policy = CurrentRegretPolicy(
        advantage_networks,
        device=device,
        encoding=cfg.encoding,
        epsilon=cfg.traversal.regret_matching_epsilon,
        all_negative_fallback=cfg.regret_matching.all_negative_fallback,
    )
    return average_policy, current_policy, game_config, payload


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0.0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _checkpoint_iteration(path: Path, payload: dict) -> int:
    if isinstance(payload.get("iteration"), int):
        return int(payload["iteration"])
    stem = path.stem
    if stem.startswith("iteration_"):
        return int(stem.removeprefix("iteration_"))
    return -1


def _select_metrics(result: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "win_rate0",
        "avg_score_diff0",
        "avg_opened_colors",
        "score_per_opened_color",
        "bad_open_rate",
        "good_open_rate",
        "negative_expedition_rate",
        "positive_expedition_rate",
        "policy_entropy",
        "play_action_rate",
        "discard_action_rate",
        "draw_deck_rate",
        "draw_pile_rate",
        "avg_game_length",
        "max_step_timeouts",
        "elapsed_seconds",
    )
    return {key: result[key] for key in keys if key in result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=79_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    opponents = args.opponent or ["random", "heuristic_cautious"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for checkpoint in args.checkpoints:
        average_policy, current_policy, game_config, payload = load_policies(
            checkpoint,
            device=args.device,
        )
        iteration = _checkpoint_iteration(checkpoint, payload)
        for policy_name, policy in (("average", average_policy), ("current", current_policy)):
            for opponent in opponents:
                result = evaluate_policy(
                    policy,
                    game_config,
                    games=args.games,
                    seed=args.seed + max(iteration, 0) * 1000,
                    opponent=opponent,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                )
                row = {
                    "checkpoint": str(checkpoint),
                    "iteration": iteration,
                    "policy": policy_name,
                    "opponent": opponent,
                    "games": args.games,
                    **_select_metrics(result),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
