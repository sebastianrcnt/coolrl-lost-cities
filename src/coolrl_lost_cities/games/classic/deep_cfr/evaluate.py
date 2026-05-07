from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.config import EncodingConfig, config_from_dict
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig
from coolrl_lost_cities.games.classic.policy import LostCitiesPolicy, PolicyInput


@dataclass
class EvalRuntimeCounters:
    policy_turns: int = 0
    opponent_turns: int = 0
    policy_select_seconds: float = 0.0
    policy_legal_mask_seconds: float = 0.0
    policy_encoding_seconds: float = 0.0
    policy_network_seconds: float = 0.0
    policy_postprocess_seconds: float = 0.0
    opponent_act_seconds: float = 0.0
    apply_action_seconds: float = 0.0
    diagnostics_seconds: float = 0.0
    final_scoring_seconds: float = 0.0

    def accumulate(self, other: EvalRuntimeCounters) -> None:
        self.policy_turns += other.policy_turns
        self.opponent_turns += other.opponent_turns
        self.policy_select_seconds += other.policy_select_seconds
        self.policy_legal_mask_seconds += other.policy_legal_mask_seconds
        self.policy_encoding_seconds += other.policy_encoding_seconds
        self.policy_network_seconds += other.policy_network_seconds
        self.policy_postprocess_seconds += other.policy_postprocess_seconds
        self.opponent_act_seconds += other.opponent_act_seconds
        self.apply_action_seconds += other.apply_action_seconds
        self.diagnostics_seconds += other.diagnostics_seconds
        self.final_scoring_seconds += other.final_scoring_seconds

    def to_dict(self) -> dict[str, float | int]:
        return {
            "policy_turns": self.policy_turns,
            "opponent_turns": self.opponent_turns,
            "policy_select_seconds": self.policy_select_seconds,
            "policy_legal_mask_seconds": self.policy_legal_mask_seconds,
            "policy_encoding_seconds": self.policy_encoding_seconds,
            "policy_network_seconds": self.policy_network_seconds,
            "policy_postprocess_seconds": self.policy_postprocess_seconds,
            "opponent_act_seconds": self.opponent_act_seconds,
            "apply_action_seconds": self.apply_action_seconds,
            "diagnostics_seconds": self.diagnostics_seconds,
            "final_scoring_seconds": self.final_scoring_seconds,
        }


@dataclass
class PolicyEvalDiagnostics:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score: list[int] = field(default_factory=list)
    opponent_score: list[int] = field(default_factory=list)
    diff: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    timeouts: int = 0
    policy_actions: int = 0
    play_actions: int = 0
    discard_actions: int = 0
    draw_deck_actions: int = 0
    draw_pile_actions: int = 0
    entropies: list[float] = field(default_factory=list)
    opened_colors: list[int] = field(default_factory=list)
    five_color_open_count: int = 0
    expedition_cards: list[int] = field(default_factory=list)
    opening_counts: list[int] = field(default_factory=list)
    bad_open_counts: list[int] = field(default_factory=list)
    weak_open_counts: list[int] = field(default_factory=list)
    good_open_counts: list[int] = field(default_factory=list)
    opening_recoverable_scores: list[float] = field(default_factory=list)
    score_per_opened_color: list[float] = field(default_factory=list)
    positive_expeditions: list[int] = field(default_factory=list)
    negative_expeditions: list[int] = field(default_factory=list)
    breakeven_expeditions: list[int] = field(default_factory=list)
    bonus_expeditions: list[int] = field(default_factory=list)
    below_minus_20_expeditions: list[int] = field(default_factory=list)
    final_expedition_scores: list[int] = field(default_factory=list)
    positive_expedition_scores: list[int] = field(default_factory=list)
    negative_expedition_scores: list[int] = field(default_factory=list)
    first_open_positive_recoverable_scores: list[float] = field(default_factory=list)
    first_open_negative_recoverable_scores: list[float] = field(default_factory=list)
    runtime: EvalRuntimeCounters = field(default_factory=EvalRuntimeCounters)

    def to_dict(self, elapsed_seconds: float) -> dict[str, float | int]:
        games = max(1, self.games)
        total_steps = sum(self.lengths)
        opened_expeditions = len(self.final_expedition_scores)
        total_policy_actions = max(1, self.policy_actions)
        data: dict[str, float | int] = {
            "games": self.games,
            "wins0": self.wins,
            "wins1": self.losses,
            "draws": self.draws,
            "win_rate0": self.wins / games,
            "win_rate1": self.losses / games,
            "avg_score0": _mean(self.score),
            "avg_score1": _mean(self.opponent_score),
            "avg_score_diff0": _mean(self.diff),
            "avg_game_length": _mean(self.lengths),
            "max_step_timeouts": self.timeouts,
            "elapsed_seconds": elapsed_seconds,
            "games_per_second": self.games / max(elapsed_seconds, 1.0e-12),
            "steps_per_second": total_steps / max(elapsed_seconds, 1.0e-12),
            "play_action_rate": self.play_actions / total_policy_actions,
            "discard_action_rate": self.discard_actions / total_policy_actions,
            "draw_deck_rate": self.draw_deck_actions / total_policy_actions,
            "draw_pile_rate": self.draw_pile_actions / total_policy_actions,
            "policy_entropy": _mean(self.entropies),
            "avg_opened_colors": _mean(self.opened_colors),
            "5_color_open_count": self.five_color_open_count,
            "avg_expedition_cards": _mean(self.expedition_cards),
            "opening_play_actions": _mean(self.opening_counts),
            "bad_open_actions": _mean(self.bad_open_counts),
            "weak_open_actions": _mean(self.weak_open_counts),
            "good_open_actions": _mean(self.good_open_counts),
            "bad_open_rate": sum(self.bad_open_counts) / max(1, sum(self.opening_counts)),
            "weak_open_rate": sum(self.weak_open_counts) / max(1, sum(self.opening_counts)),
            "good_open_rate": sum(self.good_open_counts) / max(1, sum(self.opening_counts)),
            "opening_recoverable_score_mean": _mean(self.opening_recoverable_scores),
            "score_per_opened_color": _mean(self.score_per_opened_color),
            "per_game_positive_expeditions": _mean(self.positive_expeditions),
            "per_game_negative_expeditions": _mean(self.negative_expeditions),
            "per_game_breakeven_expeditions": _mean(self.breakeven_expeditions),
            "per_game_bonus_expeditions": _mean(self.bonus_expeditions),
            "per_game_below_minus_20_expeditions": _mean(self.below_minus_20_expeditions),
            "positive_expedition_rate": sum(self.positive_expeditions) / max(1, opened_expeditions),
            "negative_expedition_rate": sum(self.negative_expeditions) / max(1, opened_expeditions),
            "bonus_expedition_rate": sum(self.bonus_expeditions) / max(1, opened_expeditions),
            "avg_final_score_per_opened_expedition": _mean(self.final_expedition_scores),
            "final_expedition_score_p25": _percentile(self.final_expedition_scores, 25),
            "final_expedition_score_median": _percentile(self.final_expedition_scores, 50),
            "final_expedition_score_p75": _percentile(self.final_expedition_scores, 75),
            "final_expedition_score_p90": _percentile(self.final_expedition_scores, 90),
            "positive_expedition_score_mean": _mean(self.positive_expedition_scores),
            "negative_expedition_score_mean": _mean(self.negative_expedition_scores),
            "first_open_recoverable_score_mean_for_positive_final": _mean(
                self.first_open_positive_recoverable_scores
            ),
            "first_open_recoverable_score_mean_for_negative_final": _mean(
                self.first_open_negative_recoverable_scores
            ),
        }
        data.update(self.runtime.to_dict())
        return data


@dataclass
class _EvalGame:
    state: GameState
    policies: list[LostCitiesPolicy]
    policy_player: int
    diagnostics: PolicyEvalDiagnostics
    first_open_recoverable_by_color: dict[int, float]
    game_index: int = 0
    game_seed: int = 0
    steps: int = 0
    done: bool = False


class StrategyNetPolicy(LostCitiesPolicy):
    def __init__(
        self,
        strategy_network: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        sample: bool = False,
        seed: int | None = None,
        encoding: EncodingConfig | None = None,
    ) -> None:
        self.strategy_network = strategy_network
        self.device = torch.device(device)
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.encoding = encoding
        self.runtime = EvalRuntimeCounters()

    def select_actions_batch(self, states: list[GameState]) -> list[tuple[int, float]]:
        if not states:
            return []
        started = time.perf_counter()
        self.runtime.policy_turns += len(states)

        legal_started = time.perf_counter()
        legal_masks = [np.asarray(state.unified_legal_mask(), dtype=bool) for state in states]
        legal_actions_list = [np.flatnonzero(legal) for legal in legal_masks]
        self.runtime.policy_legal_mask_seconds += time.perf_counter() - legal_started
        if any(len(legal_actions) == 0 for legal_actions in legal_actions_list):
            raise RuntimeError("no legal action available")

        encoding_started = time.perf_counter()
        infos = [encode_info_state(state, state.current_player, self.encoding) for state in states]
        self.runtime.policy_encoding_seconds += time.perf_counter() - encoding_started

        network_started = time.perf_counter()
        with torch.inference_mode():
            x = torch.as_tensor(np.stack(infos), dtype=torch.float32, device=self.device)
            logits = self.strategy_network(x)
        self.runtime.policy_network_seconds += time.perf_counter() - network_started

        postprocess_started = time.perf_counter()
        legal_tensor = torch.as_tensor(np.stack(legal_masks), dtype=torch.bool, device=self.device)
        masked = logits.masked_fill(~legal_tensor, torch.finfo(torch.float32).min)
        probs_tensor = torch.softmax(masked, dim=-1).masked_fill(~legal_tensor, 0.0)
        entropy_tensor = -(probs_tensor * probs_tensor.clamp_min(1.0e-12).log()).sum(dim=-1)
        if self.sample:
            probs_np = probs_tensor.detach().cpu().numpy()
            unified_actions = [
                int(self.rng.choice(legal_actions, p=probs_np[index][legal_actions]))
                for index, legal_actions in enumerate(legal_actions_list)
            ]
        else:
            unified_actions = [int(value) for value in torch.argmax(masked, dim=-1).cpu()]
        entropies = [float(value) for value in entropy_tensor.cpu()]
        actions = [
            state.from_unified_action(unified)
            for state, unified in zip(states, unified_actions, strict=True)
        ]
        self.runtime.policy_postprocess_seconds += time.perf_counter() - postprocess_started
        self.runtime.policy_select_seconds += time.perf_counter() - started
        return list(zip(actions, entropies, strict=True))

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
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_network(x).squeeze(0).detach().cpu().numpy()
        self.runtime.policy_network_seconds += time.perf_counter() - started
        started = time.perf_counter()
        masked = np.where(legal, logits, -np.inf)
        stable = masked[legal_actions] - np.max(masked[legal_actions])
        probs = np.exp(stable)
        probs = probs / probs.sum()
        distribution = np.zeros_like(masked, dtype=np.float32)
        distribution[legal_actions] = probs.astype(np.float32)
        self.runtime.policy_postprocess_seconds += time.perf_counter() - started
        return legal_actions, distribution

    def select_action(self, state: GameState) -> tuple[int, float]:
        started = time.perf_counter()
        self.runtime.policy_turns += 1
        legal_actions, distribution = self.action_distribution(state)
        probs = distribution[legal_actions]
        entropy = _entropy(probs)
        if self.sample:
            unified = int(self.rng.choice(legal_actions, p=probs))
        else:
            unified = int(legal_actions[int(np.argmax(probs))])
        action = state.from_unified_action(unified)
        self.runtime.policy_select_seconds += time.perf_counter() - started
        return action, entropy

    def act(self, obs_or_state: PolicyInput) -> int:
        if not isinstance(obs_or_state, GameState):
            legal = np.asarray(obs_or_state["legal_mask"], dtype=bool)
            legal_actions = np.flatnonzero(legal)
            if len(legal_actions) == 0:
                raise RuntimeError("no legal action available")
            return int(legal_actions[0])
        action, _entropy_value = self.select_action(obs_or_state)
        return action


def evaluate_strategy_network(
    strategy_network: torch.nn.Module,
    config: LostCitiesConfig,
    *,
    games: int,
    seed: int,
    opponent: str = "random",
    device: torch.device | str = "cpu",
    max_steps: int = 10_000,
    encoding: EncodingConfig | None = None,
    batch_size: int = 64,
    save_games_path: str | None = None,
) -> dict[str, float | int]:
    strategy_network.eval()
    return _evaluate_strategy_network_with_diagnostics(
        strategy_network,
        config,
        games=games,
        seed=seed,
        opponent=opponent,
        device=device,
        max_steps=max_steps,
        encoding=encoding,
        batch_size=batch_size,
        save_games_path=save_games_path,
    )


def _evaluate_strategy_network_with_diagnostics(
    strategy_network: torch.nn.Module,
    config: LostCitiesConfig,
    *,
    games: int,
    seed: int,
    opponent: str,
    device: torch.device | str,
    max_steps: int,
    encoding: EncodingConfig | None,
    batch_size: int,
    save_games_path: str | None = None,
) -> dict[str, float | int]:
    if games <= 0:
        raise ValueError(f"games must be positive, got {games}")
    diagnostics = PolicyEvalDiagnostics()
    game_records: list[dict] = [] if save_games_path else None
    policy = StrategyNetPolicy(
        strategy_network,
        device=device,
        seed=seed * 2,
        encoding=encoding,
    )
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
                _record_game_if_needed(game, game_records, opponent)
                continue
            state = game.state
            if state.current_player == game.policy_player and isinstance(
                game.policies[state.current_player], StrategyNetPolicy
            ):
                pending_policy_games.append(game)
            else:
                _advance_opponent_turn(game)
                if _finalize_if_done(game, max_steps=max_steps):
                    _accumulate_game_diagnostics(diagnostics, game.diagnostics)
                    _record_game_if_needed(game, game_records, opponent)
                else:
                    next_active_games.append(game)

        for start in range(0, len(pending_policy_games), max(1, int(batch_size))):
            chunk = pending_policy_games[start : start + max(1, int(batch_size))]
            actions = policy.select_actions_batch([game.state for game in chunk])
            for game, (action, entropy) in zip(chunk, actions, strict=True):
                _advance_policy_turn(game, action, entropy)
                if _finalize_if_done(game, max_steps=max_steps):
                    _accumulate_game_diagnostics(diagnostics, game.diagnostics)
                    _record_game_if_needed(game, game_records, opponent)
                else:
                    next_active_games.append(game)

        active_games = next_active_games

    if game_records is not None:
        Path(save_games_path).write_text(json.dumps(game_records, indent=2))

    diagnostics.runtime.accumulate(policy.runtime)
    return diagnostics.to_dict(time.perf_counter() - started)


def _finalize_if_done(game: _EvalGame, *, max_steps: int) -> bool:
    if game.done:
        return True
    if not game.state.terminal and game.steps < max_steps:
        return False
    timed_out = not game.state.terminal
    if timed_out:
        game.steps = max_steps
    final_started = time.perf_counter()
    _record_final_game_state(
        game.diagnostics,
        game.state,
        game.policy_player,
        game.steps,
        timed_out,
        game.first_open_recoverable_by_color,
    )
    game.diagnostics.runtime.final_scoring_seconds += time.perf_counter() - final_started
    game.done = True
    return True


def _advance_policy_turn(game: _EvalGame, action: int, entropy: float) -> None:
    game.diagnostics.entropies.append(entropy)
    diagnostics_started = time.perf_counter()
    _record_policy_action(
        game.diagnostics,
        game.state,
        action,
        game.first_open_recoverable_by_color,
    )
    game.diagnostics.runtime.diagnostics_seconds += time.perf_counter() - diagnostics_started
    apply_started = time.perf_counter()
    game.state.apply_action(action)
    game.diagnostics.runtime.apply_action_seconds += time.perf_counter() - apply_started
    game.steps += 1


def _advance_opponent_turn(game: _EvalGame) -> None:
    state = game.state
    game.diagnostics.runtime.opponent_turns += 1
    opponent_started = time.perf_counter()
    action = game.policies[state.current_player].act(state)
    game.diagnostics.runtime.opponent_act_seconds += time.perf_counter() - opponent_started
    apply_started = time.perf_counter()
    state.apply_action(action)
    game.diagnostics.runtime.apply_action_seconds += time.perf_counter() - apply_started
    game.steps += 1


def _evaluate_one_game(
    policies: list[LostCitiesPolicy],
    policy_player: int,
    config: LostCitiesConfig,
    *,
    seed: int,
    max_steps: int,
) -> PolicyEvalDiagnostics:
    state = GameState.new_game(config, seed=seed)
    diagnostics = PolicyEvalDiagnostics(games=1)
    first_open_recoverable_by_color: dict[int, float] = {}
    steps = 0
    for _ in range(max_steps):
        if state.terminal:
            break
        current_player = state.current_player
        policy = policies[current_player]
        if current_player == policy_player and isinstance(policy, StrategyNetPolicy):
            action, entropy = policy.select_action(state)
            diagnostics.entropies.append(entropy)
            diagnostics_started = time.perf_counter()
            _record_policy_action(diagnostics, state, action, first_open_recoverable_by_color)
            diagnostics.runtime.diagnostics_seconds += time.perf_counter() - diagnostics_started
        else:
            diagnostics.runtime.opponent_turns += 1
            opponent_started = time.perf_counter()
            action = policy.act(state)
            diagnostics.runtime.opponent_act_seconds += time.perf_counter() - opponent_started
        apply_started = time.perf_counter()
        state.apply_action(action)
        diagnostics.runtime.apply_action_seconds += time.perf_counter() - apply_started
        steps += 1

    timed_out = not state.terminal
    if timed_out:
        steps = max_steps
    final_started = time.perf_counter()
    _record_final_game_state(
        diagnostics,
        state,
        policy_player,
        steps,
        timed_out,
        first_open_recoverable_by_color,
    )
    diagnostics.runtime.final_scoring_seconds += time.perf_counter() - final_started
    return diagnostics


def _record_policy_action(
    diagnostics: PolicyEvalDiagnostics,
    state: GameState,
    action: int,
    first_open_recoverable_by_color: dict[int, float],
) -> None:
    diagnostics.policy_actions += 1
    if state.phase == "draw":
        if action == 0:
            diagnostics.draw_deck_actions += 1
        else:
            diagnostics.draw_pile_actions += 1
        return

    slot = action // 2
    play = action % 2 == 0
    if play:
        diagnostics.play_actions += 1
    else:
        diagnostics.discard_actions += 1
        return

    hand = state.hand_slots(state.current_player)
    if slot >= len(hand) or hand[slot] is None:
        return
    card = hand[slot]
    color = int(card.color)
    if state.expeditions[state.current_player][color]:
        return

    summary = _visible_recoverable_summary(state, state.current_player, color)
    recoverable_score = float(summary["recoverable_score"])
    has_bonus_path = bool(summary["has_bonus_path"])
    diagnostics.opening_recoverable_scores.append(recoverable_score)
    if color not in first_open_recoverable_by_color:
        first_open_recoverable_by_color[color] = recoverable_score
    if recoverable_score >= 0:
        diagnostics.good_open_counts.append(1)
        diagnostics.bad_open_counts.append(0)
        diagnostics.weak_open_counts.append(0)
    elif has_bonus_path:
        diagnostics.good_open_counts.append(0)
        diagnostics.bad_open_counts.append(0)
        diagnostics.weak_open_counts.append(1)
    else:
        diagnostics.good_open_counts.append(0)
        diagnostics.bad_open_counts.append(1)
        diagnostics.weak_open_counts.append(0)
    diagnostics.opening_counts.append(1)


def _record_final_game_state(
    diagnostics: PolicyEvalDiagnostics,
    state: GameState,
    policy_player: int,
    steps: int,
    timed_out: bool,
    first_open_recoverable_by_color: dict[int, float],
) -> None:
    policy_score = state.total_score(policy_player)
    opponent_score = state.total_score(1 - policy_player)
    diff = policy_score - opponent_score
    diagnostics.score.append(policy_score)
    diagnostics.opponent_score.append(opponent_score)
    diagnostics.diff.append(diff)
    diagnostics.lengths.append(steps)
    diagnostics.timeouts += int(timed_out)
    if diff > 0:
        diagnostics.wins += 1
    elif diff < 0:
        diagnostics.losses += 1
    else:
        diagnostics.draws += 1

    opened = 0
    expedition_cards = 0
    positive = negative = breakeven = bonus = below_minus_20 = 0
    for color, expedition in enumerate(state.expeditions[policy_player]):
        if not expedition:
            continue
        opened += 1
        expedition_cards += len(expedition)
        score = state.expedition_score(policy_player, color)
        diagnostics.final_expedition_scores.append(score)
        if score > 0:
            positive += 1
            diagnostics.positive_expedition_scores.append(score)
            if color in first_open_recoverable_by_color:
                diagnostics.first_open_positive_recoverable_scores.append(
                    first_open_recoverable_by_color[color]
                )
        elif score < 0:
            negative += 1
            diagnostics.negative_expedition_scores.append(score)
            if color in first_open_recoverable_by_color:
                diagnostics.first_open_negative_recoverable_scores.append(
                    first_open_recoverable_by_color[color]
                )
        else:
            breakeven += 1
        if len(expedition) >= state.config.bonus_threshold:
            bonus += 1
        if score < -20:
            below_minus_20 += 1

    diagnostics.opened_colors.append(opened)
    diagnostics.five_color_open_count += int(opened == state.config.n_colors)
    diagnostics.expedition_cards.append(expedition_cards)
    diagnostics.score_per_opened_color.append(policy_score / max(1, opened))
    diagnostics.positive_expeditions.append(positive)
    diagnostics.negative_expeditions.append(negative)
    diagnostics.breakeven_expeditions.append(breakeven)
    diagnostics.bonus_expeditions.append(bonus)
    diagnostics.below_minus_20_expeditions.append(below_minus_20)


def _accumulate_game_diagnostics(
    target: PolicyEvalDiagnostics,
    source: PolicyEvalDiagnostics,
) -> None:
    target.games += source.games
    target.wins += source.wins
    target.losses += source.losses
    target.draws += source.draws
    target.score.extend(source.score)
    target.opponent_score.extend(source.opponent_score)
    target.diff.extend(source.diff)
    target.lengths.extend(source.lengths)
    target.timeouts += source.timeouts
    target.policy_actions += source.policy_actions
    target.play_actions += source.play_actions
    target.discard_actions += source.discard_actions
    target.draw_deck_actions += source.draw_deck_actions
    target.draw_pile_actions += source.draw_pile_actions
    target.entropies.extend(source.entropies)
    target.opened_colors.extend(source.opened_colors)
    target.five_color_open_count += source.five_color_open_count
    target.expedition_cards.extend(source.expedition_cards)
    target.opening_counts.append(sum(source.opening_counts))
    target.bad_open_counts.append(sum(source.bad_open_counts))
    target.weak_open_counts.append(sum(source.weak_open_counts))
    target.good_open_counts.append(sum(source.good_open_counts))
    target.opening_recoverable_scores.extend(source.opening_recoverable_scores)
    target.score_per_opened_color.extend(source.score_per_opened_color)
    target.positive_expeditions.extend(source.positive_expeditions)
    target.negative_expeditions.extend(source.negative_expeditions)
    target.breakeven_expeditions.extend(source.breakeven_expeditions)
    target.bonus_expeditions.extend(source.bonus_expeditions)
    target.below_minus_20_expeditions.extend(source.below_minus_20_expeditions)
    target.final_expedition_scores.extend(source.final_expedition_scores)
    target.positive_expedition_scores.extend(source.positive_expedition_scores)
    target.negative_expedition_scores.extend(source.negative_expedition_scores)
    target.first_open_positive_recoverable_scores.extend(
        source.first_open_positive_recoverable_scores
    )
    target.first_open_negative_recoverable_scores.extend(
        source.first_open_negative_recoverable_scores
    )
    target.runtime.accumulate(source.runtime)


def _visible_recoverable_summary(
    state: GameState,
    player: int,
    color: int,
) -> dict[str, float | bool]:
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
    margin = projected_sum + state.config.expedition_penalty
    recoverable_score = margin * (projected_wagers + 1)
    return {
        "recoverable_score": float(recoverable_score),
        "has_bonus_path": projected_len >= state.config.bonus_threshold,
    }


def _numeric_value(card, min_rank: int) -> int:
    if card.rank == 0:
        return 0
    return min_rank + card.rank - 1


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0.0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[int], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def load_strategy_policy_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    sample: bool = False,
    seed: int | None = None,
) -> tuple[StrategyNetPolicy, LostCitiesConfig]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    game_config = LostCitiesConfig(**payload["game_config"])
    network = DeepCFRMLP.from_config(
        int(payload["input_dim"]),
        int(payload["action_size"]),
        cfg.network,
    ).to(device)
    network.load_state_dict(payload["strategy_network"])
    network.eval()
    return (
        StrategyNetPolicy(network, device=device, sample=sample, seed=seed, encoding=cfg.encoding),
        game_config,
    )


def _record_game_if_needed(
    game: _EvalGame,
    game_records: list[dict] | None,
    opponent: str,
) -> None:
    if game_records is None:
        return
    policy_player_score = game.state.total_score(game.policy_player)
    opponent_player = 1 - game.policy_player
    opponent_score = game.state.total_score(opponent_player)

    if policy_player_score > opponent_score:
        winner = game.policy_player
    elif opponent_score > policy_player_score:
        winner = opponent_player
    else:
        winner = -1

    record = {
        "game_index": game.game_index,
        "seed": game.game_seed,
        "policy_player": game.policy_player,
        "winner": winner,
        "score_policy": policy_player_score,
        "score_opponent": opponent_score,
        "score_diff_policy": policy_player_score - opponent_score,
        "steps": game.steps,
        "opponent": opponent,
    }
    game_records.append(record)
