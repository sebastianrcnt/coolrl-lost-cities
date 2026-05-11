from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from coolrl_lost_cities.games.classic.bots.registry import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import MctsConfig, TrainingConfig
from .info_set import canonical_info_set_key
from .mcts import IsMctsSearcher, PendingSimulation
from .network import AlphaZeroNet
from .replay_buffer import ReplaySample
from .self_play import select_from_distribution, visit_distribution


@dataclass
class _PendingDecision:
    info_state: np.ndarray
    legal_mask: np.ndarray
    pi_target: np.ndarray
    player: int
    prior: np.ndarray
    game_index: int


@dataclass
class _GameContext:
    state: GameState
    rng: random.Random
    game_index: int
    decisions: list[_PendingDecision] = field(default_factory=list)
    steps: int = 0
    # Mixed-opponent setup: when traverser_seat is not None, only that seat
    # uses MCTS+network; the other seat is played by `opponent_bot`. None for
    # pure self-play games (both seats use MCTS).
    traverser_seat: int | None = None
    opponent_bot: object | None = None


@dataclass
class _SearchJob:
    context: _GameContext
    searcher: IsMctsSearcher
    traverser: int
    remaining: int


def play_self_play_iteration(
    network: AlphaZeroNet,
    mcts_config: MctsConfig,
    training_config: TrainingConfig,
    game_config: LostCitiesConfig,
    rng: random.Random,
    *,
    device: torch.device | str = "cpu",
    encoding=None,
    temperature: float = 1.0,
    max_steps: int = 10_000,
) -> list[ReplaySample]:
    device = torch.device(device)
    completed: list[list[ReplaySample]] = []
    active: list[_GameContext] = []
    started = 0
    target_games = training_config.games_per_iter
    mixed_fraction = float(training_config.mixed_opponent_fraction)

    def fill_active() -> None:
        nonlocal started
        while len(active) < training_config.interleave_games and started < target_games:
            traverser_seat: int | None = None
            opponent_bot = None
            if mixed_fraction > 0.0 and rng.random() < mixed_fraction:
                # Alternate trainee seat so MCTS sees both first- and
                # second-player perspectives equally.
                traverser_seat = started % 2
                opponent_bot = build_bot(
                    training_config.mixed_opponent_bot,
                    seed=rng.randrange(2**31),
                )
            active.append(
                _GameContext(
                    state=GameState.new_game(game_config, seed=rng.randrange(2**31)),
                    rng=random.Random(rng.randrange(2**31)),
                    game_index=started,
                    traverser_seat=traverser_seat,
                    opponent_bot=opponent_bot,
                )
            )
            started += 1

    fill_active()
    while active:
        jobs: list[_SearchJob] = []
        still_active: list[_GameContext] = []
        for context in active:
            if context.state.terminal or context.steps >= max_steps:
                completed.append(_finalize_context(context))
                continue
            player = int(context.state.current_player)
            # Mixed-opponent: if it's the opponent's turn in a mixed game,
            # let the heuristic bot move directly (no MCTS, no sample).
            if (
                context.traverser_seat is not None
                and context.opponent_bot is not None
                and player != context.traverser_seat
            ):
                phase_action = context.opponent_bot.act(context.state)
                unified = context.state.to_unified_action(phase_action)
                context.state.apply_unified_action(unified)
                context.steps += 1
                still_active.append(context)
                continue
            searcher = IsMctsSearcher(
                network,
                mcts_config,
                device=device,
                encoding=encoding,
                rng=random.Random(context.rng.randrange(2**31)),
            )
            # Pass opponent bot into the searcher for opponent-aware
            # determinization (search models the real opponent the trainee
            # faces, not a self-play mirror).
            if (
                mcts_config.opponent_aware_search
                and context.traverser_seat is not None
                and context.opponent_bot is not None
            ):
                searcher.set_opponent_bot(
                    context.opponent_bot,
                    traverser_seat=context.traverser_seat,
                )
            jobs.append(
                _SearchJob(
                    context=context,
                    searcher=searcher,
                    traverser=player,
                    remaining=mcts_config.n_simulations,
                )
            )
            still_active.append(context)

        active = still_active
        if jobs:
            _run_search_jobs(network, jobs, training_config.interleave_max_batch, device)
            for job in jobs:
                _finish_decision(job, mcts_config, encoding, temperature)

        fill_active()

    samples: list[ReplaySample] = []
    for game_samples in completed:
        samples.extend(game_samples)
    return samples


def _run_search_jobs(
    network: AlphaZeroNet,
    jobs: list[_SearchJob],
    max_batch: int,
    device: torch.device,
) -> None:
    while any(job.remaining > 0 for job in jobs):
        pending: list[tuple[_SearchJob, PendingSimulation]] = []
        for job in jobs:
            job_quota = min(
                job.remaining,
                job.searcher.config.parallel_simulations,
                max_batch - len(pending),
            )
            if job_quota <= 0:
                break
            job_pending = job.searcher.prepare_simulation_batch(
                job.context.state,
                job.traverser,
                job_quota,
            )
            pending.extend((job, item) for item in job_pending)
            job.remaining -= len(job_pending)
            if len(pending) >= max_batch:
                break
        if not pending:
            break
        _evaluate_global_batch(network, pending, device)


def _evaluate_global_batch(
    network: AlphaZeroNet,
    pending: list[tuple[_SearchJob, PendingSimulation]],
    device: torch.device,
) -> None:
    network_pending = [(job, item) for job, item in pending if item.terminal_value is None]
    values_by_id: dict[int, float] = {}
    priors_by_id: dict[int, np.ndarray] = {}
    if network_pending:
        infos = np.stack(
            [item.info_state for _job, item in network_pending if item.info_state is not None]
        )
        masks = np.stack(
            [item.legal_mask for _job, item in network_pending if item.legal_mask is not None]
        )
        with torch.inference_mode():
            x = torch.as_tensor(infos, dtype=torch.float32, device=device)
            mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
            probs = network.policy_distribution(x, mask).detach().cpu().numpy()
            _logits, values = network(x, mask)
            values_np = values.detach().cpu().numpy()
        for index, (_job, item) in enumerate(network_pending):
            priors_by_id[id(item)] = probs[index]
            values_by_id[id(item)] = float(values_np[index])

    for job, item in pending:
        if item.terminal_value is not None:
            value = item.terminal_value
        else:
            assert item.leaf_node is not None
            value = job.searcher._expand_with_prior(
                item.leaf_node,
                item.leaf_state,
                item.leaf_player,
                item.legal_actions,
                priors_by_id[id(item)],
                values_by_id[id(item)],
                not item.path,  # is_root for Dirichlet noise
            )
        job.searcher._backup(item.path, value, item.leaf_player)


def _finish_decision(
    job: _SearchJob,
    mcts_config: MctsConfig,
    encoding,
    temperature: float,
) -> None:
    context = job.context
    state = context.state
    player = int(state.current_player)
    legal_mask = np.asarray(state.unified_legal_mask(), dtype=bool)
    info = encode_info_state(state, player, encoding)
    root_key = canonical_info_set_key(state, player)
    root = job.searcher.tree.get_or_create(root_key, player=player, terminal=state.terminal)
    visits = {action: root.visits.get(action, 0) for action in state.unified_legal_actions()}
    pi = visit_distribution(visits, state.action_size, temperature=temperature)
    if pi.sum() <= 0:
        legal_actions = np.flatnonzero(legal_mask)
        pi[legal_actions] = 1.0 / len(legal_actions)
    prior = np.zeros(state.action_size, dtype=np.float32)
    for action in state.unified_legal_actions():
        prior[action] = float(root.priors.get(action, 0.0))
    context.decisions.append(
        _PendingDecision(
            info_state=info.astype(np.float32),
            legal_mask=legal_mask.astype(bool),
            pi_target=pi.astype(np.float32),
            player=player,
            prior=prior,
            game_index=context.game_index,
        )
    )
    action = select_from_distribution(pi, context.rng)
    state.apply_unified_action(action)
    context.steps += 1


def _finalize_context(context: _GameContext) -> list[ReplaySample]:
    # If the game did not terminate naturally (hit max_steps), the score
    # reflects an incomplete game — typically a "stall" outcome where both
    # sides have under-developed expeditions. Treating that as a real win
    # for either player creates a degenerate stall-and-pray learning
    # signal. Zero it out so the trajectory is neutral.
    if not context.state.terminal:
        final_diff0 = 0.0
    else:
        final_diff0 = float(context.state.score_diff(0))
    samples: list[ReplaySample] = []
    for decision in context.decisions:
        value = final_diff0 if decision.player == 0 else -final_diff0
        samples.append(
            ReplaySample(
                info_state=decision.info_state,
                legal_mask=decision.legal_mask,
                pi_target=decision.pi_target,
                v_target=value,
                player=decision.player,
                prior=decision.prior,
                game_index=decision.game_index,
            )
        )
    return samples
