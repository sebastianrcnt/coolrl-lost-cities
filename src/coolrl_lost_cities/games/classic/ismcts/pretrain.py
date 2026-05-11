"""Behavior cloning warm-start for the SO-ISMCTS network.

Generates games from a fixed heuristic policy vs itself, then trains the
AlphaZero-style network's policy + value heads in supervised fashion:

- policy loss: cross-entropy between network logits and the heuristic's chosen
  action (one-hot target, masked to legal actions).
- value loss: MSE on the final game score diff from each decision-maker's
  perspective, normalized by value_scale (same convention as the trainer).

The resulting checkpoint can be passed to `lost-cities-ismcts train
--resume-from` to start self-play with a heuristic-level prior instead of a
random init. This addresses the self-play "weak-equilibrium" problem: starting
from random, MCTS visit distributions converge to a mutually mediocre policy
that has near-zero win rate against the heuristic. Warm-starting at heuristic
level gives self-play a meaningful baseline to improve from.

Invoked via ``lost-cities-ismcts pretrain``.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from coolrl_lost_cities.games.classic.bots.registry import build_bot
from coolrl_lost_cities.games.classic.deep_cfr.encoding import encode_info_state, input_dim
from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from .config import IsMctsConfig, load_config
from .network import AlphaZeroNet


def _collect_samples(
    game_config: LostCitiesConfig,
    encoding,
    n_games: int,
    bot_name: str = "heuristic-balanced",
    seed: int = 0,
    max_turns: int = 500,
) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    """Roll out n_games of bot vs bot, returning per-decision samples.

    Each sample: (info_state, legal_mask, action_idx, value).
    value is the player-perspective score diff at game end.
    """
    samples: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    for game_idx in range(n_games):
        bots = [
            build_bot(bot_name, seed=seed + game_idx * 2),
            build_bot(bot_name, seed=seed + game_idx * 2 + 1),
        ]
        state = GameState.new_game(game_config, seed=seed + game_idx)
        decisions: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        turns = 0
        while not state.terminal and turns < max_turns:
            player = int(state.current_player)
            info_state = encode_info_state(state, player, encoding)
            legal_mask = np.asarray(state.unified_legal_mask(), dtype=bool)
            phase_action = bots[player].act(state)
            unified = state.to_unified_action(phase_action)
            decisions.append((info_state, legal_mask, int(unified), player))
            state.apply_unified_action(unified)
            turns += 1
        final_diff0 = float(state.score_diff(0))
        for info, mask, act, player in decisions:
            value = final_diff0 if player == 0 else -final_diff0
            samples.append((info, mask, act, value))
    return samples


def _train_supervised(
    network: AlphaZeroNet,
    samples: list,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    value_loss_weight: float,
) -> None:
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=max(weight_decay, 1e-4))
    rng = random.Random(0)
    indices = list(range(len(samples)))
    v_scale = float(network.value_scale)

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        n_batches = (len(indices) + batch_size - 1) // batch_size
        epoch_pl = 0.0
        epoch_vl = 0.0
        epoch_acc = 0.0
        epoch_n = 0
        for b in range(n_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            infos = np.stack([samples[i][0] for i in batch_idx])
            masks = np.stack([samples[i][1] for i in batch_idx])
            actions = np.array([samples[i][2] for i in batch_idx], dtype=np.int64)
            values = np.array([samples[i][3] for i in batch_idx], dtype=np.float32)

            info_t = torch.as_tensor(infos, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
            action_t = torch.as_tensor(actions, device=device)
            value_t = torch.as_tensor(values, device=device)

            logits, value_pred = network(info_t, mask_t)
            policy_loss = F.cross_entropy(logits, action_t)
            value_loss = F.mse_loss(value_pred / v_scale, value_t / v_scale)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == action_t).float().mean().item()
                epoch_pl += float(policy_loss.item()) * len(batch_idx)
                epoch_vl += float(value_loss.item()) * len(batch_idx)
                epoch_acc += acc * len(batch_idx)
                epoch_n += len(batch_idx)

        print(
            f"  epoch {epoch:3d}: policy_loss={epoch_pl / epoch_n:.4f} "
            f"value_loss={epoch_vl / epoch_n:.4f} "
            f"top1_match={epoch_acc / epoch_n:.3f}",
            flush=True,
        )

    network.eval()
    return optimizer


def add_pretrain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default=None, help="ISMCTS config YAML (controls network shape + rules)."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="PATH=VALUE",
    )
    parser.add_argument(
        "--bot",
        default="heuristic-balanced",
        help="Bot to clone (default: heuristic-balanced).",
    )
    parser.add_argument(
        "--games", type=int, default=2000, help="Number of bot-vs-bot games to roll out."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Supervised training epochs.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=50.0,
        help="Multiplier on value MSE (raw_MSE / value_scale^2). Default 50 to "
        "make value loss magnitude comparable to policy CE.",
    )
    parser.add_argument(
        "--out",
        default="runs/pretrain/heuristic_clone.pt",
        help="Output checkpoint path (compatible with --resume-from).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)


def _apply_config_overrides(config: IsMctsConfig, assignments: list[str]) -> IsMctsConfig:
    import yaml

    def deep_update(base: dict, patch: dict) -> None:
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    overrides: dict = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"override must be PATH=VALUE: {assignment}")
        path, raw_value = assignment.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = overrides
        keys = path.split(".")
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = value
    data = config.model_dump(mode="python")
    deep_update(data, overrides)
    return IsMctsConfig.model_validate(data)


def run_pretrain(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else IsMctsConfig()
    config = _apply_config_overrides(config, args.config_overrides)
    game_config = config.rules.to_lost_cities_config(seed=config.run.seed)
    device = torch.device(args.device)

    probe = GameState.new_game(game_config, seed=config.run.seed)
    in_dim = input_dim(probe, config.encoding)
    network = AlphaZeroNet.from_config(in_dim, probe.action_size, config).to(device)
    print(
        f"network: input_dim={in_dim} action_size={probe.action_size} "
        f"hidden={config.network.hidden_size} layers={config.network.num_layers}",
        flush=True,
    )

    print(f"rolling out {args.games} games of {args.bot} vs {args.bot}...", flush=True)
    t0 = time.perf_counter()
    samples = _collect_samples(
        game_config,
        config.encoding,
        n_games=args.games,
        bot_name=args.bot,
        seed=args.seed,
    )
    rollout_secs = time.perf_counter() - t0
    print(
        f"collected {len(samples)} decisions from {args.games} games "
        f"in {rollout_secs:.1f}s ({len(samples) / args.games:.1f} decisions/game)",
        flush=True,
    )

    print(
        f"training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, "
        f"value_weight={args.value_loss_weight}",
        flush=True,
    )
    t1 = time.perf_counter()
    optimizer = _train_supervised(
        network,
        samples,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        value_loss_weight=args.value_loss_weight,
    )
    train_secs = time.perf_counter() - t1
    print(f"training done in {train_secs:.1f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "game_config": game_config.to_snapshot(),
        "iteration": 0,
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": {
            "pretrain/games": args.games,
            "pretrain/samples": len(samples),
            "pretrain/epochs": args.epochs,
            "pretrain/bot": args.bot,
        },
    }
    torch.save(payload, out_path)
    print(f"saved pretrained checkpoint to {out_path}", flush=True)
