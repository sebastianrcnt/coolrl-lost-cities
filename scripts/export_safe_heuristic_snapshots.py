from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

from coolrl_lost_cities.games.classic.bots.heuristic_py import SafeHeuristicBot
from coolrl_lost_cities.games.classic.bots.registry import (
    LOOSE_SAFE_HEURISTIC_PARAMS,
    STRICT_SAFE_HEURISTIC_PARAMS,
)

VARIANTS = {
    "default": None,
    "loose": LOOSE_SAFE_HEURISTIC_PARAMS,
    "strict": STRICT_SAFE_HEURISTIC_PARAMS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export safe-heuristic bot parity snapshots for external implementations."
    )
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds per config.")
    parser.add_argument("--max-steps", type=int, default=10_000)
    return parser.parse_args()


def _configs() -> list[tuple[str, LostCitiesConfig]]:
    return [
        ("classic", LostCitiesConfig()),
        ("small", LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)),
        (
            "no-handshakes",
            LostCitiesConfig(n_colors=3, n_ranks=5, n_handshakes=0, hand_size=5),
        ),
    ]


def _record(
    *,
    config_name: str,
    variant_name: str,
    seed: int,
    turn: int,
    state: GameState,
    action: int,
) -> dict[str, Any]:
    return {
        "config_name": config_name,
        "variant": variant_name,
        "seed": seed,
        "turn": turn,
        "phase": state.phase,
        "current_player": state.current_player,
        "expected_action": action,
        "state": state.to_snapshot(),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for config_name, config in _configs():
            for variant_name, params in VARIANTS.items():
                for seed in range(args.seeds):
                    bot = SafeHeuristicBot(params)
                    state = GameState.new_game(config, seed=seed)
                    for turn in range(args.max_steps):
                        if state.terminal:
                            break
                        action = bot.act(state)
                        handle.write(
                            json.dumps(
                                _record(
                                    config_name=config_name,
                                    variant_name=variant_name,
                                    seed=seed,
                                    turn=turn,
                                    state=state,
                                    action=action,
                                ),
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        count += 1
                        state.apply_action(action)
                    else:
                        raise RuntimeError(
                            f"game did not terminate: config={config_name} "
                            f"variant={variant_name} seed={seed}"
                        )
    print(f"Wrote {count} snapshots to {output}")


if __name__ == "__main__":
    main()
