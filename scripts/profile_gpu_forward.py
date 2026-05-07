"""Profile GPU forward-pass throughput for the Deep CFR trainer network.

Builds the same DeepCFRMLP that ``DeepCFRTrainer.__init__`` constructs from
``configs/deep_cfr/default.yaml``, then measures average forward-pass time on
CUDA across a sweep of batch sizes. The goal is to decide whether batched
traversal inference (Optimization Priorities #5) is worth implementing.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
from coolrl_lost_cities.games.classic.game import GameState

from coolrl_lost_cities.games.classic.deep_cfr.config import load_config
from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "deep_cfr" / "default.yaml"

BATCH_SIZES = [1, 4, 16, 64, 256, 1024]
WARMUP_ITERS = 10
MEASURE_ITERS = 1000


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script requires a CUDA-capable GPU.")

    cfg = load_config(CONFIG_PATH)

    game_config = cfg.rules.to_lost_cities_config(seed=cfg.run.seed)
    probe = GameState.new_game(game_config, seed=cfg.run.seed)
    in_dim = input_dim(probe, cfg.encoding)
    action_size = 2 * probe.config.hand_size + 1 + probe.config.n_colors

    device = torch.device("cuda")
    torch.manual_seed(cfg.run.seed)
    network = DeepCFRMLP.from_config(in_dim, action_size, cfg.network).to(device)
    network.eval()

    print(
        f"Network: DeepCFRMLP  input_dim={in_dim}  output_dim={action_size}  "
        f"hidden_size={cfg.network.hidden_size}  num_layers={cfg.network.num_layers}  "
        f"activation={cfg.network.activation}"
    )
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup iters: {WARMUP_ITERS}    Measure iters: {MEASURE_ITERS}")
    print()

    results: list[tuple[int, float, float]] = []
    with torch.inference_mode():
        for bs in BATCH_SIZES:
            x = torch.randn(bs, in_dim, device=device)

            # Warm-up
            for _ in range(WARMUP_ITERS):
                network(x)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(MEASURE_ITERS):
                network(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            us_per_call = (elapsed / MEASURE_ITERS) * 1e6
            us_per_state = us_per_call / bs
            results.append((bs, us_per_call, us_per_state))

    bs1_us_per_state = results[0][2]
    print(f"{'batch_size':>10} | {'μs/call':>10} | {'μs/state':>10} | {'speedup_vs_bs1':>14}")
    print("-" * 56)
    for bs, us_call, us_state in results:
        speedup = bs1_us_per_state / us_state
        print(f"{bs:>10} | {us_call:>10.2f} | {us_state:>10.3f} | {speedup:>13.2f}x")


if __name__ == "__main__":
    main()
