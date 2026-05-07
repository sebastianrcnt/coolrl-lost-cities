# Traversal Policy Boundary Microbench

Purpose: separate the current traversal hot path into game/encoding overhead,
single-request policy boundary overhead, and batched forward lower bounds.

Run:

```bash
uv run python experiments/traversal_policy_boundary/bench_policy_boundary.py \
  --traversals 32 \
  --runs 5 \
  --warmup 1 \
  --corpus-size 512 \
  --component-repeats 4 \
  --forward-repeats 32 \
  --device cpu
```

CUDA spot check:

```bash
uv run python experiments/traversal_policy_boundary/bench_policy_boundary.py \
  --traversals 8 \
  --runs 3 \
  --warmup 1 \
  --corpus-size 512 \
  --component-repeats 2 \
  --forward-repeats 32 \
  --device cuda \
  --output experiments/traversal_policy_boundary/results_cuda.json
```

2026-05-07 results, `configs/deep_cfr/default.yaml`, RTX 3090 host:

| Device | Component | Median us/call | p95 us/call |
| --- | --- | ---: | ---: |
| CPU | encode + legal | 3.10 | 3.81 |
| CPU | push + pop | 0.15 | 0.22 |
| CPU | policy boundary bs=1 | 111.50 | 125.46 |
| CPU | torch forward bs=64 | 12.84 | 13.14 |
| CUDA | encode + legal | 3.16 | 3.88 |
| CUDA | push + pop | 0.16 | 0.25 |
| CUDA | policy boundary bs=1 | 181.30 | 194.77 |
| CUDA | torch forward bs=64 | 2.55 | 2.75 |

Interpretation: the game mechanics and state encoding are not the dominant
cost. The current one-row policy boundary dominates, and CUDA only becomes
attractive once requests are actually batched.
