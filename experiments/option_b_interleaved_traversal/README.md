# Option B Interleaved Traversal Prototype

Experiment-only prototype for `docs/plans/option_b_interleaved_traversal.md`.
It does not wire into the trainer and does not replace the production Cython
recursive traversal path.

The prototype implements a small outcome-sampling traversal subset twice:

- recursive baseline: one policy request per forward (`bs=1`)
- explicit continuation scheduler: many traversal contexts yield at policy
  states, then policy requests are batched and scattered back to contexts

To make interleaving parity checkable, each traversal context owns its own RNG
state. This is intentionally stricter for the prototype scheduler and not a
claim that production Cython parity is solved.

Run:

```bash
uv run python experiments/option_b_interleaved_traversal/prototype_interleaved.py \
  --traversals 64 \
  --interleave-width 32 \
  --max-depth 8 \
  --max-nodes 512 \
  --device cpu
```

CUDA spot check:

```bash
uv run python experiments/option_b_interleaved_traversal/prototype_interleaved.py \
  --traversals 64 \
  --interleave-width 32 \
  --max-depth 8 \
  --max-nodes 512 \
  --device cuda \
  --output experiments/option_b_interleaved_traversal/results_cuda.json
```

2026-05-07 results, `configs/deep_cfr/default.yaml`, RTX 3090 host:

| Device | Mode | total s | forward s | scheduler s | batch mean | batch max | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | recursive | 0.084 | 0.054 | - | 1.0 | 1 | 1.00x |
| CPU | interleaved | 0.018 | 0.005 | 0.003 | 32.0 | 32 | 4.71x |
| CUDA | recursive | 0.195 | 0.144 | - | 1.0 | 1 | 1.00x |
| CUDA | interleaved | 0.028 | 0.014 | 0.003 | 32.0 | 32 | 6.98x |

Interpretation: the scheduling shape works in the prototype. Explicit
continuations can raise realized policy batch size from 1 to the interleave
width while preserving prototype value/stat/sample parity. The next risk is not
whether batching can be formed; it is whether the full production Cython CFR
state machine can be represented safely with the same continuation discipline.
