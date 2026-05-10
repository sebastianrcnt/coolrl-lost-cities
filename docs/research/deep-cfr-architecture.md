# Deep CFR Package Architecture and Design Rationale

**Last verified:** 2026-05-07, commit `ad0be89`

Source: `docs/archive/deep-cfr-v0-plan.md`

## Question

What is the package layout of this Deep CFR implementation, why are the
subsystems split the way they are, and what are the design choices that drive
the architecture?

Short answer: **the Cython/Python split mirrors the compute profile.** The
traversal hot path (`traversal.pyx`, `cfr_math.pyx`, `encoding.pyx`) calls
`GameState` C APIs directly without Python round-trips; everything that runs
once per iteration or is GPU/I/O-dominated (training, checkpointing, eval, CLI)
stays in Python. A Pydantic config tree bridges the two by handing typed
scalars and arrays across the boundary.

## Package layout (current as of `ad0be89`)

```text
src/coolrl_lost_cities/games/classic/deep_cfr/
  # Cython hot-path modules
  cfr_math.pyx / cfr_math.pxd    — regret matching arithmetic (all-negative fallback, etc.)
  encoding.pyx / encoding.pxd    — information-state feature extraction; Python + C-buffer API
  traversal.pyx / traversal.pxd  — full Deep CFR tree walking; calls GameState C APIs directly

  # Python training layer
  config.py          — Pydantic config tree for all subsystems
  memory.py          — reservoir memory buffers (advantage and strategy)
  networks.py        — PyTorch advantage and strategy network definitions
  trainer.py         — orchestration: traverse → sample → train → eval → checkpoint loop
  checkpoints.py     — save/load for networks, optimizer state, training counters
  workers.py         — multiprocess traversal worker pool and result merge

  # Inference experiments (opt-in)
  inference_server.py   — batched policy inference server (Option A batched traversal)
  inference_client.py   — client stub for the inference server
  inference_buffers.py  — shared-memory buffer management for batched inference

  # Runtime tooling
  cli.py             — training and evaluation CLI entry points
  benchmark.py       — traversal throughput benchmark CLI
  evaluate.py        — evaluation loop against registered classic bots
  analyze.py         — metrics.jsonl analysis helpers
  tracking.py        — run artifact helpers (metrics, runtime_progress.json)
  traversal_stats.py — structured traversal diagnostic metrics

  # Auxiliary training modes
  imitation.py       — heuristic imitation pretraining
  policy_gradient.py — policy-gradient fine-tuning
```

The compiled `.c` and `.so` artifacts (`cfr_math.c`, `encoding.c`, `traversal.c`
and their `.cpython-311-x86_64-linux-gnu.so` counterparts) are build outputs of
the Cython modules and live alongside the sources.

The original plan proposed 11 files; the current package has grown to 20+
Python/Cython source files as experiments and tooling have been added.

## Why Cython for traversal

The Deep CFR inner loop is an alternating-player tree walk that visits thousands
to millions of nodes per iteration. Each node needs: legal action enumeration,
regret matching, policy sampling, state push/pop for action application and
undo, and value backpropagation. In Python, each of these operations carries
dict lookups, reference counting, and interpreter overhead that compound across
the call tree.

`traversal.pyx` calls the `GameState` C APIs (`_legal_actions_c`,
`_unified_legal_actions_c`, `push_action`, `pop_action`, score caches, deck
sampling helpers) directly without Python object construction at each step. This
keeps the game-state hot path in C-land. `cfr_math.pyx` and `encoding.pyx`
extend the same principle to regret arithmetic and feature extraction
respectively, so a traversal node that produces a training sample can encode its
information state and compute regrets without leaving Cython.

The one remaining boundary cost is the PyTorch policy call: the advantage network
forward pass requires moving data to GPU and back, which necessarily crosses into
Python/PyTorch. This is the primary remaining performance target. The `inference_server`
/ `inference_client` / `inference_buffers` trio is an opt-in experiment that batches
multiple traversal-paused states into a single GPU forward pass, amortizing that
boundary cost.

## Why PyTorch for networks

PyTorch is the standard choice for the training layer. The advantage and strategy
networks are straightforward MLPs with a legal-mask head; no exotic architecture
is needed. PyTorch's autograd, optimizer API, and GPU memory management handle
the training loop cleanly. The design deliberately keeps the network definitions
(in `networks.py`) thin and the training orchestration (in `trainer.py`) separate,
so the network architecture can be swapped without touching traversal.

## Why this module split

The Cython/Python split mirrors the compute profile:

- **Cython**: everything that runs inside the traversal loop and must be fast.
  `traversal.pyx` is the main entry point; `cfr_math.pyx` and `encoding.pyx`
  are helpers it calls. These modules expose C-level APIs (`.pxd` headers) so
  they can call each other without Python object round-trips.
- **Python**: everything that runs once per iteration or is dominated by I/O or
  GPU compute. Training, checkpointing, evaluation, and CLI tooling have no
  reason to be in Cython and benefit from Python's ergonomics for development
  speed.
- **Config**: all hyperparameters live in `config.py` as a Pydantic model tree.
  This gives type checking and YAML deserialization for free, and means the
  Cython modules receive plain scalars and typed arrays at call boundaries rather
  than dict lookups.

## Non-goals (held from the original plan)

The following were explicitly out of scope for v0 and remain so:

- Multi-machine distributed training.
- Exploitability calculation.
- Full ISMCTS integration (particle-belief opponent sampling).
- Large experiment orchestration or W&B artifact integration.

These non-goals have not changed. The performance roadmap (explicit iterative
Cython traversal scheduler, C-level memory writes, fully batched policy
inference) is the next meaningful investment, not deeper experiment
infrastructure.

## Practical implication

- New training-layer features (memory variants, eval hooks, tracking) belong in
  Python; do not push them into the Cython modules unless they sit inside the
  per-node traversal loop.
- Anything called per-node (legal actions, regret update, encoding) must stay
  in Cython and use the existing `.pxd` C-level APIs — adding a Python helper
  here regresses throughput across the whole tree walk.
- The remaining boundary cost is the PyTorch policy forward pass; further
  performance work should target it (batched inference, iterative scheduler)
  rather than re-Cythonizing already-Python pieces.
- Hyperparameter additions go in `config.py` as Pydantic fields, not as
  positional kwargs threaded through Cython signatures.

## Discrepancy note

The original plan (`docs/archive/deep-cfr-v0-plan.md`) listed `traverser.py`
as the Python fallback traversal path. As of `ad0be89`, that file has been
removed; `traversal.pyx` is the sole traversal implementation. The plan also
did not anticipate `inference_server.py`, `inference_client.py`,
`inference_buffers.py`, `analyze.py`, `tracking.py`, `traversal_stats.py`,
`imitation.py`, `policy_gradient.py`, or `workers.py`, all of which exist in
the current package. These additions reflect experiments and tooling that
accumulated after the initial plan was written.
