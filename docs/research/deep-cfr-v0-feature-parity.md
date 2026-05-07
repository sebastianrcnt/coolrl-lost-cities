# Deep CFR v0: Subsystem Coverage vs. Legacy Reference

**Last verified:** 2026-05-07, commit `ad0be89`

Source: `docs/archive/deep-cfr-v0-gap-vs-coolrl.md`

## Question

What does this repository's Deep CFR implementation cover relative to the legacy
`../coolrl` reference, and where are the intentional gaps?

Short answer: **all core Deep CFR subsystems are implemented; the gaps are
tooling, not correctness.** Traversal, training, encoding, memory,
checkpoints, evaluation, self-play league, imitation pretraining, and PG
fine-tuning are all present in the Cython/PyTorch package. Missing items
relative to legacy (preset configs, per-worker observability, plotters, W&B)
are deliberately deferred. The real remaining frontier is *performance*
(Python-boundary policy calls, recursive Cython traversal), not feature parity.

## Scope

The goal of this repo from the start was *not* legacy feature parity — it was
higher training throughput via Cython hot-paths and cleaner experiment
infrastructure. The gap document tracks where parity has been achieved and where
it has not.

## What is fully implemented

**Traversal** — all of the core Deep CFR traversal logic lives in
`src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`:
recursive traversal with traverser/opponent node dispatch, outcome-sampling
with epsilon exploration and importance-weight correction, optional value
clipping, unsampled-regret modes (`zero` and `negative_node_value`), depth and
node-budget cutoffs with score-diff and rollout-based terminal values, deck-draw
chance sampling with push/pop state restoration, and instantaneous regret and
strategy memory collection.

The old Python fallback `traverser.py` has been removed from the mainline; Cython
is the sole traversal path.

**Training and memory** — `trainer.py`, `memory.py`, `networks.py` provide PyTorch
advantage networks (one per player) and a strategy network, legal-mask-aware
advantage loss, masked strategy cross-entropy, reservoir sampling with capacity
limits, single-process and multiprocess worker batches with result merging in the
parent.

**Encoding** — `encoding.pyx` exposes both a Python wrapper and a C-level buffer
write path. The feature set covers: phase flags, current/traversing player, deck
ratio, hand slot features, public expeditions for both players, public discards,
public card counts, total score and score diff, turn ratio, pending-discard
one-hot, and legal action mask.

**Runtime operations** — checkpoint save/load (`checkpoints.py`), config stored
in checkpoints and `config.json`, strategy-net policy adapter, evaluation against
registered classic bots (`evaluate.py`), training CLI and evaluation CLI
(`cli.py`), traversal benchmark CLI (`benchmark.py`), `metrics.jsonl` /
`runtime_progress.json` / `train.log` run artifacts, self-play league with
snapshot pool and weighted current/recent/older/anchor bucket sampling, safe-
heuristic anchor opponent, safe-heuristic imitation pretraining (`imitation.py`),
and policy-gradient fine-tuning (`policy_gradient.py`).

As of `ad0be89`, the package also includes `inference_server.py`,
`inference_client.py`, `inference_buffers.py`, `analyze.py`, `tracking.py`, and
`traversal_stats.py` — additions beyond the original plan that support batched
inference experiments and richer metrics collection.

## Intentional gaps (not blockers)

| Area | Legacy has | This repo | Notes |
|---|---|---|---|
| Config presets | Many experiments | Sparse YAML configs | Intentional — config-first, not preset-first |
| Multiprocess observability | Progress callbacks per worker, hotspot timing | Basic worker merge only | Low-priority tooling gap |
| Metrics visualization | Plot/status commands | `metrics.jsonl` only; no built-in plotter | `analyze.py` partially addresses this |
| Checkpoint artifacts | W&B integration | Local only | Not a correctness issue |
| Legacy visualization helpers | Present | Not ported | Not needed for training |

None of these gaps affect the correctness or usefulness of the training loop.

## Performance gap: the real remaining frontier

The original split from the legacy repo was motivated by traversal performance,
not feature parity. As of `ad0be89`:

- Game state mutation, legal-action generation, apply/undo, and cached scoring
  run in Cython (`game.pyx` in the parent package).
- Information-state encoding and regret matching have Cython modules
  (`encoding.pyx`, `cfr_math.pyx`).
- The full traversal loop runs through `traversal.pyx`.
- Policy inference and reservoir memory materialization still cross the Python
  boundary (PyTorch call and NumPy buffer write).
- Traversal is still recursive inside Cython; an explicit iterative Cython
  scheduler is a future optimization.

The batched inference path (`inference_server.py` / `inference_client.py`) is an
opt-in experiment toward reducing the Python boundary cost for policy calls, but
it is not the default training path.

The performance roadmap (C-level action enumeration, push/pop in tight loops,
batched memory writes, batched policy inference, and ultimately a Cython
iterative traversal scheduler) is the main remaining work, not legacy feature
parity.

## Practical implication

- Don't reach for the legacy repo to fill correctness gaps — there are none in
  scope. Reach for it only for tooling references (preset YAMLs, plotting,
  W&B wiring) where porting is deliberately deferred.
- New experiments should land in this repo's Cython traversal path; backporting
  to the legacy traverser is not a goal.
- When prioritizing optimization work, target the Python boundary
  (policy inference, NumPy buffer writes) before re-tuning anything already
  fully in Cython — that is where the remaining throughput is hiding.
- Treat tooling-gap items in the table above as "open tickets, not blockers";
  they should not gate training or eval work.
