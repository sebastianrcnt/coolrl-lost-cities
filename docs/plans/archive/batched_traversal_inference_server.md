# Plan: Batched Traversal Inference Server (Priority #5, Option A)

**Status:** Archived. Implemented and benchmarked on 2026-05-07 as Option A; end-to-end traversal regressed because sync-blocking traversal could not feed large batches. Superseded by `docs/plans/option_b_interleaved_traversal.md`.
**Owner:** Codex
**Background:** See `docs/performance.md` → "Batched Traversal Inference: Design Decision (2026-05-07)" for the A/B/C analysis and rationale. This plan implements Option A.

## Goal

Replace per-worker single-state CPU policy forwards in Deep CFR traversal with a central GPU inference server that batches policy requests across all workers. Targets the dominant phase (`traversal_seconds` ≈ 60% of iteration time).

## Non-goals

- Do not modify the Cython traversal recursion structure.
- Do not modify the game engine, replay buffer, or training loop math.
- Do not implement Option C (single-process vectorized traversal). Keep it as future work.
- Do not require Cython `nogil`-cleanliness.
- Do not change the public CLI surface.

## Success criteria

1. With `traversal.inference_backend: server` enabled on `configs/deep_cfr/default.yaml`, end-to-end training produces eval-winrate trajectories indistinguishable (within seed noise) from the current `local` backend over at least 50 iterations on `home` hardware.
2. On `home` (6-core + RTX 3090), `traversal_seconds` decreases by at least 30% compared to the current run profile in `docs/performance.md`.
3. On `remote` (32-core + weak GPU), `traversal_seconds` decreases or stays within 10% of current; if it regresses more, fall back to `local` is the operator's choice — the plan still ships.
4. With `traversal.inference_backend: local`, behavior is byte-identical to current `main`.
5. All existing tests pass. New unit tests for the inference client/server round-trip pass.

## Key files (current)

- `src/coolrl_lost_cities/games/classic/deep_cfr/workers.py` — `run_traversal_worker_batch`. Spawns CPU networks per worker (`device = torch.device("cpu")` at line 62). This is the worker entry point that must learn about the inference server.
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx` — recursive traversal. Two policy call sites:
  - Lines 473–475: `networks[player](x).squeeze(0).detach().cpu().numpy()` — advantage-net forward.
  - Lines 550–552: `self.strategy_network(x).squeeze(0).detach().cpu().numpy()` — strategy-net forward.
  - Both are the seams that route through the inference client when the backend is `server`.
- `src/coolrl_lost_cities/games/classic/deep_cfr/networks.py` — `DeepCFRMLP`.
- `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py` — main loop; spawns the worker pool and owns the trainer-side networks. Must also start/stop the inference server and push weights periodically.
- `src/coolrl_lost_cities/games/classic/deep_cfr/config.py` — config schema. Add `traversal.inference_backend` and an `inference_server` block.

## New files

- `src/coolrl_lost_cities/games/classic/deep_cfr/inference_server.py` — server process: owns models on GPU, drains the request queue, runs batched forward, writes responses.
- `src/coolrl_lost_cities/games/classic/deep_cfr/inference_client.py` — client used inside workers: claims a request slot, posts the encoded state, waits for the response. Provides a `forward(network_id, player, state)` API.
- `src/coolrl_lost_cities/games/classic/deep_cfr/inference_buffers.py` — shared-memory tensor pool: pre-allocated `[num_slots, input_dim]` request buffer and `[num_slots, action_size]` response buffer, plus per-slot ready events and a free-slot stack.
- `tests/games/classic/deep_cfr/test_inference_server.py` — unit/integration tests for the request path.

## Architecture

### Process layout

```
main process (trainer)
 ├─ inference server process (GPU)
 │   - owns advantage networks (per-player), strategy network, league snapshots
 │   - request buffer (shared mem): [num_slots, input_dim] float32
 │   - response buffer (shared mem): [num_slots, action_size] float32
 │   - control queue (mp.Queue): control messages only
 │   - per-slot ready event (mp.Event[num_slots])
 ├─ traversal worker 1 ─┐
 ├─ ...                 ├─ inference clients post requests, wait on per-slot event
 └─ traversal worker N ─┘
```

### Request protocol

A worker policy call becomes:
1. Client pops a free slot id from a shared `mp.Queue`-backed free-slot stack (small int).
2. Client writes the encoded state into `request_buffer[slot_id]` (shared memory; no pickle).
3. Client puts a `RequestMessage(slot_id, network_kind, player, network_index)` onto the request control queue. `network_kind ∈ {ADVANTAGE, STRATEGY, LEAGUE}`. Pickle cost is negligible (small struct of ints).
4. Server drains the request queue with a short batch window (`batch_window_us`, default 200μs) up to `max_batch` (default 256). Empty drain blocks on the queue with a small timeout.
5. Server stacks the requested rows from `request_buffer`, runs forward, writes outputs back to `response_buffer[slot_id]` for each request.
6. Server fires the per-slot ready event for each completed request.
7. Client wakes on its slot's event, reads `response_buffer[slot_id]`, copies to a local numpy array, returns the slot to the free-slot stack.

### Weight sync

- Trainer holds master weights. On a fixed cadence (`inference_server.weight_sync_every` iterations, default 1 — push every iter), trainer sends a `WeightUpdateMessage` containing `state_dict`s via `torch.multiprocessing` (auto-shares tensor storage; cheap once and copied into server's GPU model).
- Server applies `load_state_dict` and signals `weight_sync_complete`. Trainer waits before kicking off the next traversal iteration.
- League snapshots: passed via the same channel when the league updates. League list is small relative to per-iter cost.

### Backpressure & lifecycle

- Free-slot stack size = `num_slots` (default `max(64, 4 * num_workers * worker_chunk_size)`). Workers block on slot allocation if all slots in flight; this is the natural backpressure.
- Server shutdown: trainer puts a `Shutdown` sentinel on the control queue at training end. Server drains, exits.
- Crash isolation: if the server dies, workers will hang on their slot events. Trainer monitors the server process; on death, it raises and tears down the pool. No silent corruption.

### IPC choice

- `torch.multiprocessing` for weight passing (auto-shares tensors).
- `multiprocessing.shared_memory.SharedMemory` (numpy view) for request/response buffers — manual slot management. Pre-allocated once at startup; no per-call allocation.
- `multiprocessing.Queue` for control messages only (slot ids and small structs).
- `multiprocessing.Event` array for per-slot wakeups.

Rationale: see `docs/performance.md` § "IPC mechanism: multiprocessing + shared memory".

## Config schema

Extend `config.py`:

```python
@dataclass
class InferenceServerConfig:
    enabled: bool = False               # if False, behave like current code
    device: str = "cuda"                # server-side device
    num_slots: int | None = None        # None → auto: max(64, 4 * num_workers * worker_chunk_size)
    max_batch: int = 256
    batch_window_us: int = 200
    weight_sync_every: int = 1          # iterations
    use_amp: bool = False               # eval-only AMP for forward (no grad)

@dataclass
class TraversalConfig:
    # ... existing fields ...
    inference_backend: Literal["local", "server"] = "local"
```

In `default.yaml`, leave `inference_backend: local` for now. Add an explicit `configs/deep_cfr/default_server.yaml` variant that flips it on for benchmarking.

## Worker integration

`run_traversal_worker_batch` (workers.py) currently:
- Builds CPU `DeepCFRMLP` instances and loads `state_dict`s.
- Passes them as positional `networks` into the `Traversal` Cython object.

When `inference_backend == "server"`:
- Skip building local networks. Instead, build an `InferenceClient` bound to the shared buffers and queues that the trainer wires in via `TraversalWorkerBatch`.
- Pass a small `NetworkProxy` object with the same call signature as the current network: `proxy(x: torch.Tensor) -> torch.Tensor`. Internally it converts to numpy, calls `client.forward(...)`, returns a torch tensor.
- Critically: the Cython traversal sites at `traversal.pyx:473-475` and `:550-552` should not need source changes if `NetworkProxy` is a callable returning a 2-D tensor. The existing `.squeeze(0).detach().cpu().numpy()` chain still works on the proxy's returned tensor (which can just be a CPU tensor wrapping the numpy result). **Verify this; if Cython has typed assumptions that reject a Python proxy, fall back to a thin Python helper invoked from the `.pyx` instead.**

Add to `TraversalWorkerBatch`:
- `inference_handles: InferenceClientHandles | None` — shared-memory names, queue handles, event arrays. None when `inference_backend == "local"`.

## Trainer integration

`trainer.py`:
1. On run start, if `inference_backend == "server"`: instantiate `InferenceServer` (spawns process), build `InferenceClientHandles`, push initial weights, wait for `weight_sync_complete`.
2. Per iteration: before `pool.starmap(run_traversal_worker_batch, ...)`, push fresh weights if `iter % weight_sync_every == 0`. Pass `inference_handles` into each `TraversalWorkerBatch`.
3. After traversal: same as today.
4. On run end: send shutdown sentinel; join the server process.

## Implementation steps (ordered, each independently mergeable)

### Step 1: shared-memory buffer module

- Create `inference_buffers.py` with `InferenceBuffers` class: pre-allocates request/response numpy arrays via `SharedMemory`, exposes `attach(name)` for child processes, has `release()` cleanup.
- Free-slot management: `mp.Queue` of slot ids, populated at startup with `range(num_slots)`.
- Per-slot ready events: `[mp.Event() for _ in range(num_slots)]`.
- Unit test: parent creates buffers, child attaches, writes a row, parent reads. Verify zero-copy semantics.

### Step 2: inference client

- Create `inference_client.py`. `InferenceClient.forward(network_kind, player, network_index, state_np: np.ndarray) -> np.ndarray`:
  - Pop free slot, write state, post request, wait on event, copy response, return slot.
- Add a `NetworkProxy` callable that wraps `client.forward` to look like a `torch.nn.Module` for traversal call sites.
- Unit test: stub a server thread that echoes state*2; assert client gets the expected output.

### Step 3: inference server

- Create `inference_server.py`. Spawn-friendly entry function `run_inference_server(handles, model_config, control_queue, weight_queue)`.
- Owns models on `device`. Sets `eval()` and `inference_mode()`.
- Main loop: drain request queue with `batch_window_us` deadline, group by `(network_kind, network_index)`, run batched forward per group, scatter outputs to response buffer slots, fire events.
- Handles `WeightUpdateMessage` and `Shutdown`.
- Optional `use_amp`: wrap forward in `torch.autocast` when configured.
- Unit test: send N requests across multiple kinds; verify outputs match `model(stacked_input)`.

### Step 4: config + trainer wiring

- Extend `config.py` with `InferenceServerConfig` and `TraversalConfig.inference_backend`.
- Update `trainer.py` to start/stop server, push weights, attach handles to `TraversalWorkerBatch`.
- Update `workers.py`: when `inference_backend == "server"`, build proxies instead of CPU networks.
- Add `configs/deep_cfr/default_server.yaml` flipping `traversal.inference_backend: server`.

### Step 5: traversal call-site verification

- Run with `inference_backend: server` and a tiny config (1 worker, 1 traversal). Confirm the proxy is callable from `traversal.pyx:473-475` and `:550-552` without Cython type errors.
- If Cython rejects the proxy: refactor those two call sites to invoke a Python helper that takes `(networks, player, info_state)` and returns a numpy array. The helper picks `local` or `server` path by inspecting the object. This is a 2-line change per site.

### Step 6: integration tests

- `tests/games/classic/deep_cfr/test_inference_server.py`:
  - End-to-end smoke: 1 iteration of training with `server` backend on CPU device; assert no crash, replay buffer populated.
  - Equivalence: same seed, same initial weights, both backends → assert traversal samples match within numerical tolerance for at least 1 iteration. (May require `weight_sync_every = 1` and deterministic GPU forward; if exact match is fragile, accept distributional equivalence over 10 iterations.)

### Step 7: benchmarking

- Add `scripts/bench_inference_backend.py` (mirrors `scripts/profile_gpu_forward.py` style):
  - Run 10 iterations on `default.yaml` with `inference_backend=local`.
  - Run 10 iterations on `default_server.yaml`.
  - Print iteration-mean and per-phase mean times.
- Document results in a new `docs/performance.md` experiment subsection (date-stamped).

## Risks and mitigations

- **Cython proxy incompatibility (Step 5).** Mitigation: pre-prototype with a 5-line Python script that imports `Traversal` and passes a stub callable in place of a `DeepCFRMLP`. If it works, the rest of the plan stands.
- **Weight sync staleness invalidates CFR.** Mitigation: default `weight_sync_every: 1` (every iteration). Only loosen after measuring.
- **GPU contention with eval.** Eval already runs on the trainer's device; the inference server adds another tenant on the same GPU. Mitigation: serialize eval and traversal phases (they already are sequential in the iteration loop). Document this in `inference_server.py`.
- **Deadlock on server crash.** Mitigation: trainer monitors server `process.is_alive()` between iterations and raises if dead. Workers waiting on events will be torn down with the pool.
- **Slot exhaustion under bursty load.** Mitigation: default `num_slots = 4 * num_workers * worker_chunk_size` to absorb bursts. Operators can tune.
- **Remote (weak GPU) regression.** If `traversal_seconds` regresses on remote, the operator can flip `inference_backend: local` and ship without it. The plan succeeds either way.

## What this plan explicitly does not do

- Does not implement async client (workers stay sync-blocking on slot events). Async client is a future optimization if IPC round-trip dominates after measurement.
- Does not implement `nogil` threading or Option C. See `docs/performance.md` for the deferred path.
- Does not optimize encoding (`policy_encoding_seconds`). If post-A measurements show encoding dominates, that is a separate work item.

## Out-of-scope follow-ups (do not start)

- Option C (vectorized traversal) — only re-evaluate if A's measurements show GPU forward is no longer on the critical path.
- TensorRT / `torch.compile` on the inference server's models — defer until A baseline numbers are collected.
- Batched encoding inside the server (compute encoding from raw game state on GPU) — only if `policy_encoding_seconds` becomes the new bottleneck.
