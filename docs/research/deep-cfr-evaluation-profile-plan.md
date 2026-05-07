# Deep CFR Evaluation Profiling

**Last verified:** 2026-05-08, commit `5c221fb`
**Source:** `docs/archive/deep-cfr-evaluation-profile-plan.md`

## Question

How are evaluation runtime costs categorized in Deep CFR, and what do these metrics reveal about system bottlenecks?

Evaluation is a critical path for measuring agent progress, but its runtime can be unpredictable. To move beyond wall-clock guessing, the system instruments the evaluation loop with granular counters that distinguish between neural network inference, state encoding, and game engine overhead.

## Code reference

The primary instrumentation structure is the `EvalRuntimeCounters` dataclass in `src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py` (line 20). It tracks elapsed seconds across several distinct phases of a single game step:

```python
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
```

These metrics are updated in `select_actions` and `action_distribution` (lines 200-280), capturing the micro-timing of every policy request.

## Performance Analysis

The instrumentation allows for a tiered analysis of the evaluation bottleneck. By comparing these counters, one can pinpoint the specific layer responsible for performance degradation:

### 1. The Policy Path (`policy_select_seconds`)
This is the total time spent by the agent under evaluation. It is further subdivided to identify efficiency gaps in the neural pipeline:
- **`policy_network_seconds`**: Time spent inside the PyTorch `forward` pass. If this dominates, the bottleneck is model inference. For small models on CUDA, this often signals high kernel launch overhead for batch-size-1 requests.
- **`policy_encoding_seconds`**: Time spent converting `GameState` objects into numerical info-state tensors. High values here suggest that the Python-based feature engineering is a bottleneck.
- **`policy_legal_mask_seconds`**: Time spent calculating legal moves. In Lost Cities, this involves scanning the hand and board state.

### 2. Environment and Opponents
- **`opponent_act_seconds`**: Time spent by the opponent bot. When evaluating against expensive bots (like heuristic-heavy search agents), this metric isolates their cost from the main agent's performance.
- **`apply_action_seconds`**: The cost of the game engine itself (`GameState.apply_action`). High values indicate that the Cython game logic is the primary constraint.

## Interpretation

The relationship between these metrics dictates the optimization strategy. If `policy_network_seconds` is the primary driver, the system is "model-bound," and improvements should focus on batching evaluation games or using inference accelerators like TensorRT. Conversely, if `policy_encoding_seconds` dominates, the system is "feature-bound," and the feature extraction logic should be moved to Cython or vectorized.

When `opponent_act_seconds` dominates, any local optimizations to the strategy network or encoding will have negligible impact on total evaluation time, as the bottleneck resides in the external bot's implementation.

## Practical Implications

- **Optimization Priority**: Always check the ratio of `policy_network_seconds` to `policy_select_seconds` before attempting model optimizations.
- **Device Selection**: Large `policy_network_seconds` on CUDA relative to CPU for small models is a known symptom of launch-latency saturation, justifying a move to CPU for serial evaluation.
- **Regression Testing**: Evaluation metrics should be compared across iterations (e.g., comparing iteration 5 vs 10) to detect memory leaks or data structure bloat in the diagnostics path (`diagnostics_seconds`).

## References

- `docs/research/deep-cfr-evaluation-profile.md` (Analysis of CUDA vs CPU latency)
- `src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py` (Implementation)