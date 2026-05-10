# Deep CFR Evaluation Performance

**Last verified:** 2026-05-08, commit `b0b3855`
**Source:** `docs/archive/deep-cfr-evaluation-profile-2026-05-07.md`

## Question

Why is CUDA evaluation significantly slower than CPU evaluation in the current Deep CFR implementation, and how can evaluation throughput be improved?

Short answer: **Model forward latency (batch size 1) dominates evaluation wall-clock.** For the current small-model architecture, CUDA kernel launch and synchronization overhead outweighs its parallel processing advantage. Evaluation currently executes serial games with single-sample policy requests, making CPU the faster device by a factor of ~1.6x.

## Code reference

`src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py`, function `action_distribution` (around line 250):

```python
with torch.inference_mode():
    x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
    logits = self.strategy_network(x).squeeze(0).detach().cpu().numpy()
```

This single-sample forward pass is the tightest loop in evaluation. In a typical 100-game evaluation run against a variety of opponents, this function is called hundreds of thousands of times (e.g., ~236k calls in the 2026-05-07 profile run).

## Performance Analysis

Profiling data reveals a sharp divide between training and evaluation efficiency when using CUDA:

| Phase | CPU Time | CUDA Time | Speedup (CUDA) |
| :--- | :--- | :--- | :--- |
| **Advantage Train** | 4.05s | 2.88s | 1.40x |
| **Strategy Train** | 1.64s | 1.37s | 1.20x |
| **Evaluation** | 38.92s | 61.83s | **0.63x (Slower)** |

The discrepancy arises because training uses large batches (e.g., `batch_size: 512`), which allows the GPU to saturate and amortizes kernel launch overhead. Evaluation, however, steps through games one action at a time.

On CPU, the `network / turn` cost is approximately **0.074 ms**. On CUDA, this rises to **0.162 ms**. This 2x increase in per-turn latency is typical for small MLP models on CUDA, where the compute time is shorter than the host-to-device synchronization and kernel scheduling latency.

### Secondary Bottlenecks
- **Post-processing:** Moving tensors back to CPU (`.cpu().numpy()`) and calculating entropy adds measurable overhead on CUDA that is largely absent on CPU.
- **Opponent Logic:** Heuristic opponents (e.g., `heuristic_balanced`) contribute significant `opponent_act_seconds` (up to 3.5s per eval iteration). Since this logic is pure Python/Cython and runs on the CPU, it does not benefit from GPU acceleration, further diluting any potential CUDA wins.

## Practical Implications

- **Device Choice:** For the current serial evaluation implementation, always use `--device cpu` for evaluation. If training on CUDA, transferring weights to a CPU-based evaluation worker is significantly more efficient than evaluating on the GPU.
- **Batched Evaluation:** To make CUDA evaluation viable, the implementation must be refactored to use `select_actions_batch` across multiple concurrent games. This would move the evaluation pattern closer to the training pattern, allowing the GPU to process multiple info-states in a single kernel launch.
- **Model Scaling:** As the strategy network size increases (e.g., larger hidden layers or more blocks), the relative overhead of CUDA will decrease. At a certain model scale, the compute advantage will eventually overcome the latency penalty even at batch size 1.

## References

- `docs/archive/deep-cfr-evaluation-profile-2026-05-07.md` (Profiling source)
- `docs/performance.md` (Top-level performance log)
- `src/coolrl_lost_cities/games/classic/deep_cfr/evaluate.py` (Implementation)