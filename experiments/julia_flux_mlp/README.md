# Julia Flux MLP Forward Benchmark

Criterion 4 for the Julia port evaluation.

Compares PyTorch `DeepCFRMLP` against a Flux/CUDA implementation with the
same shape and identical exported weights:

- input dim: 365
- hidden size: 512
- hidden layers: 3
- output dim: 22
- activation: ReLU

Run:

```bash
uv run python experiments/julia_flux_mlp/bench_pytorch_vs_flux.py
```

The runner exports PyTorch weights and inputs to `runs/tmp/`, runs Flux on the
same payload, checks output equivalence, then writes `results.json`.

## Results (2026-05-07)

Host GPU: NVIDIA GeForce RTX 3090. Timing uses 10 warmup forwards, then 100
timed forwards, with CUDA synchronized around the timed loop in both runtimes.

| Backend | batch | forward ms | μs/state | ratio vs PyTorch |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 1 | 0.0829 | 82.8755 | 1.00× |
| Flux | 1 | 0.1669 | 166.8867 | 2.01× |
| PyTorch | 64 | 0.0927 | 1.4477 | 1.00× |
| Flux | 64 | 0.1909 | 2.9836 | 2.06× |
| PyTorch | 256 | 0.0877 | 0.3427 | 1.00× |
| Flux | 256 | 0.1758 | 0.6867 | 2.00× |

Maximum output difference: `5.215e-08`.

Criterion 4 threshold:

- bs=64 must be within ±20% of PyTorch.
- bs=1 and bs=256 must be within ±30% of PyTorch.

**Verdict:** FAIL. Flux/CUDA is ~2.0× slower than PyTorch at all measured
batch sizes for this model shape.
