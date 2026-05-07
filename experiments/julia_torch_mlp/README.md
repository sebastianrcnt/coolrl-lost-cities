# Julia Torch MLP Forward Benchmark

Attempted Criterion 4 re-measurement using Torch.jl for the same
`DeepCFRMLP` shape used in `experiments/julia_flux_mlp/`:

- input dim: 365
- hidden size: 512
- hidden layers: 3
- output dim: 22
- activation: ReLU

Run:

```bash
tools/julia/current/bin/julia --project=experiments/julia_torch_mlp \
  experiments/julia_torch_mlp/bench_torch.jl
```

## Result (2026-05-07)

Torch.jl could not be loaded on this host:

```text
Torch.jl v0.1.3 failed to precompile:
UndefVarError: libtorch_c_api not defined in Torch.Wrapper
```

This happens before any model construction or timing is possible, so no
DeepCFRMLP forward benchmark could be run.

**Verdict:** BLOCKED. Criterion 4 remains FAIL based on the completed Flux.jl
measurement. Torch.jl does not reverse the decision because it is not usable in
the current Julia 1.11.9 environment.
