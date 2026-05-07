# Deep CFR Legacy Parity and Hyperparameter Mapping

**Last verified:** 2026-05-08, commit `5c221fb`

Source: `docs/archive/deep-cfr-legacy-experiment-reproduction.md`

## Question

How are legacy `coolrl` Deep CFR hyperparameters and feature semantics mapped to the current implementation to ensure parity and meaningful experiment reproduction?

## Code reference

- `src/coolrl_lost_cities/games/classic/deep_cfr/config.py`, lines 63–64 (`EncodingConfig`) and 98–99 (`TraversalConfig`): Definition of parity-critical flags and traversal parameters.
- `src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx`, lines 156 and 209: Implementation of `_append_derived_playability_features_c` and `_append_slot_aware_playability_features_c`.
- `src/coolrl_lost_cities/games/classic/deep_cfr/networks.py`, lines 21–33: `_build_mlp` implementation that supports the legacy 3-layer, 256-hidden-unit architecture.
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`, lines 408–411 and 606–607: Logic for outcome-sampling mixture and value clipping.

## Analysis

Reproduction of legacy Deep CFR experiments requires strict adherence to both hyperparameter values and specific feature engineering. The mapping between the legacy `coolrl` environment and this repository is achieved through the following core components:

### 1. Information-State Encoding
The legacy experiment relied on specialized features beyond the raw board state. These are preserved through two critical flags in `EncodingConfig`:
- **`derived_playability`**: Color-level features that calculate the utility or risk of playing into specific expeditions.
- **`slot_aware_playability`**: Hand-slot local features that anchor actions to specific hand positions. This allows the model to distinguish between identical cards in different slots or empty slots, which is critical for high-tier play.

### 2. Network Architecture
The current `DeepCFRMLP` supports configurable `hidden_size` and `num_layers`. To match legacy performance, the model must be configured with a 3-layer ReLU MLP (hidden size 256). The `_build_mlp` helper in `networks.py` ensures the layer stack is constructed identically to the legacy Torch implementation.

### 3. Traversal and Optimization
Deep CFR performance is highly sensitive to the traversal mechanism. The current implementation matches legacy semantics via:
- **Outcome Sampling Mixture**: On-policy sampling mixed with ε-uniform exploration (`outcome_sampling_epsilon: 0.2`).
- **Value Clipping**: Restricting the range of sampled values (`outcome_sampling_value_clip: 500`) to prevent gradient instability.
- **Batching**: Separate advantage and strategy batch sizes (legacy default: 1024) and update counts (legacy default: 256 per iteration).

## Practical implication

Maintaining this mapping allows for direct comparison between modern runs and legacy baselines. The **slot-aware encoding** is identified as the single most critical factor for parity in tier3 rulesets; without it, the information-state is insufficiently descriptive for the policy network to replicate legacy performance.

When evaluating current performance against legacy reports, ensure the `encoding.slot_aware_playability` flag is enabled and the `traversal` parameters match the 0.2/500 epsilon/clip baseline.

## References

- `docs/archive/deep-cfr-legacy-experiment-reproduction.md` (Reproduction plan)
- `configs/archive/deep-cfr-selfplay-full-depth-slot-playability.yaml` (Reference configuration)
- `docs/research/deep-cfr-v0-feature-parity.md` (General subsystem coverage)