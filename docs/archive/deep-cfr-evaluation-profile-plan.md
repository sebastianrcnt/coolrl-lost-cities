# Deep CFR Evaluation Profile Plan

Goal: split evaluation runtime into the main per-step costs so eval iteration
spikes can be explained without guessing.

The profiling run should keep the normal full config and `eval_every: 5`, then
compare iterations 5 and 10.

## Metrics

Each metric is emitted with the existing opponent prefix:
`eval_<opponent>_<metric_name>`.

Top-level counters:

- `policy_turns`
- `opponent_turns`
- `avg_game_length`
- `elapsed_seconds`
- `games_per_second`
- `steps_per_second`

Step-level runtime:

- `policy_select_seconds`
- `opponent_act_seconds`
- `apply_action_seconds`
- `diagnostics_seconds`
- `final_scoring_seconds`

Policy select breakdown:

- `policy_legal_mask_seconds`
- `policy_encoding_seconds`
- `policy_network_seconds`
- `policy_postprocess_seconds`

## Interpretation

If `policy_network_seconds` dominates, evaluation is mostly batch-size-1 model
inference overhead.

If `policy_encoding_seconds` or `policy_legal_mask_seconds` dominates, the eval
policy path is paying per-turn state feature/mask construction costs.

If `opponent_act_seconds` dominates, the opponent bot implementation is the
main eval cost for that opponent.

If `apply_action_seconds` dominates, the game step path itself is the eval
bottleneck.
