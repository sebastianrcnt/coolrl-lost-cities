from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraversalStats:
    nodes: int = 0
    terminals: int = 0
    depth_cutoffs: int = 0
    node_limit_cutoffs: int = 0
    max_depth_reached: int = 0
    advantage_samples: int = 0
    strategy_samples: int = 0
    sampled_actions: int = 0
    cutoff_rollouts: int = 0
    cutoff_rollout_steps: int = 0
    cutoff_rollout_timeouts: int = 0
    endpoint_depth_sum: int = 0
    endpoint_depth_buckets: dict[str, int] = field(default_factory=dict)

    def accumulate(self, other: TraversalStats) -> None:
        self.nodes += other.nodes
        self.terminals += other.terminals
        self.depth_cutoffs += other.depth_cutoffs
        self.node_limit_cutoffs += other.node_limit_cutoffs
        self.max_depth_reached = max(self.max_depth_reached, other.max_depth_reached)
        self.advantage_samples += other.advantage_samples
        self.strategy_samples += other.strategy_samples
        self.sampled_actions += other.sampled_actions
        self.cutoff_rollouts += other.cutoff_rollouts
        self.cutoff_rollout_steps += other.cutoff_rollout_steps
        self.cutoff_rollout_timeouts += other.cutoff_rollout_timeouts
        self.endpoint_depth_sum += other.endpoint_depth_sum
        for key, value in other.endpoint_depth_buckets.items():
            self.endpoint_depth_buckets[key] = self.endpoint_depth_buckets.get(key, 0) + value

    @property
    def endpoints(self) -> int:
        return self.terminals + self.depth_cutoffs + self.node_limit_cutoffs

    @property
    def avg_endpoint_depth(self) -> float:
        return self.endpoint_depth_sum / max(1, self.endpoints)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "traversal_nodes": self.nodes,
            "traversal_terminals": self.terminals,
            "traversal_depth_cutoffs": self.depth_cutoffs,
            "traversal_node_limit_cutoffs": self.node_limit_cutoffs,
            "traversal_max_depth_reached": self.max_depth_reached,
            "traversal_advantage_samples": self.advantage_samples,
            "traversal_strategy_samples": self.strategy_samples,
            "traversal_sampled_actions": self.sampled_actions,
            "traversal_cutoff_rollouts": self.cutoff_rollouts,
            "traversal_cutoff_rollout_steps": self.cutoff_rollout_steps,
            "traversal_cutoff_rollout_timeouts": self.cutoff_rollout_timeouts,
            "traversal_endpoint_depth_sum": self.endpoint_depth_sum,
            "traversal_endpoints": self.endpoints,
            "traversal_avg_endpoint_depth": self.avg_endpoint_depth,
            **{
                f"traversal_endpoint_depth_bucket_{key}": value
                for key, value in self.endpoint_depth_buckets.items()
            },
        }
