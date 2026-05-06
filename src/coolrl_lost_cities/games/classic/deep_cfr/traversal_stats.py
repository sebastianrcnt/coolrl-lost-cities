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
    regret_matching_decisions: int = 0
    regret_fallback_count: int = 0
    regret_fallback_depth_sum: int = 0
    regret_fallback_opened_colors_sum: int = 0
    regret_fallback_legal_actions_sum: int = 0
    regret_fallback_legal_play_existing_sum: int = 0
    regret_fallback_legal_open_new_sum: int = 0
    regret_fallback_legal_discard_sum: int = 0
    regret_fallback_legal_draw_deck_sum: int = 0
    regret_fallback_legal_draw_pile_sum: int = 0
    regret_fallback_action_play_existing: int = 0
    regret_fallback_action_open_new: int = 0
    regret_fallback_action_discard: int = 0
    regret_fallback_action_draw_deck: int = 0
    regret_fallback_action_draw_pile: int = 0
    regret_fallback_argmax_tie_count: int = 0
    regret_fallback_argmax_tie_size_sum: int = 0
    regret_fallback_argmax_full_tie_count: int = 0
    regret_fallback_depth_buckets: dict[str, int] = field(default_factory=dict)
    regret_fallback_opened_colors_buckets: dict[int, int] = field(default_factory=dict)
    regret_fallback_open_new_available_by_color: dict[int, int] = field(default_factory=dict)
    regret_fallback_open_new_selected_by_color: dict[int, int] = field(default_factory=dict)
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
        self.regret_matching_decisions += other.regret_matching_decisions
        self.regret_fallback_count += other.regret_fallback_count
        self.regret_fallback_depth_sum += other.regret_fallback_depth_sum
        self.regret_fallback_opened_colors_sum += other.regret_fallback_opened_colors_sum
        self.regret_fallback_legal_actions_sum += other.regret_fallback_legal_actions_sum
        self.regret_fallback_legal_play_existing_sum += (
            other.regret_fallback_legal_play_existing_sum
        )
        self.regret_fallback_legal_open_new_sum += other.regret_fallback_legal_open_new_sum
        self.regret_fallback_legal_discard_sum += other.regret_fallback_legal_discard_sum
        self.regret_fallback_legal_draw_deck_sum += other.regret_fallback_legal_draw_deck_sum
        self.regret_fallback_legal_draw_pile_sum += other.regret_fallback_legal_draw_pile_sum
        self.regret_fallback_action_play_existing += other.regret_fallback_action_play_existing
        self.regret_fallback_action_open_new += other.regret_fallback_action_open_new
        self.regret_fallback_action_discard += other.regret_fallback_action_discard
        self.regret_fallback_action_draw_deck += other.regret_fallback_action_draw_deck
        self.regret_fallback_action_draw_pile += other.regret_fallback_action_draw_pile
        self.regret_fallback_argmax_tie_count += other.regret_fallback_argmax_tie_count
        self.regret_fallback_argmax_tie_size_sum += other.regret_fallback_argmax_tie_size_sum
        self.regret_fallback_argmax_full_tie_count += other.regret_fallback_argmax_full_tie_count
        for key, value in other.regret_fallback_depth_buckets.items():
            self.regret_fallback_depth_buckets[key] = (
                self.regret_fallback_depth_buckets.get(key, 0) + value
            )
        for key, value in other.regret_fallback_opened_colors_buckets.items():
            self.regret_fallback_opened_colors_buckets[key] = (
                self.regret_fallback_opened_colors_buckets.get(key, 0) + value
            )
        for key, value in other.regret_fallback_open_new_available_by_color.items():
            self.regret_fallback_open_new_available_by_color[key] = (
                self.regret_fallback_open_new_available_by_color.get(key, 0) + value
            )
        for key, value in other.regret_fallback_open_new_selected_by_color.items():
            self.regret_fallback_open_new_selected_by_color[key] = (
                self.regret_fallback_open_new_selected_by_color.get(key, 0) + value
            )
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

    @property
    def regret_fallback_rate(self) -> float:
        return self.regret_fallback_count / max(1, self.regret_matching_decisions)

    @property
    def regret_fallback_open_new_selected_rate(self) -> float:
        return self.regret_fallback_action_open_new / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_depth(self) -> float:
        return self.regret_fallback_depth_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_opened_colors_before_action(self) -> float:
        return self.regret_fallback_opened_colors_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_legal_actions(self) -> float:
        return self.regret_fallback_legal_actions_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_legal_play_existing(self) -> float:
        return self.regret_fallback_legal_play_existing_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_legal_open_new(self) -> float:
        return self.regret_fallback_legal_open_new_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_legal_discard(self) -> float:
        return self.regret_fallback_legal_discard_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_legal_draw_deck(self) -> float:
        return self.regret_fallback_legal_draw_deck_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_avg_legal_draw_pile(self) -> float:
        return self.regret_fallback_legal_draw_pile_sum / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_open_new_available_rate(self) -> float:
        return self.regret_fallback_legal_open_new_sum / max(
            1, self.regret_fallback_legal_actions_sum
        )

    @property
    def regret_fallback_open_new_selection_over_availability(self) -> float:
        available_rate = self.regret_fallback_open_new_available_rate
        if available_rate <= 0.0:
            return 0.0
        return self.regret_fallback_open_new_selected_rate / available_rate

    @property
    def regret_fallback_argmax_tie_rate(self) -> float:
        return self.regret_fallback_argmax_tie_count / max(1, self.regret_fallback_count)

    @property
    def regret_fallback_argmax_avg_tie_size(self) -> float:
        return self.regret_fallback_argmax_tie_size_sum / max(
            1, self.regret_fallback_argmax_tie_count
        )

    @property
    def regret_fallback_argmax_full_tie_rate(self) -> float:
        return self.regret_fallback_argmax_full_tie_count / max(1, self.regret_fallback_count)

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
            "traversal_regret_matching_decisions": self.regret_matching_decisions,
            "traversal_regret_fallback_count": self.regret_fallback_count,
            "traversal_regret_fallback_rate": self.regret_fallback_rate,
            "traversal_regret_fallback_avg_depth": self.regret_fallback_avg_depth,
            "traversal_regret_fallback_action_play_existing": self.regret_fallback_action_play_existing,
            "traversal_regret_fallback_action_open_new": self.regret_fallback_action_open_new,
            "traversal_regret_fallback_action_discard": self.regret_fallback_action_discard,
            "traversal_regret_fallback_action_draw_deck": self.regret_fallback_action_draw_deck,
            "traversal_regret_fallback_action_draw_pile": self.regret_fallback_action_draw_pile,
            "traversal_regret_fallback_legal_actions_mean": (
                self.regret_fallback_avg_legal_actions
            ),
            "traversal_regret_fallback_legal_play_existing_mean": (
                self.regret_fallback_avg_legal_play_existing
            ),
            "traversal_regret_fallback_legal_open_new_mean": (
                self.regret_fallback_avg_legal_open_new
            ),
            "traversal_regret_fallback_legal_discard_mean": (
                self.regret_fallback_avg_legal_discard
            ),
            "traversal_regret_fallback_legal_draw_deck_mean": (
                self.regret_fallback_avg_legal_draw_deck
            ),
            "traversal_regret_fallback_legal_draw_pile_mean": (
                self.regret_fallback_avg_legal_draw_pile
            ),
            "traversal_regret_fallback_open_new_available_rate": (
                self.regret_fallback_open_new_available_rate
            ),
            "traversal_regret_fallback_open_new_selected": self.regret_fallback_action_open_new,
            "traversal_regret_fallback_open_new_selected_rate": (
                self.regret_fallback_open_new_selected_rate
            ),
            "traversal_regret_fallback_open_new_selection_over_availability": (
                self.regret_fallback_open_new_selection_over_availability
            ),
            "traversal_regret_fallback_avg_opened_colors_before_action": (
                self.regret_fallback_avg_opened_colors_before_action
            ),
            "traversal_regret_fallback_argmax_tie_count": (self.regret_fallback_argmax_tie_count),
            "traversal_regret_fallback_argmax_tie_rate": (self.regret_fallback_argmax_tie_rate),
            "traversal_regret_fallback_argmax_tie_size_mean": (
                self.regret_fallback_argmax_avg_tie_size
            ),
            "traversal_regret_fallback_argmax_full_tie_count": (
                self.regret_fallback_argmax_full_tie_count
            ),
            "traversal_regret_fallback_argmax_full_tie_rate": (
                self.regret_fallback_argmax_full_tie_rate
            ),
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
            **{
                f"traversal_regret_fallback_depth_bucket_{key}": value
                for key, value in self.regret_fallback_depth_buckets.items()
            },
            **{
                f"traversal_regret_fallback_opened_colors_count_{key}": value
                for key, value in self.regret_fallback_opened_colors_buckets.items()
            },
            **{
                f"traversal_regret_fallback_open_new_available_color_{key}": value
                for key, value in self.regret_fallback_open_new_available_by_color.items()
            },
            **{
                f"traversal_regret_fallback_open_new_selected_color_{key}": value
                for key, value in self.regret_fallback_open_new_selected_by_color.items()
            },
        }
