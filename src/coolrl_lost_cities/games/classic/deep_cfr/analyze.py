from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SMOOTHING_WINDOW = 1


@dataclass(frozen=True)
class PlotSpec:
    title: str
    metrics: tuple[str, ...]
    ylabel: str
    scale: float = 1.0
    kind: str = "eval"
    fixed_ylim: tuple[float, float] | None = None


@dataclass(frozen=True)
class SectionSpec:
    name: str
    filename: str
    plots: tuple[PlotSpec, ...]


SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        "Loss",
        "analysis_01_loss.png",
        (
            PlotSpec("Advantage Loss", ("advantage_loss",), "loss", kind="train"),
            PlotSpec("Strategy Loss", ("strategy_loss",), "loss", kind="train"),
            PlotSpec(
                "Samples",
                ("advantage_samples", "strategy_samples"),
                "samples",
                kind="train",
            ),
            PlotSpec(
                "Memory Size",
                ("advantage_memory_size", "strategy_memory_size"),
                "samples",
                kind="train",
            ),
        ),
    ),
    SectionSpec(
        "Match",
        "analysis_02_match.png",
        (
            PlotSpec("Win Rate", ("win_rate0",), "rate (%)", scale=100.0, fixed_ylim=(0, 100)),
            PlotSpec("Avg Score Diff", ("avg_score_diff0",), "score diff"),
            PlotSpec("Avg Score", ("avg_score0",), "score"),
            PlotSpec("Policy Entropy", ("policy_entropy",), "entropy"),
        ),
    ),
    SectionSpec(
        "Action",
        "analysis_03_action.png",
        (
            PlotSpec(
                "Play Action Rate",
                ("play_action_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Discard Action Rate",
                ("discard_action_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Draw Deck Rate",
                ("draw_deck_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Draw Pile Rate",
                ("draw_pile_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
        ),
    ),
    SectionSpec(
        "GameFlow",
        "analysis_04_gameflow.png",
        (
            PlotSpec("Opened Colors", ("avg_opened_colors",), "colors", fixed_ylim=(0, 5)),
            PlotSpec("Opened Colors Std", ("opened_colors_std",), "std"),
            PlotSpec(
                "5-Color Open Count",
                ("5_color_open_count",),
                "games / eval",
                fixed_ylim=(0, 100),
            ),
            PlotSpec("Expedition Cards", ("avg_expedition_cards",), "cards"),
        ),
    ),
    SectionSpec(
        "OpenQuality",
        "analysis_05_open_quality.png",
        (
            PlotSpec(
                "Bad Open Rate",
                ("bad_open_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Weak Open Rate",
                ("weak_open_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Bad or Weak Open Rate",
                ("bad_or_weak_open_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Good Open Rate",
                ("good_open_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec("Bad Open per Game", ("bad_open_per_game",), "opens / game"),
            PlotSpec(
                "Bad or Weak Open per Game",
                ("bad_or_weak_open_per_game",),
                "opens / game",
            ),
            PlotSpec("Opening Play Actions", ("opening_play_actions",), "actions / game"),
            PlotSpec("Opening Recoverable p25", ("opening_recoverable_score_p25",), "score"),
        ),
    ),
    SectionSpec(
        "ExpeditionOutcomes",
        "analysis_06_expedition_outcomes.png",
        (
            PlotSpec(
                "Positive Expedition Rate",
                ("positive_expedition_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Negative Expedition Rate",
                ("negative_expedition_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Bonus Expedition Rate",
                ("bonus_expedition_rate",),
                "rate (%)",
                scale=100.0,
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Per-Game Expedition Counts",
                (
                    "per_game_positive_expeditions",
                    "per_game_negative_expeditions",
                    "per_game_breakeven_expeditions",
                    "per_game_below_minus_20_expeditions",
                ),
                "expeditions / game",
            ),
            PlotSpec("Final Expedition Score", ("avg_final_score_per_opened_expedition",), "score"),
            PlotSpec("Score per Opened Color", ("score_per_opened_color",), "score / color"),
        ),
    ),
    SectionSpec(
        "Calibration",
        "analysis_07_calibration.png",
        (
            PlotSpec(
                "First Open Recoverable Score",
                (
                    "first_open_recoverable_score_mean_for_positive_final",
                    "first_open_recoverable_score_mean_for_negative_final",
                ),
                "score",
            ),
            PlotSpec("Opening Recoverable Mean", ("opening_recoverable_score_mean",), "score"),
            PlotSpec("Opening Recoverable p25", ("opening_recoverable_score_p25",), "score"),
            PlotSpec("Calibration Gap", ("calibration_gap",), "score"),
        ),
    ),
    SectionSpec(
        "Traversal",
        "analysis_08_traversal.png",
        (
            PlotSpec("Iteration Time", ("iteration_seconds",), "seconds", kind="train"),
            PlotSpec("Throughput", ("nodes_per_second",), "nodes / second", kind="train"),
            PlotSpec(
                "Traversal Depth",
                ("traversal_avg_endpoint_depth", "traversal_max_depth_reached"),
                "depth",
                kind="train",
            ),
            PlotSpec(
                "Traversal Endpoints",
                ("traversal_terminals", "traversal_node_limit_cutoffs", "traversal_depth_cutoffs"),
                "count",
                kind="train",
            ),
            PlotSpec(
                "Traversal Endpoint Rates",
                (
                    "traversal_terminal_rate",
                    "traversal_node_limit_cutoff_rate",
                    "traversal_depth_cutoff_rate",
                ),
                "rate (%)",
                scale=100.0,
                kind="train",
                fixed_ylim=(0, 100),
            ),
            PlotSpec("Traversal Nodes", ("traversal_nodes",), "nodes", kind="train"),
            PlotSpec(
                "Regret Fallback Rate",
                ("traversal_regret_fallback_rate",),
                "rate (%)",
                scale=100.0,
                kind="train",
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Regret Fallback Count",
                ("traversal_regret_fallback_count",),
                "count",
                kind="train",
            ),
            PlotSpec(
                "Fallback Selected Actions",
                (
                    "traversal_regret_fallback_action_play_existing",
                    "traversal_regret_fallback_action_open_new",
                    "traversal_regret_fallback_action_discard",
                    "traversal_regret_fallback_action_draw_deck",
                    "traversal_regret_fallback_action_draw_pile",
                ),
                "count",
                kind="train",
            ),
            PlotSpec(
                "Fallback Open-New Rates",
                (
                    "traversal_regret_fallback_open_new_available_rate",
                    "traversal_regret_fallback_open_new_selected_rate",
                ),
                "rate (%)",
                scale=100.0,
                kind="train",
                fixed_ylim=(0, 100),
            ),
            PlotSpec(
                "Fallback Open-New Bias",
                ("traversal_regret_fallback_open_new_selection_over_availability",),
                "selected / available",
                kind="train",
            ),
            PlotSpec(
                "Fallback Avg Depth",
                ("traversal_regret_fallback_avg_depth",),
                "depth",
                kind="train",
            ),
            PlotSpec(
                "Fallback Opened Colors Before Action",
                ("traversal_regret_fallback_avg_opened_colors_before_action",),
                "colors",
                kind="train",
            ),
            PlotSpec(
                "Fallback Legal Actions Mean",
                ("traversal_regret_fallback_legal_actions_mean",),
                "value",
                kind="train",
            ),
            PlotSpec(
                "Argmax Tie Diagnostics",
                (
                    "traversal_regret_fallback_argmax_tie_rate",
                    "traversal_regret_fallback_argmax_full_tie_rate",
                    "traversal_regret_fallback_argmax_tie_size_mean",
                ),
                "rate / size",
                kind="train",
            ),
        ),
    ),
)

SELECTIVITY_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec("Opened Colors", ("avg_opened_colors",), "colors"),
    PlotSpec("5-Color Open Count", ("5_color_open_count",), "games / eval"),
    PlotSpec("Opening Recoverable Mean", ("opening_recoverable_score_mean",), "score"),
    PlotSpec("Calibration Gap", ("calibration_gap",), "score"),
    PlotSpec(
        "First Open Recoverable Score",
        (
            "first_open_recoverable_score_mean_for_positive_final",
            "first_open_recoverable_score_mean_for_negative_final",
        ),
        "score",
    ),
    PlotSpec("Opening Play Actions", ("opening_play_actions",), "actions / game"),
)

SUMMARY_EVAL_METRICS: tuple[tuple[str, str, float], ...] = (
    ("win_rate0", "win rate (%)", 100.0),
    ("avg_score_diff0", "avg score diff", 1.0),
    ("avg_score0", "avg score", 1.0),
    ("play_action_rate", "play rate (%)", 100.0),
    ("avg_opened_colors", "opened colors", 1.0),
    ("bad_open_rate", "bad open (%)", 100.0),
    ("score_per_opened_color", "score / opened color", 1.0),
    ("calibration_gap", "calibration gap", 1.0),
    ("bonus_contribution_per_game", "bonus / game", 1.0),
)

OPPONENT_COLORS: dict[str, str] = {
    "noisy_safe": "tab:blue",
    "passive_discard": "tab:orange",
    "random": "tab:green",
    "safe_heuristic": "tab:red",
    "safe_heuristic_loose": "tab:purple",
    "safe_heuristic_strict": "tab:brown",
}

TRAVERSAL_COLORS: dict[str, str] = {
    "iteration_seconds": "#4c78a8",
    "nodes_per_second": "#4c78a8",
    "traversal_avg_endpoint_depth": "#72b7b2",
    "traversal_max_depth_reached": "#f58518",
    "traversal_terminals": "#54a24b",
    "traversal_node_limit_cutoffs": "#e45756",
    "traversal_depth_cutoffs": "#b279a2",
    "traversal_terminal_rate": "#54a24b",
    "traversal_node_limit_cutoff_rate": "#e45756",
    "traversal_depth_cutoff_rate": "#b279a2",
    "traversal_nodes": "#4c78a8",
    "traversal_regret_fallback_rate": "#e45756",
    "traversal_regret_fallback_count": "#e45756",
    "traversal_regret_fallback_action_play_existing": "#4c78a8",
    "traversal_regret_fallback_action_open_new": "#f58518",
    "traversal_regret_fallback_action_discard": "#54a24b",
    "traversal_regret_fallback_action_draw_deck": "#b279a2",
    "traversal_regret_fallback_action_draw_pile": "#72b7b2",
    "traversal_regret_fallback_open_new_available_rate": "#9d755d",
    "traversal_regret_fallback_open_new_selected_rate": "#f58518",
    "traversal_regret_fallback_open_new_selection_over_availability": "#f58518",
    "traversal_regret_fallback_avg_depth": "#72b7b2",
    "traversal_regret_fallback_avg_opened_colors_before_action": "#f58518",
    "traversal_regret_fallback_legal_actions_mean": "#54a24b",
    "traversal_regret_fallback_argmax_tie_rate": "#e45756",
    "traversal_regret_fallback_argmax_full_tie_rate": "#b279a2",
    "traversal_regret_fallback_argmax_tie_size_mean": "#4c78a8",
}

ACTION_RATE_METRICS = {
    "play_action_rate",
    "discard_action_rate",
    "draw_deck_rate",
    "draw_pile_rate",
}


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def opponent_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key in row:
            if not key.startswith("eval_"):
                continue
            rest = key[len("eval_") :]
            for metric in _all_eval_metrics():
                suffix = f"_{metric}"
                if rest.endswith(suffix):
                    names.add(rest[: -len(suffix)])
    return sorted(names)


def plot_section(
    rows: list[dict[str, Any]],
    section: SectionSpec,
    output: Path,
    *,
    smoothing_window: int,
) -> bool:
    import matplotlib.pyplot as plt

    opponents = opponent_names(rows)
    cols = 2
    rows_count = math.ceil((len(section.plots) + _extra_plot_count(section)) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(16, 4.2 * rows_count), squeeze=False)
    axes_flat = list(axes.flat)
    plotted_any = False

    for ax, spec in zip(axes_flat, section.plots, strict=False):
        if spec.kind == "train":
            plotted = _plot_train_spec(ax, rows, spec, smoothing_window=smoothing_window)
        else:
            plotted = _plot_eval_spec(ax, rows, opponents, spec, smoothing_window=smoothing_window)
        _finish_axis(
            ax, spec.title, ylabel=spec.ylabel, plotted=plotted, fixed_ylim=spec.fixed_ylim
        )
        if spec.title == "Fallback Open-New Bias" and plotted:
            ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
        plotted_any = plotted_any or plotted

    next_axis = len(section.plots)
    if section.name == "Traversal" and next_axis < len(axes_flat):
        _plot_latest_depth_buckets(axes_flat[next_axis], rows)
        plotted_any = plotted_any or bool(_latest_depth_buckets(rows))
        next_axis += 1

    for ax in axes_flat[next_axis:]:
        ax.axis("off")

    if section.name != "Traversal":
        handles, labels = _legend_items(axes_flat)
        if handles:
            fig.legend(
                handles, labels, loc="upper center", ncols=min(len(labels), 6), fontsize="small"
            )
    suffix = f" ({smoothing_window}-iter moving average)" if smoothing_window > 1 else ""
    fig.suptitle(
        f"Lost Cities Deep CFR {section.name} metrics{suffix}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if not plotted_any:
        plt.close(fig)
        return False
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def plot_final_eval_summary(rows: list[dict[str, Any]], output: Path) -> bool:
    import matplotlib.pyplot as plt

    opponents = opponent_names(rows)
    latest = _latest_eval_row(rows)
    if not opponents or latest is None:
        return False

    cols = 3
    rows_count = math.ceil(len(SUMMARY_EVAL_METRICS) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(18, 4.2 * rows_count), squeeze=False)
    plotted_any = False
    for ax, (metric, title, scale) in zip(axes.flat, SUMMARY_EVAL_METRICS, strict=False):
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for opponent in opponents:
            value = _eval_value(latest, opponent, metric)
            if value is None or not math.isfinite(value):
                continue
            labels.append(opponent)
            values.append(value * scale)
            colors.append(_opponent_color(opponent))
        if values:
            ax.bar(labels, values, color=colors)
            plotted_any = True
            ax.tick_params(axis="x", labelrotation=35, labelsize="x-small")
        fixed_ylim = (0, 100) if "rate (%)" in title or title == "opened colors" else None
        if title == "opened colors":
            fixed_ylim = (0, 5)
        _finish_axis(ax, title, xlabel="", plotted=bool(values), fixed_ylim=fixed_ylim)

    for ax in list(axes.flat)[len(SUMMARY_EVAL_METRICS) :]:
        ax.axis("off")

    iteration = latest.get("iteration", "latest")
    fig.suptitle(
        f"Lost Cities Deep CFR final eval summary: iteration {iteration}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    if not plotted_any:
        plt.close(fig)
        return False
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def analyze_run(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    max_iteration: int | None = None,
) -> list[Path]:
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metrics(metrics_path)
    if max_iteration is not None:
        rows = [
            row for row in rows if "iteration" in row and int(row["iteration"]) <= max_iteration
        ]
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    filename_suffix = _iteration_filename_suffix(max_iteration)

    for section in SECTIONS:
        path = output_dir / _with_filename_suffix(section.filename, filename_suffix)
        if plot_section(rows, section, path, smoothing_window=smoothing_window):
            written.append(path)

    selectivity_path = output_dir / _with_filename_suffix(
        "analysis_09_selectivity.png", filename_suffix
    )
    if plot_selectivity(rows, selectivity_path, smoothing_window=smoothing_window):
        written.append(selectivity_path)

    final_eval_path = output_dir / _with_filename_suffix(
        "analysis_final_eval_summary.png", filename_suffix
    )
    if plot_final_eval_summary(rows, final_eval_path):
        written.append(final_eval_path)
    return written


def _iteration_filename_suffix(max_iteration: int | None) -> str:
    if max_iteration is None:
        return ""
    return f"_upto_{max_iteration:05d}"


def _with_filename_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix}"


def plot_selectivity(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    smoothing_window: int,
) -> bool:
    import matplotlib.pyplot as plt

    opponents = opponent_names(rows)
    if not opponents:
        return False

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), squeeze=False)
    axes_flat = list(axes.flat)
    plotted_any = False

    for ax, spec in zip(axes_flat, SELECTIVITY_PLOTS, strict=True):
        plotted = _plot_eval_spec(ax, rows, opponents, spec, smoothing_window=smoothing_window)
        _finish_axis(ax, spec.title, ylabel=spec.ylabel, plotted=plotted)
        plotted_any = plotted_any or plotted

    handles, labels = _legend_items(axes_flat)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 6), fontsize="small")
    suffix = f" ({smoothing_window}-iter moving average)" if smoothing_window > 1 else ""
    fig.suptitle(
        f"Lost Cities Deep CFR selectivity metrics{suffix}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if not plotted_any:
        plt.close(fig)
        return False
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def _all_eval_metrics() -> set[str]:
    metrics: set[str] = set()
    for section in SECTIONS:
        for plot in section.plots:
            if plot.kind == "eval":
                metrics.update(plot.metrics)
    metrics.update(metric for metric, _, _ in SUMMARY_EVAL_METRICS)
    metrics.update(_base_metrics_for_derived_values())
    return metrics


def _plot_train_spec(
    ax: Any,
    rows: list[dict[str, Any]],
    spec: PlotSpec,
    *,
    smoothing_window: int,
) -> bool:
    x = [int(row["iteration"]) for row in rows if "iteration" in row]
    if not x:
        return False

    plotted = False
    for metric in spec.metrics:
        pairs: list[tuple[int, float]] = []
        for row in rows:
            if "iteration" not in row:
                continue
            value = _train_value(row, metric)
            if value is None:
                continue
            pairs.append((int(row["iteration"]), value * spec.scale))
        plotted = (
            _plot_pairs(
                ax,
                pairs,
                label=_train_metric_label(metric),
                color=_train_metric_color(metric, section_title=spec.title),
                smoothing_window=smoothing_window,
            )
            or plotted
        )
    return plotted


def _train_value(row: dict[str, Any], metric: str) -> float | None:
    if metric == "traversal_terminal_rate":
        return _ratio(row, "traversal_terminals", "traversal_endpoints")
    if metric == "traversal_node_limit_cutoff_rate":
        return _ratio(row, "traversal_node_limit_cutoffs", "traversal_endpoints")
    if metric == "traversal_depth_cutoff_rate":
        return _ratio(row, "traversal_depth_cutoffs", "traversal_endpoints")
    value = row.get(metric)
    if value is None:
        return None
    return float(value)


def _ratio(row: dict[str, Any], numerator: str, denominator: str) -> float | None:
    num = row.get(numerator)
    den = row.get(denominator)
    if num is None or den is None or float(den) == 0.0:
        return None
    return float(num) / float(den)


def _plot_eval_spec(
    ax: Any,
    rows: list[dict[str, Any]],
    opponents: list[str],
    spec: PlotSpec,
    *,
    smoothing_window: int,
) -> bool:
    plotted = False
    multi_metric = len(spec.metrics) > 1
    for opponent in opponents:
        for metric in spec.metrics:
            pairs: list[tuple[int, float]] = []
            for row in rows:
                if "iteration" not in row:
                    continue
                value = _eval_value(row, opponent, metric)
                if value is None:
                    continue
                if _should_mask_open_rate(row, opponent, metric):
                    value = float("nan")
                pairs.append((int(row["iteration"]), value * spec.scale))

            label = opponent
            if multi_metric:
                label = f"{opponent}: {_short_metric_label(metric)}"
            plotted = (
                _plot_pairs(
                    ax,
                    pairs,
                    label=label,
                    color=_opponent_color(opponent),
                    linestyle=_metric_linestyle(metric) if multi_metric else "-",
                    smoothing_window=smoothing_window,
                )
                or plotted
            )
    return plotted


def _plot_pairs(
    ax: Any,
    pairs: list[tuple[int, float]],
    *,
    label: str,
    color: str | None,
    smoothing_window: int,
    linestyle: str = "-",
) -> bool:
    if not pairs:
        return False
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    y = _moving_average(y, smoothing_window)
    if all(not math.isfinite(value) for value in y):
        return False
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.5,
        markersize=3,
        label=label,
        color=color,
        linestyle=linestyle,
    )
    return True


def _eval_value(row: dict[str, Any], opponent: str, metric: str) -> float | None:
    if metric == "bad_open_per_game":
        return _first_existing_eval(row, opponent, ("bad_open_per_game", "bad_open_actions"))
    if metric == "bad_or_weak_open_per_game":
        direct = _first_existing_eval(row, opponent, ("bad_or_weak_open_per_game",))
        if direct is not None:
            return direct
        bad = _first_existing_eval(row, opponent, ("bad_open_actions", "bad_open_per_game"))
        weak = _first_existing_eval(row, opponent, ("weak_open_actions", "weak_open_per_game"))
        if bad is None and weak is None:
            return None
        return (bad or 0.0) + (weak or 0.0)
    if metric == "bad_or_weak_open_rate":
        direct = _first_existing_eval(row, opponent, ("bad_or_weak_open_rate",))
        if direct is not None:
            return direct
        bad = _first_existing_eval(row, opponent, ("bad_open_rate",))
        weak = _first_existing_eval(row, opponent, ("weak_open_rate",))
        if bad is None and weak is None:
            return None
        return (bad or 0.0) + (weak or 0.0)
    if metric == "calibration_gap":
        positive = _first_existing_eval(
            row, opponent, ("first_open_recoverable_score_mean_for_positive_final",)
        )
        negative = _first_existing_eval(
            row, opponent, ("first_open_recoverable_score_mean_for_negative_final",)
        )
        if positive is None or negative is None:
            return None
        return positive - negative
    if metric == "bonus_contribution_per_game":
        per_game_bonus = _first_existing_eval(row, opponent, ("per_game_bonus_expeditions",))
        if per_game_bonus is not None:
            return per_game_bonus * 20.0
        bonus_rate = _first_existing_eval(row, opponent, ("bonus_expedition_rate",))
        opened_colors = _first_existing_eval(row, opponent, ("avg_opened_colors",))
        if bonus_rate is None or opened_colors is None:
            return None
        return bonus_rate * opened_colors * 20.0
    return _first_existing_eval(row, opponent, (metric,))


def _first_existing_eval(
    row: dict[str, Any], opponent: str, metrics: tuple[str, ...]
) -> float | None:
    for metric in metrics:
        value = row.get(f"eval_{opponent}_{metric}")
        if value is not None:
            return float(value)
    return None


def _should_mask_open_rate(row: dict[str, Any], opponent: str, metric: str) -> bool:
    if metric in ACTION_RATE_METRICS or not metric.endswith("_rate"):
        return False
    opening_play_actions = _first_existing_eval(row, opponent, ("opening_play_actions",))
    return opening_play_actions is not None and opening_play_actions < 1.0


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = [value for value in values[start : idx + 1] if math.isfinite(value)]
        if not window_values:
            smoothed.append(float("nan"))
        else:
            smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def _plot_latest_depth_buckets(ax: Any, rows: list[dict[str, Any]]) -> None:
    buckets = _latest_depth_buckets(rows)
    if not buckets:
        _finish_axis(ax, "Latest Endpoint Depth Buckets", xlabel="endpoint depth", plotted=False)
        return
    labels = [label for label, _ in buckets]
    values = [value for _, value in buckets]
    ax.bar(labels, values, color="tab:blue")
    ax.tick_params(axis="x", labelrotation=45, labelsize="x-small")
    _finish_axis(
        ax,
        "Latest Endpoint Depth Buckets",
        xlabel="endpoint depth",
        ylabel="traversals",
        plotted=True,
    )


def _latest_depth_buckets(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    for row in reversed(rows):
        buckets = [
            (
                key.removeprefix("traversal_endpoint_depth_bucket_"),
                float(value),
            )
            for key, value in row.items()
            if key.startswith("traversal_endpoint_depth_bucket_")
        ]
        if buckets:
            return sorted(buckets, key=lambda item: item[0])
    return []


def _latest_eval_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        if any(key.startswith("eval_") for key in row):
            return row
    return None


def _finish_axis(
    ax: Any,
    title: str,
    *,
    xlabel: str = "iteration",
    ylabel: str | None = None,
    plotted: bool,
    fixed_ylim: tuple[float, float] | None = None,
) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    ax.grid(True, alpha=0.3)
    if plotted:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="x-small")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def _legend_items(axes: list[Any]) -> tuple[list[Any], list[str]]:
    handles_by_label: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            handles_by_label.setdefault(label, handle)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    labels = list(handles_by_label)
    return [handles_by_label[label] for label in labels], labels


def _opponent_color(opponent: str) -> str:
    return OPPONENT_COLORS.get(opponent, "tab:gray")


def _train_metric_color(metric: str, *, section_title: str) -> str | None:
    if section_title.startswith("Fallback") or section_title.startswith("Traversal"):
        return TRAVERSAL_COLORS.get(metric)
    if metric.startswith("traversal_"):
        return TRAVERSAL_COLORS.get(metric)
    return None


def _metric_linestyle(metric: str) -> str:
    if "negative" in metric or metric.endswith("_negative_final"):
        return "--"
    if "breakeven" in metric or "weak" in metric:
        return ":"
    if "below_minus_20" in metric:
        return "-."
    return "-"


def _label(metric: str) -> str:
    return metric.replace("_", " ")


def _train_metric_label(metric: str) -> str:
    labels = {
        "advantage_memory_size": "advantage",
        "advantage_samples": "advantage",
        "iteration_seconds": "iteration",
        "nodes_per_second": "nodes/sec",
        "strategy_memory_size": "strategy",
        "strategy_samples": "strategy",
        "traversal_avg_endpoint_depth": "avg endpoint depth",
        "traversal_depth_cutoff_rate": "depth cutoff",
        "traversal_depth_cutoffs": "depth cutoff",
        "traversal_max_depth_reached": "max depth",
        "traversal_node_limit_cutoff_rate": "node limit cutoff",
        "traversal_node_limit_cutoffs": "node limit cutoff",
        "traversal_nodes": "nodes",
        "traversal_regret_fallback_action_discard": "discard",
        "traversal_regret_fallback_action_draw_deck": "draw deck",
        "traversal_regret_fallback_action_draw_pile": "draw pile",
        "traversal_regret_fallback_action_open_new": "open new",
        "traversal_regret_fallback_action_play_existing": "play existing",
        "traversal_regret_fallback_argmax_full_tie_rate": "full tie rate",
        "traversal_regret_fallback_argmax_tie_rate": "tie rate",
        "traversal_regret_fallback_argmax_tie_size_mean": "tie size",
        "traversal_regret_fallback_avg_depth": "avg depth",
        "traversal_regret_fallback_avg_opened_colors_before_action": "opened colors",
        "traversal_regret_fallback_count": "fallbacks",
        "traversal_regret_fallback_legal_actions_mean": "legal actions",
        "traversal_regret_fallback_open_new_available_rate": "available",
        "traversal_regret_fallback_open_new_selected_rate": "selected",
        "traversal_regret_fallback_open_new_selection_over_availability": "selected / available",
        "traversal_regret_fallback_rate": "fallback rate",
        "traversal_terminal_rate": "terminal",
        "traversal_terminals": "terminal",
    }
    return labels.get(metric, _label(metric))


def _short_metric_label(metric: str) -> str:
    labels = {
        "first_open_recoverable_score_mean_for_positive_final": "positive final",
        "first_open_recoverable_score_mean_for_negative_final": "negative final",
        "per_game_positive_expeditions": "positive",
        "per_game_negative_expeditions": "negative",
        "per_game_breakeven_expeditions": "breakeven",
        "per_game_below_minus_20_expeditions": "below -20",
    }
    return labels.get(metric, _label(metric))


def _base_metrics_for_derived_values() -> set[str]:
    return {
        "avg_opened_colors",
        "bad_open_actions",
        "bad_open_rate",
        "bonus_expedition_rate",
        "first_open_recoverable_score_mean_for_negative_final",
        "first_open_recoverable_score_mean_for_positive_final",
        "opening_play_actions",
        "per_game_bonus_expeditions",
        "weak_open_actions",
        "weak_open_rate",
    }


def _extra_plot_count(section: SectionSpec) -> int:
    return 1 if section.name == "Traversal" else 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Lost Cities Deep CFR metrics.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Moving-average window. Default: 1 (no smoothing).",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable moving-average smoothing.",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        help="Only plot metrics up to and including this iteration.",
    )
    args = parser.parse_args(argv)
    smoothing_window = 1 if args.no_smoothing else max(1, args.smoothing_window)
    written = analyze_run(
        args.run,
        args.output_dir,
        smoothing_window=smoothing_window,
        max_iteration=args.max_iteration,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
