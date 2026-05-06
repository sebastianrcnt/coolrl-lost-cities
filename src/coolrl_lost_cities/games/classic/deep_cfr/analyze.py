from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRAINING_PLOTS: list[tuple[str, list[str], str]] = [
    ("Advantage Loss", ["advantage_loss"], "loss"),
    ("Strategy Loss", ["strategy_loss"], "loss"),
    ("Iteration Time", ["iteration_seconds"], "seconds"),
    ("Throughput", ["nodes_per_second"], "nodes / second"),
    ("Samples", ["advantage_samples", "strategy_samples"], "samples"),
    ("Memory Size", ["advantage_memory_size", "strategy_memory_size"], "samples"),
    (
        "Traversal Depth",
        ["traversal_avg_endpoint_depth", "traversal_max_depth_reached"],
        "depth",
    ),
    (
        "Traversal Endpoints",
        [
            "traversal_terminals",
            "traversal_node_limit_cutoffs",
            "traversal_depth_cutoffs",
        ],
        "count",
    ),
]

EVAL_PLOTS: list[tuple[str, str, str, float]] = [
    ("Win Rate", "win_rate0", "rate (%)", 100.0),
    ("Avg Score Diff", "avg_score_diff0", "score diff", 1.0),
    ("Avg Score", "avg_score0", "score", 1.0),
    ("Policy Entropy", "policy_entropy", "entropy", 1.0),
    ("Play Action Rate", "play_action_rate", "rate (%)", 100.0),
    ("Discard Action Rate", "discard_action_rate", "rate (%)", 100.0),
    ("Draw Deck Rate", "draw_deck_rate", "rate (%)", 100.0),
    ("Draw Pile Rate", "draw_pile_rate", "rate (%)", 100.0),
    ("Opened Colors", "avg_opened_colors", "colors", 1.0),
    ("5-Color Open Count", "5_color_open_count", "games / eval", 1.0),
    ("Expedition Cards", "avg_expedition_cards", "cards", 1.0),
    ("Bad Open Rate", "bad_open_rate", "rate (%)", 100.0),
    ("Good Open Rate", "good_open_rate", "rate (%)", 100.0),
    ("Opening Recoverable Score", "opening_recoverable_score_mean", "score", 1.0),
    ("Score per Opened Color", "score_per_opened_color", "score / color", 1.0),
    ("Negative Expedition Rate", "negative_expedition_rate", "rate (%)", 100.0),
    ("Positive Expedition Rate", "positive_expedition_rate", "rate (%)", 100.0),
    (
        "Final Score per Expedition",
        "avg_final_score_per_opened_expedition",
        "score",
        1.0,
    ),
    ("Max Step Timeouts", "max_step_timeouts", "timeouts", 1.0),
]

SUMMARY_EVAL_METRICS: list[tuple[str, str, float]] = [
    ("win_rate0", "win rate (%)", 100.0),
    ("avg_score_diff0", "avg score diff", 1.0),
    ("avg_score0", "avg score", 1.0),
    ("play_action_rate", "play rate (%)", 100.0),
    ("avg_opened_colors", "opened colors", 1.0),
    ("bad_open_rate", "bad open (%)", 100.0),
    ("good_open_rate", "good open (%)", 100.0),
    ("score_per_opened_color", "score / opened color", 1.0),
]

OPPONENT_COLORS: dict[str, str] = {
    "noisy_safe": "tab:blue",
    "passive_discard": "tab:orange",
    "random": "tab:green",
    "safe_heuristic": "tab:red",
    "safe_heuristic_loose": "tab:purple",
    "safe_heuristic_strict": "tab:brown",
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


def plot_training_dashboard(rows: list[dict[str, Any]], output: Path) -> bool:
    import matplotlib.pyplot as plt

    x = [int(row["iteration"]) for row in rows if "iteration" in row]
    if not x:
        return False

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = list(axes.flat)
    plotted_any = False
    for ax, (title, metrics, ylabel) in zip(axes_flat, TRAINING_PLOTS, strict=False):
        plotted = _plot_row_metrics(ax, rows, x, metrics)
        _finish_axis(ax, title, ylabel=ylabel, plotted=plotted)
        plotted_any = plotted_any or plotted

    _plot_latest_depth_buckets(axes_flat[len(TRAINING_PLOTS)], rows)
    plotted_any = plotted_any or bool(_latest_depth_buckets(rows))

    fig.suptitle("Lost Cities Deep CFR training metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if not plotted_any:
        plt.close(fig)
        return False
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def plot_eval_dashboard(rows: list[dict[str, Any]], output: Path) -> bool:
    import matplotlib.pyplot as plt

    opponents = opponent_names(rows)
    if not opponents:
        return False

    fig, axes = plt.subplots(5, 4, figsize=(20, 18))
    axes_flat = list(axes.flat)
    plotted_any = False
    for ax, (title, metric, ylabel, scale) in zip(axes_flat, EVAL_PLOTS, strict=False):
        plotted = _plot_eval_metric(ax, rows, opponents, metric, scale)
        _finish_axis(ax, title, ylabel=ylabel, plotted=plotted)
        plotted_any = plotted_any or plotted

    for ax in axes_flat[len(EVAL_PLOTS) :]:
        ax.axis("off")

    handles, labels = _legend_items(axes_flat)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 6), fontsize="small")
    fig.suptitle("Lost Cities Deep CFR eval metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    plotted_any = False
    for ax, (metric, title, scale) in zip(axes.flat, SUMMARY_EVAL_METRICS, strict=False):
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for opponent in opponents:
            value = latest.get(f"eval_{opponent}_{metric}")
            if value is None:
                continue
            labels.append(opponent)
            values.append(float(value) * scale)
            colors.append(_opponent_color(opponent))
        if values:
            ax.bar(labels, values, color=colors)
            plotted_any = True
            ax.tick_params(axis="x", labelrotation=35, labelsize="x-small")
        _finish_axis(ax, title, xlabel="", plotted=bool(values))

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


def analyze_run(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metrics(metrics_path)
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    training_path = output_dir / "analysis_training_dashboard.png"
    if plot_training_dashboard(rows, training_path):
        written.append(training_path)

    eval_path = output_dir / "analysis_eval_dashboard.png"
    if _has_eval_history(rows) and plot_eval_dashboard(rows, eval_path):
        written.append(eval_path)

    final_eval_path = output_dir / "analysis_final_eval_summary.png"
    if plot_final_eval_summary(rows, final_eval_path):
        written.append(final_eval_path)
    return written


def _all_eval_metrics() -> set[str]:
    metrics = {metric for _, metric, _, _ in EVAL_PLOTS}
    metrics.update(metric for metric, _, _ in SUMMARY_EVAL_METRICS)
    return metrics


def _plot_row_metrics(
    ax: Any, rows: list[dict[str, Any]], x: list[int], metrics: list[str]
) -> bool:
    plotted = False
    for metric in metrics:
        values = [row.get(metric) for row in rows]
        if all(value is None for value in values):
            continue
        y = [float("nan") if value is None else float(value) for value in values]
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=_label(metric))
        plotted = True
    return plotted


def _plot_eval_metric(
    ax: Any,
    rows: list[dict[str, Any]],
    opponents: list[str],
    metric: str,
    scale: float,
) -> bool:
    plotted = False
    for opponent in opponents:
        pairs = [
            (int(row["iteration"]), float(row[f"eval_{opponent}_{metric}"]) * scale)
            for row in rows
            if "iteration" in row and f"eval_{opponent}_{metric}" in row
        ]
        if not pairs:
            continue
        x, y = zip(*pairs, strict=True)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=opponent,
            color=_opponent_color(opponent),
        )
        plotted = True
    return plotted


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


def _has_eval_history(rows: list[dict[str, Any]]) -> bool:
    eval_iterations = {
        int(row["iteration"])
        for row in rows
        if "iteration" in row and any(key.startswith("eval_") for key in row)
    }
    return len(eval_iterations) >= 2


def _finish_axis(
    ax: Any,
    title: str,
    *,
    xlabel: str = "iteration",
    ylabel: str | None = None,
    plotted: bool,
) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
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


def _label(metric: str) -> str:
    return metric.replace("_", " ")


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Lost Cities Deep CFR evaluation metrics.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    written = analyze_run(args.run, args.output_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
