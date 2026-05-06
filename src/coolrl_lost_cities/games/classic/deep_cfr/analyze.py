from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PLOT_GROUPS: dict[str, list[str]] = {
    "Action Distribution": [
        "play_action_rate",
        "discard_action_rate",
        "draw_deck_rate",
        "draw_pile_rate",
    ],
    "Game Flow": [
        "avg_opened_colors",
        "5_color_open_count",
        "avg_expedition_cards",
    ],
    "Open Quality": [
        "bad_open_rate",
        "weak_open_rate",
        "good_open_rate",
        "opening_recoverable_score_mean",
    ],
    "Expedition Outcomes": [
        "positive_expedition_rate",
        "negative_expedition_rate",
        "bonus_expedition_rate",
        "final_expedition_score_p25",
        "final_expedition_score_median",
        "final_expedition_score_p75",
        "final_expedition_score_p90",
    ],
    "Calibration": [
        "first_open_recoverable_score_mean_for_positive_final",
        "first_open_recoverable_score_mean_for_negative_final",
    ],
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
            for metric in _all_group_metrics():
                suffix = f"_{metric}"
                if rest.endswith(suffix):
                    names.add(rest[: -len(suffix)])
    return sorted(names)


def plot_group(
    rows: list[dict[str, Any]],
    *,
    opponent: str,
    title: str,
    metrics: list[str],
    output: Path,
) -> bool:
    import matplotlib.pyplot as plt

    x = [int(row["iteration"]) for row in rows if "iteration" in row]
    if not x:
        return False
    plotted = False
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in metrics:
        key = f"eval_{opponent}_{metric}"
        values = [row.get(key) for row in rows]
        if all(value is None for value in values):
            continue
        y = [float("nan") if value is None else float(value) for value in values]
        ax.plot(x, y, marker="o", linewidth=1.5, label=metric)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(f"{title} - {opponent}")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return True


def analyze_run(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metrics(metrics_path)
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for opponent in opponent_names(rows):
        for title, metrics in PLOT_GROUPS.items():
            filename = f"analysis_{_slug(opponent)}_{_slug(title)}.png"
            path = output_dir / filename
            if plot_group(rows, opponent=opponent, title=title, metrics=metrics, output=path):
                written.append(path)
    return written


def _all_group_metrics() -> set[str]:
    return {metric for metrics in PLOT_GROUPS.values() for metric in metrics}


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
