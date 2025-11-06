"""Visualization helpers for benchmark logs produced by :mod:`run_all`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import log_utils

__all__ = ["plot_benchmark_history"]


def _mean(values: Sequence[float | None]) -> float:
    finite = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _normalize(value: float, baseline: float) -> float:
    if np.isnan(value) or np.isnan(baseline) or baseline == 0:
        return float("nan")
    return float(value / baseline)


def _resolve_run_id(log_root: Path, run_directory: Path | None, rows: Sequence[dict[str, object]]) -> str | None:
    if run_directory is not None:
        try:
            rel = run_directory.resolve().relative_to(log_root.resolve())
            parts = rel.parts
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return "/".join(parts)
        except Exception:  # noqa: BLE001
            pass
    if not rows:
        return None
    # Fall back to the most recent timestamp observed in the aggregated rows.
    ordered = sorted(rows, key=lambda r: (r.get("timestamp"), r.get("run_id")))
    return ordered[-1].get("run_id") if ordered else None


def plot_benchmark_history(
    log_root: Path | str = log_utils.LOG_ROOT,
    run_directory: Path | None = None,
):
    """Render a bar chart comparing the latest benchmark run to historical data."""

    root = Path(log_root)
    rows = log_utils.aggregate_logs(root)
    if not rows:
        print("No benchmark logs found; skipping plot generation.")
        return None

    latest_run_id = _resolve_run_id(root, run_directory, rows)
    if latest_run_id is None:
        print("Unable to determine the latest run identifier; skipping plot generation.")
        return None

    categories = sorted(
        {
            (row.get("benchmark"), row.get("backend"))
            for row in rows
            if row.get("benchmark") and row.get("backend")
        }
    )

    systems = sorted({row.get("system_id") for row in rows if row.get("system_id")})
    if not systems:
        print("No system identifiers found in benchmark logs; skipping plot generation.")
        return None

    labels: list[str] = []
    per_system_values: dict[str, list[float]] = {system: [] for system in systems}
    latest_values: list[float] = []
    latest_system_id: str | None = None
    if latest_run_id:
        latest_system_id = str(latest_run_id).split("/", 1)[0]

    for benchmark, backend in categories:
        if benchmark is None or backend is None:
            continue
        label = f"{benchmark} ({str(backend).upper()})"

        category_rows = [
            row
            for row in rows
            if row.get("benchmark") == benchmark and row.get("backend") == backend
        ]
        combined = [row.get("mean_time") for row in category_rows if row.get("mean_time") is not None]
        if not combined:
            continue

        baseline = _mean(combined)
        if np.isnan(baseline) or baseline == 0:
            continue

        labels.append(label)
        for system in systems:
            system_values = [
                row.get("mean_time")
                for row in category_rows
                if row.get("system_id") == system and row.get("mean_time") is not None
            ]
            mean_value = _mean(system_values)
            per_system_values[system].append(_normalize(mean_value, baseline))

        latest_times = [
            row.get("mean_time")
            for row in category_rows
            if row.get("run_id") == latest_run_id and row.get("mean_time") is not None
        ]
        latest_values.append(_normalize(_mean(latest_times), baseline))

    if not labels:
        print("No benchmark entries were found to plot.")
        return None

    series: list[dict[str, object]] = []
    cmap = plt.get_cmap("tab20")

    for index, system in enumerate(systems):
        values = per_system_values.get(system, [])
        if len(values) < len(labels):
            values = values + [float("nan")] * (len(labels) - len(values))

        series.append(
            {
                "label": system,
                "values": values,
                "facecolor": cmap(index % cmap.N),
                "edgecolor": "black" if system == latest_system_id else None,
                "linewidth": 1.5 if system == latest_system_id else 0,
                "linestyle": "-",
                "zorder": 2 if system == latest_system_id else 1,
            }
        )

    latest_label: str | None = None
    if latest_run_id:
        if len(latest_values) < len(labels):
            latest_values = latest_values + [float("nan")] * (len(labels) - len(latest_values))
        if any(not np.isnan(value) for value in latest_values):
            latest_label = (
                f"{latest_system_id} (latest run)" if latest_system_id else "Latest run"
            )
            series.append(
                {
                    "label": latest_label,
                    "values": latest_values,
                    "facecolor": "none",
                    "edgecolor": "black",
                    "linewidth": 2.0,
                    "linestyle": "--",
                    "zorder": 3,
                }
            )

    x = np.arange(len(labels))
    bar_count = len(series)
    width = 0.8 / max(bar_count, 1)
    offsets = (
        np.arange(bar_count) * width - (width * (bar_count - 1) / 2)
        if bar_count > 1
        else np.array([0.0])
    )

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 6))

    for index, config in enumerate(series):
        values = cast(Sequence[float], config["values"])
        ax.bar(
            x + offsets[index],
            values,
            width,
            label=config["label"],
            facecolor=config["facecolor"],
            edgecolor=config["edgecolor"],
            linewidth=config["linewidth"],
            linestyle=config["linestyle"],
            zorder=config["zorder"],
        )

    ax.set_ylabel("Relative runtime (× average)")
    title = "Benchmark runtime comparison by system (normalised)"
    if latest_label:
        title += "\n(latest run outlined with dashed bars)"
    ax.set_title(title)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.legend(title="System ID", fontsize="small", title_fontsize="medium")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    fig.tight_layout()

    return fig


if __name__ == "__main__":  # pragma: no cover - manual invocation
    figure = plot_benchmark_history()
    if figure is not None:
        plt.show()

