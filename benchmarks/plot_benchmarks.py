"""Visualization helpers for benchmark logs produced by :mod:`run_all`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, cast

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import log_utils

__all__ = ["plot_benchmark_history"]


def _strip_benchmark(label: str) -> str:
    """Remove the substring "benchmark" (case-insensitive) from tick labels."""

    cleaned = re.sub("(?i)benchmark", "", label)
    # Collapse extra whitespace introduced by removal and trim the result.
    return " ".join(cleaned.split())


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

    backend_labels: dict[str, list[str]] = {}
    backend_per_system_values: dict[str, dict[str, list[float]]] = {}
    backend_latest_values: dict[str, list[float]] = {}
    latest_system_id: str | None = None
    if latest_run_id:
        latest_system_id = str(latest_run_id).split("/", 1)[0]

    for benchmark, backend in categories:
        if benchmark is None or backend is None:
            continue
        backend_key = str(backend)
        label = str(benchmark)

        backend_labels.setdefault(backend_key, [])
        backend_per_system_values.setdefault(
            backend_key, {system: [] for system in systems}
        )
        backend_latest_values.setdefault(backend_key, [])

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

        backend_labels[backend_key].append(label)
        for system in systems:
            system_values = [
                row.get("mean_time")
                for row in category_rows
                if row.get("system_id") == system and row.get("mean_time") is not None
            ]
            mean_value = _mean(system_values)
            backend_per_system_values[backend_key][system].append(
                _normalize(mean_value, baseline)
            )

        latest_times = [
            row.get("mean_time")
            for row in category_rows
            if row.get("run_id") == latest_run_id and row.get("mean_time") is not None
        ]
        backend_latest_values[backend_key].append(
            _normalize(_mean(latest_times), baseline)
        )

    if not any(backend_labels.values()):
        print("No benchmark entries were found to plot.")
        return None

    cmap = plt.get_cmap("tab20")

    ordered_backends: list[str] = []
    for candidate in ("cpu", "gpu"):
        if backend_labels.get(candidate):
            ordered_backends.append(candidate)
    remaining_backends = [
        backend
        for backend in backend_labels
        if backend not in ordered_backends and backend_labels[backend]
    ]
    ordered_backends.extend(sorted(remaining_backends))

    if not ordered_backends:
        print("No benchmark entries were found to plot.")
        return None

    max_label_count = max(len(backend_labels[backend]) for backend in ordered_backends)
    fig_height = 4 * len(ordered_backends)
    fig, axes = plt.subplots(
        nrows=len(ordered_backends),
        ncols=1,
        figsize=(max(8, max_label_count * 0.9), fig_height),
        sharex=True,
    )

    if not isinstance(axes, np.ndarray):
        axes_list = [axes]
    else:
        axes_list = list(axes)

    legend_entries: dict[str, object] = {}

    for axis, backend in zip(axes_list, ordered_backends):
        labels = backend_labels[backend]
        per_system_values = backend_per_system_values[backend]
        latest_values = backend_latest_values[backend]

        series: list[dict[str, object]] = []

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
                latest_values = latest_values + [
                    float("nan")
                ] * (len(labels) - len(latest_values))
            if any(not np.isnan(value) for value in latest_values):
                latest_label = (
                    f"{latest_system_id} (latest run)" if latest_system_id else "Latest run"
                )
                series.append(
                    {
                        "label": latest_label,
                        "values": latest_values,
                        "facecolor": "red",
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

        for index, config in enumerate(series):
            values = cast(Sequence[float], config["values"])
            axis.bar(
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

        axis.set_ylabel("Relative Runtime")
        title = f"{backend.upper()} Benchmark"
        axis.set_title(title)
        display_labels = [_strip_benchmark(str(label)) for label in labels]
        axis.set_xticks(x, display_labels, rotation=45, ha="right")
        axis.grid(axis="y", linestyle=":", alpha=0.4)
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)

        handles, labels = axis.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            if label and label not in legend_entries:
                legend_entries[label] = handle

    # Ensure the shared x-axis span accommodates the largest backend and hide
    # duplicated tick labels on all but the bottom subplot.
    if axes_list:
        axes_list[-1].set_xlim(-0.5, max_label_count - 0.5)
        for axis in axes_list[:-1]:
            axis.tick_params(labelbottom=False)

    fig.tight_layout(rect=(0, 0.12, 1, 1))

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(len(legend_entries), 4),
            title="System ID",
            fontsize="small",
            title_fontsize="medium",
            frameon=False,
        )

    return fig


if __name__ == "__main__":  # pragma: no cover - manual invocation
    figure = plot_benchmark_history()
    if figure is not None:
        plt.show()

