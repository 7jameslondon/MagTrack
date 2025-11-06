"""Visualization helpers for benchmark logs produced by :mod:`run_all`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import log_utils

__all__ = ["plot_benchmark_history"]


def _mean(values: Sequence[float | None]) -> float:
    finite = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


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

    latest_values = []
    historical_values = []
    labels = []
    for benchmark, backend in categories:
        if benchmark is None or backend is None:
            continue
        label = f"{benchmark} ({str(backend).upper()})"
        labels.append(label)
        latest = [row.get("mean_time") for row in rows if row.get("run_id") == latest_run_id and row.get("benchmark") == benchmark and row.get("backend") == backend]
        historical = [
            row.get("mean_time")
            for row in rows
            if row.get("run_id") != latest_run_id
            and row.get("benchmark") == benchmark
            and row.get("backend") == backend
        ]
        latest_values.append(_mean(latest))
        historical_values.append(_mean(historical))

    if not labels:
        print("No benchmark entries were found to plot.")
        return None

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    ax.bar(x - width / 2, historical_values, width, label="Historical mean")
    ax.bar(x + width / 2, latest_values, width, label="Latest run")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Benchmark runtime comparison")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    return fig


if __name__ == "__main__":  # pragma: no cover - manual invocation
    figure = plot_benchmark_history()
    if figure is not None:
        plt.show()

