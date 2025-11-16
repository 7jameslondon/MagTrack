"""Plotting utilities for XY localization accuracy sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.accuracy.xy_accuracy import AccuracySweepResults


SweepTable = AccuracySweepResults | Mapping[str, Sequence[Any]]


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _get_column(table: SweepTable, column: str) -> np.ndarray:
    if isinstance(table, AccuracySweepResults):
        if column not in table:
            raise KeyError(f"Column '{column}' not found in AccuracySweepResults")
        return table[column]
    if column not in table:
        raise KeyError(f"Column '{column}' not found in data")
    return np.asarray(table[column])


def _has_column(table: SweepTable, column: str) -> bool:
    if isinstance(table, AccuracySweepResults):
        return column in table
    if isinstance(table, Mapping):
        return column in table
    return False


def compute_xy_metrics(
    df: SweepTable,
    factor: str | None = None,
) -> list[dict[str, Any]]:
    """Compute summary XY-error metrics per method (optionally per factor)."""

    required = {"method", "dx_nm", "dy_nm"}
    missing = [col for col in required if not _has_column(df, col)]
    if missing:
        raise KeyError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )
    if factor is not None and not _has_column(df, factor):
        raise KeyError(f"Requested factor '{factor}' not found in sweep data")

    methods = _get_column(df, "method")
    dx_nm = _get_column(df, "dx_nm").astype(np.float64, copy=False)
    dy_nm = _get_column(df, "dy_nm").astype(np.float64, copy=False)
    if dx_nm.size == 0:
        return []

    factor_values = _get_column(df, factor) if factor is not None else None
    metrics: dict[tuple[Any, ...], dict[str, Any]] = {}

    for idx in range(dx_nm.size):
        method_val = str(_scalar(methods[idx]))
        key: tuple[Any, ...]
        factor_val = None
        if factor_values is not None:
            factor_val = _scalar(factor_values[idx])
            key = (method_val, factor_val)
        else:
            key = (method_val,)

        entry = metrics.setdefault(
            key,
            {
                "method": method_val,
                "factor": factor_val,
                "abs_dx_sum": 0.0,
                "abs_dy_sum": 0.0,
                "r_sq_sum": 0.0,
                "count": 0,
            },
        )
        entry["abs_dx_sum"] += float(abs(dx_nm[idx]))
        entry["abs_dy_sum"] += float(abs(dy_nm[idx]))
        entry["r_sq_sum"] += float(dx_nm[idx] ** 2 + dy_nm[idx] ** 2)
        entry["count"] += 1

    metrics_rows: list[dict[str, Any]] = []
    for key in sorted(metrics.keys()):
        entry = metrics[key]
        count = entry["count"]
        if count == 0:
            continue
        row = {
            "method": entry["method"],
            "mean_abs_dx_nm": entry["abs_dx_sum"] / count,
            "mean_abs_dy_nm": entry["abs_dy_sum"] / count,
            "rmse_r_nm": (entry["r_sq_sum"] / count) ** 0.5,
        }
        if factor is not None:
            row[factor] = entry["factor"]
        metrics_rows.append(row)

    return metrics_rows


def _pretty_label(name: str) -> str:
    name = name.replace("_", " ")
    if name.endswith(" nm"):
        return f"{name[:-3]} (nm)"
    return name


def plot_metric_vs_factor(
    df: SweepTable,
    factor: str,
    metric: str = "rmse_r_nm",
    methods: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a chosen metric vs a factor for each method."""

    metrics_rows = compute_xy_metrics(df, factor=factor)

    if metrics_rows and metric not in metrics_rows[0]:
        raise KeyError(f"Metric '{metric}' not computed by compute_xy_metrics")

    if methods is None:
        method_list = sorted({row["method"] for row in metrics_rows})
    else:
        available = {row["method"] for row in metrics_rows}
        method_list = [m for m in methods if m in available]

    if ax is None:
        _, ax = plt.subplots()

    for method in method_list:
        subset = [row for row in metrics_rows if row["method"] == method]
        subset.sort(key=lambda row: row.get(factor))
        if not subset:
            continue
        ax.plot(
            [row[factor] for row in subset],
            [row[metric] for row in subset],
            marker="o",
            linestyle="-",
            label=method,
        )

    ax.set_xlabel(_pretty_label(factor))
    ax.set_ylabel(_pretty_label(metric))
    if method_list:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def make_default_plots(
    df: SweepTable,
    out_dir: Path | None = None,
    *,
    show: bool = False,
) -> None:
    """Generate a set of standard XY-accuracy plots from sweep results.

    Parameters
    ----------
    df : Mapping or AccuracySweepResults
        Sweep results from :func:`run_accuracy_sweep`.
    out_dir : Path or None, optional
        Directory to store PNGs. If ``None``, plots are not saved.
    show : bool, optional
        Whether to display the generated plots via :func:`matplotlib.pyplot.show`.
    """

    figures: list[tuple[str, plt.Figure]] = []
    for factor in ("radius_nm", "z_true_nm", "contrast_scale"):
        if not _has_column(df, factor):
            continue
        fig, ax = plt.subplots()
        plot_metric_vs_factor(df, factor=factor, metric="rmse_r_nm", ax=ax)
        fig.tight_layout()
        figures.append((factor, fig))

    if out_dir is not None and figures:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for factor, fig in figures:
            fig.savefig(out_path / f"xy_{factor}_rmse_r_nm.png", dpi=150)

    if show and figures:
        plt.show()

    for _, fig in figures:
        plt.close(fig)


__all__ = [
    "compute_xy_metrics",
    "make_default_plots",
    "plot_metric_vs_factor",
]
