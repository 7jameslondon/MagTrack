"""Plotting utilities for XY localization accuracy sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

try:
    matplotlib.use("tkAgg")
except Exception:  # pragma: no cover - fallback for headless environments
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def compute_xy_metrics(
    df: pd.DataFrame,
    factor: str | None = None,
) -> pd.DataFrame:
    """Compute summary XY-error metrics per method (optionally per factor)."""

    if "method" not in df.columns:
        raise KeyError("DataFrame must include a 'method' column")
    if any(col not in df.columns for col in ("dx_nm", "dy_nm")):
        raise KeyError("DataFrame must include 'dx_nm' and 'dy_nm' columns")
    if factor is not None and factor not in df.columns:
        raise KeyError(f"Requested factor '{factor}' not found in DataFrame")

    r_nm = (df["dx_nm"] ** 2 + df["dy_nm"] ** 2) ** 0.5
    working_df = df.assign(r_nm=r_nm)

    group_keys: list[str] = ["method"]
    if factor is not None:
        group_keys.append(factor)

    grouped = working_df.groupby(group_keys, dropna=False)
    metrics = grouped.agg(
        mean_abs_dx_nm=("dx_nm", lambda s: s.abs().mean()),
        mean_abs_dy_nm=("dy_nm", lambda s: s.abs().mean()),
        mean_r_sq=("r_nm", lambda s: (s ** 2).mean()),
    )
    metrics = metrics.reset_index()
    metrics["rmse_r_nm"] = metrics.pop("mean_r_sq") ** 0.5
    return metrics


def _pretty_label(name: str) -> str:
    name = name.replace("_", " ")
    if name.endswith(" nm"):
        return f"{name[:-3]} (nm)"
    return name


def plot_metric_vs_factor(
    df: pd.DataFrame,
    factor: str,
    metric: str = "rmse_r_nm",
    methods: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a chosen metric vs a factor for each method."""

    metrics_df = compute_xy_metrics(df, factor=factor)

    if metric not in metrics_df.columns:
        raise KeyError(f"Metric '{metric}' not computed by compute_xy_metrics")

    if methods is None:
        method_list = sorted(metrics_df["method"].unique())
    else:
        method_list = [m for m in methods if m in metrics_df["method"].unique()]

    if ax is None:
        _, ax = plt.subplots()

    for method in method_list:
        subset = metrics_df[metrics_df["method"] == method].sort_values(factor)
        if subset.empty:
            continue
        ax.plot(
            subset[factor],
            subset[metric],
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
    df: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    show: bool = False,
) -> None:
    """Generate a set of standard XY-accuracy plots from a sweep DataFrame.

    Parameters
    ----------
    df : DataFrame
        Sweep results from :func:`run_accuracy_sweep`.
    out_dir : Path or None, optional
        Directory to store PNGs. If ``None``, plots are not saved.
    show : bool, optional
        Whether to display the generated plots via :func:`matplotlib.pyplot.show`.
    """

    figures: list[tuple[str, plt.Figure]] = []
    for factor in ("radius_nm", "z_true_nm", "contrast_scale"):
        if factor not in df.columns:
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point to generate standard XY-accuracy plots from a CSV file."""

    parser = argparse.ArgumentParser(
        description="Generate XY-accuracy plots from a sweep CSV file."
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="Path to CSV produced by run_accuracy_sweep.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to store generated plots (default: show only).",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else None
    make_default_plots(df, out_dir=out_dir, show=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
