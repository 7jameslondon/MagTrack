from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.accuracy.plot_xy_accuracy import (
    compute_xy_metrics,
    make_default_plots,
    plot_metric_vs_factor,
)


def _synthetic_df() -> pd.DataFrame:
    data = []
    for method in ("method_a", "method_b"):
        for radius in (1000.0, 2000.0):
            data.append(
                {
                    "method": method,
                    "image_index": len(data),
                    "dx_nm": 1.0 if method == "method_a" else -0.5,
                    "dy_nm": -0.5 if radius == 1000.0 else 0.75,
                    "radius_nm": radius,
                    "background_level": 0.5,
                    "contrast_scale": 1.0,
                    "z_true_nm": 50.0 if method == "method_a" else -50.0,
                }
            )
    return pd.DataFrame(data)


def test_compute_xy_metrics_no_factor() -> None:
    df = _synthetic_df()
    metrics = compute_xy_metrics(df)
    assert set(metrics.columns) == {
        "method",
        "mean_abs_dx_nm",
        "mean_abs_dy_nm",
        "rmse_r_nm",
    }
    assert set(metrics["method"]) == {"method_a", "method_b"}
    assert np.all(
        np.isfinite(metrics[["mean_abs_dx_nm", "mean_abs_dy_nm", "rmse_r_nm"]].to_numpy())
    )


def test_compute_xy_metrics_with_factor() -> None:
    df = _synthetic_df()
    metrics = compute_xy_metrics(df, factor="radius_nm")
    assert set(metrics.columns) == {
        "method",
        "radius_nm",
        "mean_abs_dx_nm",
        "mean_abs_dy_nm",
        "rmse_r_nm",
    }
    assert len(metrics) == 4
    assert set(metrics["radius_nm"]) == {1000.0, 2000.0}


def test_plot_metric_vs_factor_returns_axes() -> None:
    df = _synthetic_df()
    ax = plot_metric_vs_factor(df, factor="radius_nm")
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_make_default_plots_creates_pngs(tmp_path: Path) -> None:
    df = _synthetic_df()
    make_default_plots(df, tmp_path)
    for factor in ("radius_nm", "z_true_nm", "contrast_scale"):
        path = tmp_path / f"xy_{factor}_rmse_r_nm.png"
        assert path.exists()
