from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np

from benchmarks.accuracy.plot_xy_accuracy import (
    compute_xy_metrics,
    make_default_plots,
    plot_metric_vs_factor,
)
from benchmarks.accuracy.runner import create_accuracy_montage
from benchmarks.accuracy.xy_accuracy import AccuracySweepResults


def _synthetic_results() -> AccuracySweepResults:
    rows = []
    for method in ("method_a", "method_b"):
        for radius in (1000.0, 2000.0):
            rows.append(
                {
                    "method": method,
                    "image_index": len(rows),
                    "dx_nm": 1.0 if method == "method_a" else -0.5,
                    "dy_nm": -0.5 if radius == 1000.0 else 0.75,
                    "radius_nm": radius,
                    "background_level": 0.5,
                    "contrast_scale": 1.0,
                    "z_true_nm": 50.0 if method == "method_a" else -50.0,
                }
            )
    return AccuracySweepResults.from_rows(rows)


def test_compute_xy_metrics_no_factor() -> None:
    results = _synthetic_results()
    metrics = compute_xy_metrics(results)
    assert {row["method"] for row in metrics} == {"method_a", "method_b"}
    for row in metrics:
        assert {"method", "mean_abs_dx_nm", "mean_abs_dy_nm", "rmse_r_nm"}.issubset(row)
        assert np.isfinite(row["mean_abs_dx_nm"])
        assert np.isfinite(row["mean_abs_dy_nm"])
        assert np.isfinite(row["rmse_r_nm"])


def test_compute_xy_metrics_with_factor() -> None:
    results = _synthetic_results()
    metrics = compute_xy_metrics(results, factor="radius_nm")
    assert len(metrics) == 4
    assert set(row["radius_nm"] for row in metrics) == {1000.0, 2000.0}
    for row in metrics:
        assert "rmse_r_nm" in row


def test_plot_metric_vs_factor_returns_axes() -> None:
    results = _synthetic_results()
    ax = plot_metric_vs_factor(results, factor="radius_nm")
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_make_default_plots_creates_pngs(tmp_path: Path) -> None:
    results = _synthetic_results()
    make_default_plots(results, tmp_path, show=False)
    for factor in ("radius_nm", "z_true_nm", "contrast_scale"):
        path = tmp_path / f"xy_{factor}_rmse_r_nm.png"
        assert path.exists()


def test_make_default_plots_show_only(tmp_path: Path) -> None:
    results = _synthetic_results()
    make_default_plots(results, out_dir=None, show=False)
    assert not any(tmp_path.iterdir())


def test_create_accuracy_montage_from_existing_plots(tmp_path: Path) -> None:
    results = _synthetic_results()
    make_default_plots(results, tmp_path, show=False)
    montage_path = create_accuracy_montage(
        plot_dir=tmp_path,
        out_path=tmp_path / "montage.png",
        columns=2,
        padding_px=0,
    )
    assert montage_path.exists()
    montage = mpimg.imread(montage_path)
    assert montage.ndim == 3
    assert montage.shape[0] > 0 and montage.shape[1] > 0
