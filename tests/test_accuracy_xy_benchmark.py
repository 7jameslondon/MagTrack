"""Tests for the XY localization accuracy benchmark."""

from __future__ import annotations

import numpy as np

from benchmarks.accuracy.xy_accuracy import run_accuracy_sweep


EXPECTED_COLUMNS = {
    "method",
    "image_index",
    "x_true_px",
    "y_true_px",
    "x_est_px",
    "y_est_px",
    "dx_px",
    "dy_px",
    "dx_nm",
    "dy_nm",
    "radius_nm",
    "background_level",
    "contrast_scale",
    "z_true_nm",
}


def test_run_accuracy_sweep_smoke():
    df = run_accuracy_sweep(n_images=4, rng_seed=0)
    assert not df.empty
    assert EXPECTED_COLUMNS.issubset(df.columns)
    combos = 2 * 2 * 2  # radius_nm_choices * background_levels * contrast_scales
    assert df["image_index"].nunique() == 4 * combos


def test_default_methods_present():
    df = run_accuracy_sweep(n_images=3, rng_seed=1)
    methods = set(df["method"].unique())
    assert {"com_xy", "com_autoconv_xy"}.issubset(methods)


def test_reproducible_errors_with_seed():
    df1 = run_accuracy_sweep(n_images=3, rng_seed=123)
    df2 = run_accuracy_sweep(n_images=3, rng_seed=123)
    summary1 = df1.groupby("method")["dx_nm"].mean().sort_index().to_numpy()
    summary2 = df2.groupby("method")["dx_nm"].mean().sort_index().to_numpy()
    np.testing.assert_allclose(summary1, summary2)


def test_parameter_combinations_produce_expected_count():
    df = run_accuracy_sweep(
        n_images=1,
        nm_per_px=(80.0, 100.0),
        size_px=(32, 64),
        radius_nm_choices=(1000.0,),
        background_levels=(0.2, 0.4),
        contrast_scales=(0.5,),
        rng_seed=42,
    )
    expected_combos = 2 * 2 * 1 * 2 * 1
    assert df["image_index"].nunique() == expected_combos
    assert set(df["nm_per_px"].unique()) == {80.0, 100.0}
    assert set(df["size_px"].unique()) == {32, 64}
