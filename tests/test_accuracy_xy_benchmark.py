"""Tests for the XY localization accuracy benchmark."""

from __future__ import annotations

import numpy as np

from benchmarks.accuracy.xy_accuracy import (
    DEFAULT_CAMERA_PIXEL_SIZE_NM,
    run_accuracy_sweep,
)


DEFAULT_Z_CHOICES = (0.0, 1000.0)
DEFAULT_X_FRACTIONS = (0.0, 0.01)
DEFAULT_Y_FRACTIONS = (0.0, 0.01)
DEFAULT_MAGNIFICATIONS = (100.0,)
DEFAULT_SIZE_NM = (6400.0,)


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
    "nm_per_px",
    "size_px",
    "size_nm",
    "magnification",
    "photons_per_unit",
    "x_fraction_of_size",
    "y_fraction_of_size",
}


def _run_default(**kwargs):
    kwargs.setdefault("x_fraction_choices", DEFAULT_X_FRACTIONS)
    kwargs.setdefault("y_fraction_choices", DEFAULT_Y_FRACTIONS)
    kwargs.setdefault("z_nm_choices", DEFAULT_Z_CHOICES)
    kwargs.setdefault("magnification_choices", DEFAULT_MAGNIFICATIONS)
    kwargs.setdefault("size_nm_choices", DEFAULT_SIZE_NM)
    return run_accuracy_sweep(**kwargs)


def test_run_accuracy_sweep_smoke():
    results = _run_default(n_images=4, rng_seed=0)
    assert len(results) > 0
    assert EXPECTED_COLUMNS.issubset(set(results.columns))
    combos = (
        2  # radius_nm_choices
        * 2  # background_levels
        * 2  # contrast_scales
        * len(DEFAULT_Z_CHOICES)
        * len(DEFAULT_X_FRACTIONS)
        * len(DEFAULT_Y_FRACTIONS)
    )
    assert np.unique(results["image_index"]).size == 4 * combos


def test_default_methods_present():
    results = _run_default(n_images=3, rng_seed=1)
    methods = {str(m) for m in np.unique(results["method"])}
    assert {"com_xy", "com_autoconv_xy"}.issubset(methods)


def test_reproducible_errors_with_seed():
    res1 = _run_default(n_images=3, rng_seed=123)
    res2 = _run_default(n_images=3, rng_seed=123)

    def _mean_dx(results):
        means = {}
        methods = np.unique(results["method"])
        for method in methods:
            mask = results["method"] == method
            means[str(method)] = float(np.mean(results["dx_nm"][mask]))
        return means

    summary1 = _mean_dx(res1)
    summary2 = _mean_dx(res2)
    ordered = sorted(summary1)
    np.testing.assert_allclose(
        [summary1[key] for key in ordered],
        [summary2[key] for key in ordered],
    )


def test_parameter_combinations_produce_expected_count():
    results = _run_default(
        n_images=1,
        magnification_choices=(100.0, 125.0),
        size_nm_choices=(3200.0, 6400.0),
        radius_nm_choices=(1000.0,),
        background_levels=(0.2, 0.4),
        contrast_scales=(0.5,),
        photons_per_unit_choices=(2000.0, 5000.0),
        rng_seed=42,
        x_fraction_choices=(-0.2, 0.0),
        y_fraction_choices=(0.0, 0.3),
    )
    expected_combos = (
        2  # magnification_choices
        * 2  # size_nm_choices
        * 1  # radius_nm_choices
        * 2  # background_levels
        * 1  # contrast_scales
        * 2  # photons_per_unit_choices
        * len(DEFAULT_Z_CHOICES)
        * 2  # x_fraction_choices
        * 2  # y_fraction_choices
    )
    assert np.unique(results["image_index"]).size == expected_combos
    assert set(np.unique(results["magnification"])) == {100.0, 125.0}
    assert set(np.unique(results["nm_per_px"])) == {
        DEFAULT_CAMERA_PIXEL_SIZE_NM / 100.0,
        DEFAULT_CAMERA_PIXEL_SIZE_NM / 125.0,
    }
    assert set(np.unique(results["size_nm"])) == {3200.0, 6400.0}
    assert set(np.unique(results["size_px"])) == {32, 40, 64, 80}
    assert set(np.unique(results["photons_per_unit"])) == {2000.0, 5000.0}
    assert set(np.unique(results["x_fraction_of_size"])) == {-0.2, 0.0}
    assert set(np.unique(results["y_fraction_of_size"])) == {0.0, 0.3}


def test_z_choices_expand_combinations():
    z_values = (-400.0, 0.0, 250.0)
    results = _run_default(
        n_images=2,
        magnification_choices=(111.11111111111111,),
        size_nm_choices=(4320.0,),
        radius_nm_choices=(1200.0,),
        background_levels=(0.25,),
        contrast_scales=(0.75,),
        z_nm_choices=z_values,
        rng_seed=7,
    )
    expected_combos = (
        1
        * 1
        * 1
        * 1
        * 1
        * len(z_values)
        * len(DEFAULT_X_FRACTIONS)
        * len(DEFAULT_Y_FRACTIONS)
    )
    assert np.unique(results["image_index"]).size == 2 * expected_combos
    assert set(np.unique(results["z_true_nm"])) == set(z_values)


def test_single_photon_choice_matches_default():
    results_default = _run_default(n_images=2, rng_seed=0)
    results_custom = _run_default(
        n_images=2,
        photons_per_unit_choices=(5000.0,),
        rng_seed=0,
    )
    np.testing.assert_allclose(
        results_default["photons_per_unit"],
        results_custom["photons_per_unit"],
    )


def test_fraction_coordinates_align_with_expected_pixels():
    results = _run_default(
        n_images=1,
        magnification_choices=(100.0,),
        size_nm_choices=(2000.0,),
        x_fraction_choices=(0.1,),
        y_fraction_choices=(-0.1,),
    )
    center = 20 // 2
    expected_x = center + 0.1 * 20
    expected_y = center - 0.1 * 20
    np.testing.assert_allclose(results["x_true_px"], expected_x)
    np.testing.assert_allclose(results["y_true_px"], expected_y)
