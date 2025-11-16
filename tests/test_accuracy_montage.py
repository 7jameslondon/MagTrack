"""Tests for the accuracy sweep montage utility."""

import numpy as np
import pytest

from benchmarks.accuracy.xy_accuracy import create_accuracy_sweep_montage


def _montage_kwargs() -> dict:
    return {
        "n_images": 2,
        "magnification_choices": (100.0,),
        "size_nm_choices": (6400.0,),
        "radius_nm_choices": (1500.0,),
        "background_levels": (0.3,),
        "contrast_scales": (0.5,),
        "z_nm_choices": (0.0,),
        "x_fraction_choices": (0.0,),
        "y_fraction_choices": (0.0,),
        "photons_per_unit_choices": (5000.0,),
        "rng_seed": 0,
        "tile_columns": 2,
        "tile_spacing_px": 2,
    }


def test_create_accuracy_sweep_montage_dimensions():
    montage = create_accuracy_sweep_montage(**_montage_kwargs())
    assert montage.shape == (64, 130)
    assert montage.dtype == np.float32
    assert np.all(np.isfinite(montage))


def test_create_accuracy_sweep_montage_validates_columns():
    kwargs = _montage_kwargs()
    kwargs["tile_columns"] = 0
    with pytest.raises(ValueError):
        create_accuracy_sweep_montage(**kwargs)


def test_fill_value_populates_spacing_region():
    kwargs = _montage_kwargs()
    kwargs["tile_spacing_px"] = 3
    kwargs["fill_value"] = -1.0
    montage = create_accuracy_sweep_montage(**kwargs)
    spacing_region = montage[:, 64:67]
    assert spacing_region.shape == (64, 3)
    assert np.all(spacing_region == -1.0)
