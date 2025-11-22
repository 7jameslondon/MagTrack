import numpy as np

from magtrack.core import radial_profile
from magtrack.simulation import simulate_beads, simulate_zlut


def test_simulate_zlut_matches_manual_profile():
    z_nm = np.array([-200.0, 0.0, 200.0], dtype=float)
    size_px = 32
    nm_per_px = 100.0

    xyz_nm = np.column_stack([
        np.zeros_like(z_nm),
        np.zeros_like(z_nm),
        z_nm,
    ])
    stack = simulate_beads(xyz_nm, nm_per_px=nm_per_px, size_px=size_px)

    center = np.full(z_nm.shape, size_px / 2.0, dtype=float)
    expected_profiles = radial_profile(stack, center, center)
    expected_zlut = np.vstack([z_nm, expected_profiles])

    zlut = simulate_zlut(z_nm, nm_per_px=nm_per_px, size_px=size_px)

    np.testing.assert_allclose(zlut, expected_zlut)


def test_simulate_zlut_respects_oversample():
    z_nm = np.array([0.0, 300.0], dtype=float)
    size_px = 20
    oversample = 2

    zlut = simulate_zlut(z_nm, size_px=size_px, oversample=oversample)

    expected_bins = (size_px // 2) * oversample
    assert zlut.shape == (1 + expected_bins, z_nm.size)
    np.testing.assert_array_equal(zlut[0, :], z_nm)
