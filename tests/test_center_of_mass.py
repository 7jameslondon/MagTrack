import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp
from magtrack.simulation import simulate_beads


class TestCenterOfMass(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    nm_per_px = 100.0
    size_px = 64

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.xyz_um = np.array(
            [
                [-0.02, 0.015, -0.05],
                [0.018, -0.022, 0.02],
                [0.005, 0.007, -0.01],
                [-0.012, -0.018, 0.04],
                [0.02, 0.012, -0.03],
                [-0.015, 0.022, 0.0],
                [0.017, -0.005, 0.01],
                [-0.018, 0.009, -0.02],
                [0.008, -0.01, 0.03],
                [-0.01, 0.0, -0.04],
            ],
            dtype=np.float64,
        )
        cls.stack_np = simulate_beads(
            cls.xyz_um,
            nm_per_px=cls.nm_per_px,
            size_px=cls.size_px,
        ).astype(np.float64)
        cls.expected_x_np, cls.expected_y_np = cls._compute_expected_centers()

    @classmethod
    def _compute_expected_centers(cls):
        base = (cls.size_px - 1) / 2.0
        scale = cls.nm_per_px * 1e-3
        expected_x = base + cls.xyz_um[:, 0] / scale
        expected_y = base + cls.xyz_um[:, 1] / scale
        return expected_x.astype(np.float64), expected_y.astype(np.float64)

    def _to_numpy(self, xp, value):
        if xp is cp:
            return cp.asnumpy(value)
        return np.asarray(value)

    def _to_xp(self, xp, value):
        if xp is cp:
            return cp.asarray(value)
        return np.asarray(value)

    def _compute_offsets(self, xp, centers_x, centers_y, expected_x, expected_y):
        dx = centers_x - expected_x
        dy = centers_y - expected_y
        return xp.sqrt(dx * dx + dy * dy)

    def _assert_within_tolerance(self, xp, offsets, tolerance):
        offsets_np = self._to_numpy(xp, offsets)
        max_offset = float(offsets_np.max())
        self.assertTrue(
            bool(np.all(offsets_np <= tolerance)),
            msg=f"Maximum offset {max_offset:.3f} exceeded tolerance {tolerance}",
        )

    def _run_center_of_mass_test(self, background, tolerance):
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np)
            expected_x = self._to_xp(xp, self.expected_x_np)
            expected_y = self._to_xp(xp, self.expected_y_np)

            centers_x, centers_y = magtrack.center_of_mass(stack, background=background)
            offsets = self._compute_offsets(xp, centers_x, centers_y, expected_x, expected_y)

            self._assert_within_tolerance(xp, offsets, tolerance)

    def test_center_of_mass_matches_simulated_centers_without_background(self):
        self._run_center_of_mass_test(background="none", tolerance=0.35)

    def test_center_of_mass_matches_simulated_centers_with_mean_background(self):
        self._run_center_of_mass_test(background="mean", tolerance=0.75)

    def test_center_of_mass_matches_simulated_centers_with_median_background(self):
        self._run_center_of_mass_test(background="median", tolerance=0.8)


if __name__ == "__main__":
    unittest.main()
