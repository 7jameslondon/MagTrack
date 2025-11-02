import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp
from magtrack.simulation import simulate_beads


class TestRadialProfile(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    nm_per_px = 100.0
    size_px = 64

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.xyz_nm = (
            np.array(
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
            * 1e3
        )
        cls.stack_np = simulate_beads(
            cls.xyz_nm,
            nm_per_px=cls.nm_per_px,
            size_px=cls.size_px,
        ).astype(np.float64)
        cls.expected_x_np, cls.expected_y_np = cls._compute_expected_centers()

    @classmethod
    def _compute_expected_centers(cls):
        base = cls.size_px / 2.0
        scale = cls.nm_per_px
        expected_x = base + cls.xyz_nm[:, 0] / scale
        expected_y = base + cls.xyz_nm[:, 1] / scale
        return expected_x.astype(np.float64), expected_y.astype(np.float64)

    def _manual_radial_profile(self, oversample):
        width = self.stack_np.shape[0]
        n_images = self.stack_np.shape[2]
        n_bins = (width // 2) * oversample
        grid = np.indices((width, width), dtype=np.float32)
        r = np.round(
            np.hypot(
                grid[1][:, :, None] - self.expected_x_np,
                grid[0][:, :, None] - self.expected_y_np,
            )
            * oversample
        ).astype(np.uint16)
        r = r.reshape(-1, n_images)
        flat_stack = self.stack_np.reshape(width * width, n_images)
        profiles = np.zeros((n_bins, n_images), dtype=np.float64)
        for idx in range(n_images):
            bins = np.minimum(r[:, idx], n_bins)
            values = flat_stack[:, idx]
            counts = np.bincount(bins, minlength=n_bins + 1)
            sums = np.bincount(bins, weights=values, minlength=n_bins + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                means = sums / counts
            profiles[:, idx] = means[:-1]
        return profiles

    def _to_numpy(self, xp, value):
        if xp is cp:
            return cp.asnumpy(value)
        return np.asarray(value)

    def _to_xp(self, xp, value):
        if xp is cp:
            return cp.asarray(value)
        return np.asarray(value)

    def test_radial_profile_matches_manual_computation(self):
        oversample = 1
        expected_profiles = self._manual_radial_profile(oversample)
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np)
            expected_x = self._to_xp(xp, self.expected_x_np)
            expected_y = self._to_xp(xp, self.expected_y_np)

            profiles = magtrack.radial_profile(
                stack,
                expected_x,
                expected_y,
                oversample=oversample,
            )
            profiles_np = self._to_numpy(xp, profiles)

            self.assertTrue(
                np.allclose(profiles_np, expected_profiles, atol=1e-6, equal_nan=True)
            )

    def test_radial_profile_respects_oversample_factor(self):
        oversample = 3
        expected_profiles = self._manual_radial_profile(oversample)
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np)
            expected_x = self._to_xp(xp, self.expected_x_np)
            expected_y = self._to_xp(xp, self.expected_y_np)

            profiles = magtrack.radial_profile(
                stack,
                expected_x,
                expected_y,
                oversample=oversample,
            )
            profiles_np = self._to_numpy(xp, profiles)

            self.assertEqual(profiles_np.shape[0], expected_profiles.shape[0])
            self.assertTrue(
                np.allclose(profiles_np, expected_profiles, atol=1e-6, equal_nan=True)
            )

    def test_radial_profile_does_not_modify_inputs(self):
        oversample = 2
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np.copy())
            expected_x = self._to_xp(xp, self.expected_x_np.copy())
            expected_y = self._to_xp(xp, self.expected_y_np.copy())

            stack_copy = stack.copy()
            expected_x_copy = expected_x.copy()
            expected_y_copy = expected_y.copy()

            magtrack.radial_profile(
                stack,
                expected_x,
                expected_y,
                oversample=oversample,
            )

            stack_diff = self._to_numpy(xp, stack - stack_copy)
            expected_x_diff = self._to_numpy(xp, expected_x - expected_x_copy)
            expected_y_diff = self._to_numpy(xp, expected_y - expected_y_copy)

            self.assertTrue(np.allclose(stack_diff, 0.0))
            self.assertTrue(np.allclose(expected_x_diff, 0.0))
            self.assertTrue(np.allclose(expected_y_diff, 0.0))


if __name__ == "__main__":
    unittest.main()
