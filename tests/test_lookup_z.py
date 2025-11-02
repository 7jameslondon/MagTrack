import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp
from magtrack.simulation import simulate_beads


class TestLookupZ(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    nm_per_px = 100.0
    roi_px = 64
    n_local = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.z_reference_nm = np.arange(-10000.0, 10001.0, 200.0, dtype=np.float64)
        cls._create_reference_zlut()
        cls._create_evaluation_profiles()

    @classmethod
    def _create_reference_zlut(cls):
        xyz_reference_nm = np.column_stack(
            [
                np.zeros_like(cls.z_reference_nm),
                np.zeros_like(cls.z_reference_nm),
                cls.z_reference_nm,
            ]
        )
        reference_stack = simulate_beads(
            xyz_reference_nm,
            size_px=cls.roi_px,
            nm_per_px=cls.nm_per_px,
        ).astype(np.float64)
        center = cls.roi_px / 2.0
        centers = np.full(cls.z_reference_nm.shape, center, dtype=np.float64)
        reference_profiles = magtrack.radial_profile(reference_stack, centers, centers)
        cls.zlut_np = np.vstack([cls.z_reference_nm, reference_profiles.astype(np.float64)])

    @classmethod
    def _create_evaluation_profiles(cls):
        frames = 60
        cls.z_true_nm = np.linspace(-3000.0, 3000.0, frames, dtype=np.float64)
        cls.z_true_nm += 800.0 * np.sin(np.linspace(0.0, 6.0 * np.pi, frames, dtype=np.float64))
        xyz_eval_nm = np.column_stack(
            [
                np.zeros_like(cls.z_true_nm),
                np.zeros_like(cls.z_true_nm),
                cls.z_true_nm,
            ]
        )
        eval_stack = simulate_beads(
            xyz_eval_nm,
            size_px=cls.roi_px,
            nm_per_px=cls.nm_per_px,
        ).astype(np.float64)
        center = cls.roi_px / 2.0
        centers = np.full(cls.z_true_nm.shape, center, dtype=np.float64)
        cls.eval_profiles_np = magtrack.radial_profile(eval_stack, centers, centers).astype(
            np.float64
        )

    def _to_numpy(self, xp, value):
        if xp is cp:
            return cp.asnumpy(value)
        return np.asarray(value)

    def _to_xp(self, xp, value):
        if xp is cp:
            return cp.asarray(value)
        return np.asarray(value)

    def _assert_within_tolerance(self, xp, values, tolerance):
        values_np = self._to_numpy(xp, values)
        max_value = float(np.max(np.abs(values_np)))
        self.assertTrue(
            bool(np.all(np.abs(values_np) <= tolerance)),
            msg=f"Maximum deviation {max_value:.3f} exceeded tolerance {tolerance}",
        )

    def test_lookup_z_recovers_true_z_positions(self):
        tolerance = 150.0  # nanometers
        for xp in self.xp_modules:
            profiles = self._to_xp(xp, self.eval_profiles_np)
            zlut = self._to_xp(xp, self.zlut_np)
            expected_z = self._to_xp(xp, self.z_true_nm)

            z_fit = magtrack.lookup_z(profiles, zlut, n_local=self.n_local)

            self.assertEqual(z_fit.shape, expected_z.shape)
            errors = z_fit - expected_z
            self._assert_within_tolerance(xp, errors, tolerance)

    def test_lookup_z_does_not_modify_inputs(self):
        for xp in self.xp_modules:
            profiles = self._to_xp(xp, self.eval_profiles_np.copy())
            zlut = self._to_xp(xp, self.zlut_np.copy())

            profiles_copy = profiles.copy()
            zlut_copy = zlut.copy()

            magtrack.lookup_z(profiles, zlut, n_local=self.n_local)

            profiles_diff = self._to_numpy(xp, profiles - profiles_copy)
            zlut_diff = self._to_numpy(xp, zlut - zlut_copy)

            self.assertTrue(np.allclose(profiles_diff, 0.0))
            self.assertTrue(np.allclose(zlut_diff, 0.0))


if __name__ == "__main__":
    unittest.main()
