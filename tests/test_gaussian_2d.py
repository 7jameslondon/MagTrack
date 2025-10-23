import unittest

import cupy as cp
import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack


class TestGaussian2D(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    def _to_numpy(self, xp, value):
        if xp is cp:
            return cp.asnumpy(value)
        return np.asarray(value)

    def _to_xp(self, xp, value):
        if xp is cp:
            return cp.asarray(value)
        return np.asarray(value)

    def _compute_expected_numpy(self, x_np, y_np, mu_x_np, mu_y_np, sigma_np):
        x_arr = np.asarray(x_np, dtype=float)
        y_arr = np.asarray(y_np, dtype=float)
        mu_x_arr = np.atleast_1d(np.asarray(mu_x_np, dtype=float))
        mu_y_arr = np.atleast_1d(np.asarray(mu_y_np, dtype=float))
        sigma_arr = np.atleast_1d(np.asarray(sigma_np, dtype=float))

        denom = 2.0 * sigma_arr ** 2
        x_term = (x_arr[:, None, None] - mu_x_arr[None, None, :]) ** 2 / denom
        y_term = (y_arr[None, :, None] - mu_y_arr[None, None, :]) ** 2 / denom

        return np.exp(-(x_term + y_term))

    def _assert_allclose(self, xp, result, expected, rtol=1e-7, atol=1e-12, equal_nan=False):
        if equal_nan:
            a, b = xp.broadcast_arrays(result, expected)
            na, nb = xp.isnan(a), xp.isnan(b)

            xp.testing.assert_array_equal(na, nb)

            z_a = xp.zeros((), dtype=a.dtype)
            z_b = xp.zeros((), dtype=b.dtype)
            result = xp.where(na, z_a, a)
            expected = xp.where(nb, z_b, b)
        xp.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

    def test_gaussian_2d_matches_reference_values_for_grid_input(self):
        for xp in self.xp_modules:
            x = xp.linspace(-1.5, 1.5, 7, dtype=xp.float64)
            y = xp.linspace(-2.0, 2.0, 9, dtype=xp.float64)
            mu_x = xp.array([-0.5, 0.5], dtype=xp.float64)
            mu_y = xp.array([-1.0, 1.0], dtype=xp.float64)
            sigma = xp.float64(0.75)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, x),
                self._to_numpy(xp, y),
                self._to_numpy(xp, mu_x),
                self._to_numpy(xp, mu_y),
                self._to_numpy(xp, sigma),
            )
            expected = self._to_xp(xp, expected_numpy)

            result = magtrack.gaussian_2d(x, y, mu_x, mu_y, sigma)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, (x.shape[0], y.shape[0], mu_x.shape[0]))

    def test_gaussian_2d_supports_vector_sigma_per_center(self):
        for xp in self.xp_modules:
            x = xp.array([-1.0, 0.0, 1.0], dtype=xp.float64)
            y = xp.array([-2.0, 0.0, 2.0, 4.0], dtype=xp.float64)
            mu_x = xp.array([-1.0, 1.0, 0.0], dtype=xp.float64)
            mu_y = xp.array([0.0, 2.0, -2.0], dtype=xp.float64)
            sigma = xp.array([0.5, 1.0, 1.5], dtype=xp.float64)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, x),
                self._to_numpy(xp, y),
                self._to_numpy(xp, mu_x),
                self._to_numpy(xp, mu_y),
                self._to_numpy(xp, sigma),
            )
            expected = self._to_xp(xp, expected_numpy)

            result = magtrack.gaussian_2d(x, y, mu_x, mu_y, sigma)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, (x.shape[0], y.shape[0], mu_x.shape[0]))

    def test_gaussian_2d_handles_zero_sigma(self):
        for xp in self.xp_modules:
            x = xp.array([0.0, 1.0], dtype=xp.float64)
            y = xp.array([0.0, 1.0], dtype=xp.float64)
            mu_x = xp.array([0.0, 1.0], dtype=xp.float64)
            mu_y = xp.array([0.0, 1.0], dtype=xp.float64)
            sigma = xp.zeros(2, dtype=xp.float64)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, x),
                self._to_numpy(xp, y),
                self._to_numpy(xp, mu_x),
                self._to_numpy(xp, mu_y),
                self._to_numpy(xp, sigma),
            )
            expected = self._to_xp(xp, expected_numpy)

            result = magtrack.gaussian_2d(x, y, mu_x, mu_y, sigma)

            self._assert_allclose(xp, result, expected, equal_nan=True)
            self.assertTrue(bool(xp.any(xp.isnan(result))))

    def test_gaussian_2d_preserves_input_dtype(self):
        for xp in self.xp_modules:
            x = xp.array([-1.0, 0.0, 1.0], dtype=xp.float32)
            y = xp.array([-2.0, 0.0, 2.0], dtype=xp.float32)
            mu_x = xp.array([0.0], dtype=xp.float32)
            mu_y = xp.array([0.0], dtype=xp.float32)
            sigma = xp.array([1.0], dtype=xp.float32)

            result = magtrack.gaussian_2d(x, y, mu_x, mu_y, sigma)

            self.assertEqual(result.dtype, xp.float32)
            self.assertEqual(result.shape, (x.shape[0], y.shape[0], mu_x.shape[0]))

    def test_gaussian_2d_matches_single_center(self):
        for xp in self.xp_modules:
            x = xp.linspace(-1.0, 1.0, 5, dtype=xp.float64)
            y = xp.linspace(-1.0, 1.0, 5, dtype=xp.float64)
            mu_x = xp.array([0.25], dtype=xp.float64)
            mu_y = xp.array([-0.25], dtype=xp.float64)
            sigma = xp.array([0.5], dtype=xp.float64)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, x),
                self._to_numpy(xp, y),
                self._to_numpy(xp, mu_x),
                self._to_numpy(xp, mu_y),
                self._to_numpy(xp, sigma),
            )
            expected = self._to_xp(xp, expected_numpy)

            result = magtrack.gaussian_2d(x, y, mu_x, mu_y, sigma)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, (x.shape[0], y.shape[0], mu_x.shape[0]))


if __name__ == "__main__":
    unittest.main()
