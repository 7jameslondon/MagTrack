import unittest

import cupy as cp
import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack


class TestGaussian(unittest.TestCase):
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

    def _compute_expected_numpy(self, x_np, mu_np, sigma_np):
        return np.exp(-((x_np - mu_np) ** 2) / (2.0 * sigma_np ** 2))

    def _assert_allclose(self, xp, result, expected, rtol=1e-7, atol=1e-12):
        if xp is cp:
            cp.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
        else:
            np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

    def test_gaussian_matches_reference_values_for_array_input(self):
        for xp in self.xp_modules:
            x = xp.linspace(-2.0, 2.0, 9, dtype=xp.float64)
            mu = xp.float64(0.5)
            sigma = xp.float64(0.75)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, x),
                self._to_numpy(xp, mu),
                self._to_numpy(xp, sigma),
            )
            expected = self._to_xp(xp, expected_numpy)

            result = magtrack.gaussian(x, mu, sigma)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, x.shape)

    def test_gaussian_supports_array_mu_and_sigma(self):
        for xp in self.xp_modules:
            x = xp.array([0.0, 1.0, 2.0, 3.0], dtype=xp.float64)
            mu = xp.array([0.0, 0.5, 1.0, 1.5], dtype=xp.float64)
            sigma = xp.array([1.0, 0.8, 0.6, 0.4], dtype=xp.float64)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, x),
                self._to_numpy(xp, mu),
                self._to_numpy(xp, sigma),
            )
            expected = self._to_xp(xp, expected_numpy)

            result = magtrack.gaussian(x, mu, sigma)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, x.shape)

    def test_gaussian_preserves_nd_shape(self):
        for xp in self.xp_modules:
            x = xp.arange(12, dtype=xp.float64).reshape(3, 4)
            mu = xp.float64(5.0)
            sigma = xp.float64(1.5)

            result = magtrack.gaussian(x, mu, sigma)

            self.assertEqual(result.shape, x.shape)
            self.assertTrue(result.dtype == xp.float64)

    def test_gaussian_handles_zero_sigma(self):
        for xp in self.xp_modules:
            x = xp.array([0.0, 1.0, 2.0], dtype=xp.float64)
            mu = xp.zeros_like(x)
            sigma = xp.float64(0.0)

            result = magtrack.gaussian(x, mu, sigma)

            equal_mask = x == mu
            not_equal_mask = ~equal_mask

            self.assertTrue(bool(xp.all(xp.isnan(result[equal_mask]))))
            self.assertTrue(bool(xp.all(result[not_equal_mask] == xp.float64(0.0))))

    def test_gaussian_respects_input_dtype(self):
        for xp in self.xp_modules:
            x = xp.array([0.0, 1.0, 2.0], dtype=xp.float32)
            mu = xp.float32(0.0)
            sigma = xp.float32(1.0)

            result = magtrack.gaussian(x, mu, sigma)

            self.assertEqual(result.dtype, xp.float32)


if __name__ == "__main__":
    unittest.main()
