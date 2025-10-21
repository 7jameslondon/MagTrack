import unittest

import cupy as cp
import numpy as np

import conftests  # noqa: F401  # Ensure test package path setup
import magtrack


class TestPearson(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    def _to_numpy(self, xp, array):
        if xp is cp:
            return cp.asnumpy(array)
        return np.asarray(array)

    def _to_xp(self, xp, array):
        if xp is cp:
            return cp.asarray(array)
        return np.asarray(array)

    def _compute_expected_numpy(self, x_np, y_np):
        mean_x = np.nanmean(x_np, axis=0)
        mean_y = np.nanmean(y_np, axis=0)
        dif_x = x_np - mean_x
        dif_y = y_np - mean_y
        dif_x2 = dif_x ** 2
        dif_y2 = dif_y ** 2

        n_features = x_np.shape[1]
        n_targets = y_np.shape[1]
        expected = np.empty((n_targets, n_features), dtype=float)

        for target_idx in range(n_targets):
            numerator = np.nansum(dif_x * dif_y[:, target_idx:target_idx + 1], axis=0)
            denominator = np.sqrt(
                np.nansum(dif_x2, axis=0)
                * np.nansum(dif_y2[:, target_idx:target_idx + 1], axis=0)
            )
            expected[target_idx, :] = numerator / denominator

        return expected

    def _assert_allclose(self, xp, result, expected):
        xp.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-12)

    def test_pearson_matches_reference_values_for_multiple_series(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [1.0, 4.0, 7.0],
                    [2.0, 3.0, 6.0],
                    [3.0, 2.0, 5.0],
                    [4.0, 1.0, 4.0],
                    [5.0, 0.0, 3.0],
                ],
                dtype=xp.float64,
            )
            y = xp.array(
                [
                    [10.0, 0.0],
                    [20.0, 1.0],
                    [30.0, 2.0],
                    [40.0, 3.0],
                    [50.0, 4.0],
                ],
                dtype=xp.float64,
            )

            expected_numpy = self._compute_expected_numpy(self._to_numpy(xp, x), self._to_numpy(xp, y))
            expected = self._to_xp(xp, expected_numpy)
            result = magtrack.pearson(x, y)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, (y.shape[1], x.shape[1]))

    def test_pearson_handles_nan_values(self):
        for xp in self.xp_modules:
            nan = xp.nan
            x = xp.array(
                [
                    [1.0, nan, 3.0],
                    [nan, 2.0, 4.0],
                    [3.0, 3.0, nan],
                    [4.0, 4.0, 6.0],
                ],
                dtype=xp.float64,
            )
            y = xp.array(
                [
                    [1.0, nan],
                    [2.0, 2.0],
                    [nan, nan],
                    [4.0, 4.0],
                ],
                dtype=xp.float64,
            )

            expected_numpy = self._compute_expected_numpy(self._to_numpy(xp, x), self._to_numpy(xp, y))
            expected = self._to_xp(xp, expected_numpy)
            result = magtrack.pearson(x, y)

            self._assert_allclose(xp, result, expected)

    def test_pearson_returns_nan_when_any_series_has_zero_variance(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [5.0, 1.0],
                    [5.0, 2.0],
                    [5.0, 3.0],
                    [5.0, 4.0],
                ],
                dtype=xp.float64,
            )
            y = xp.array(
                [
                    [1.0, 7.0],
                    [1.0, 8.0],
                    [1.0, 9.0],
                    [1.0, 10.0],
                ],
                dtype=xp.float64,
            )

            result = magtrack.pearson(x, y)

            self.assertTrue(bool(xp.all(xp.isnan(result[:, 0]))))
            self.assertFalse(bool(xp.isnan(result[1, 1])))

    def test_pearson_output_dtype_is_float64(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                ],
                dtype=xp.float32,
            )
            y = xp.array(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                ],
                dtype=xp.float32,
            )

            result = magtrack.pearson(x, y)

            self.assertEqual(result.dtype, xp.float64)


if __name__ == "__main__":
    unittest.main()
