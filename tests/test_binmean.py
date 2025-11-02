import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp, check_cupy


class TestBinMean(unittest.TestCase):
    if check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    def _compute_expected(self, xp, x, weights, n_bins):
        """Compute expected bin means using the reference definition."""
        n_datasets = x.shape[1]
        expected = xp.full((n_bins, n_datasets), xp.nan, dtype=weights.dtype)
        for dataset_idx in range(n_datasets):
            dataset_x = x[:, dataset_idx]
            dataset_w = weights[:, dataset_idx]
            for bin_idx in range(n_bins):
                mask = dataset_x == bin_idx
                count = int(mask.sum())
                if count:
                    expected[bin_idx, dataset_idx] = dataset_w[mask].mean()
        return expected

    def _assert_allclose(self, xp, result, expected):
        xp.testing.assert_allclose(result, expected)

    def test_binmean_computes_weighted_mean_for_multiple_datasets(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [0, 1, 2],
                    [1, 1, 0],
                    [1, 2, 1],
                    [2, 0, 2],
                    [2, 2, 2],
                ],
                dtype="int64",
            )
            weights = xp.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 2.0, 1.0],
                    [2.0, 3.0, 2.0],
                    [3.0, 4.0, 3.0],
                    [4.0, 5.0, 4.0],
                ],
                dtype="float64",
            )
            n_bins = 4

            expected = self._compute_expected(xp, x, weights, n_bins)
            result = magtrack.binmean(x.copy(), weights, n_bins)

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, (n_bins, x.shape[1]))

    def test_binmean_returns_nan_for_empty_bins(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [0, 3],
                    [0, 3],
                    [3, 0],
                    [3, 0],
                ],
                dtype="int64",
            )
            weights = xp.ones_like(x, dtype="float64")
            n_bins = 4

            result = magtrack.binmean(x.copy(), weights, n_bins)

            # Bins 1 and 2 have no contributions in either dataset
            for bin_idx in (1, 2):
                for dataset_idx in (0, 1):
                    self.assertTrue(bool(xp.isnan(result[bin_idx, dataset_idx])))

    def test_binmean_ignores_values_greater_than_or_equal_to_number_of_bins(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [0, 0],
                    [1, 3],
                    [3, 5],
                    [2, 2],
                ],
                dtype="int64",
            )
            weights = xp.array(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                ],
                dtype="float64",
            )
            n_bins = 3

            expected = self._compute_expected(xp, x, weights, n_bins)
            result = magtrack.binmean(x.copy(), weights, n_bins)

            self._assert_allclose(xp, result, expected)

    def test_binmean_preserves_weight_dtype(self):
        for xp in self.xp_modules:
            x = xp.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 0],
                ],
                dtype="int64",
            )
            weights = xp.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                ],
                dtype=xp.float32,
            )
            n_bins = 3

            result = magtrack.binmean(x.copy(), weights, n_bins)

            self.assertEqual(result.dtype, weights.dtype)

    def test_binmean_handles_single_dataset(self):
        for xp in self.xp_modules:
            x = xp.array([[0], [1], [1], [2], [4]], dtype="int64")
            weights = xp.array([[1.0], [2.0], [4.0], [8.0], [16.0]], dtype="float64")
            n_bins = 5

            expected = self._compute_expected(xp, x, weights, n_bins)
            result = magtrack.binmean(x.copy(), weights, n_bins)

            self.assertEqual(result.shape, (n_bins, 1))
            self._assert_allclose(xp, result, expected)


if __name__ == "__main__":
    unittest.main()
