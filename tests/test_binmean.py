import sys
import types
import unittest
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("tifffile", types.ModuleType("tifffile"))

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - cupy not installed
    cp = None  # type: ignore[misc]


def _install_cupyx_stub():
    """Provide a minimal stub for cupyx when CUDA libraries are unavailable."""
    cupyx_module = types.ModuleType("cupyx")
    scipy_module = types.ModuleType("cupyx.scipy")
    signal_module = types.ModuleType("cupyx.scipy.signal")

    def _get_array_module(x):
        if cp is not None:
            return cp.get_array_module(x)
        return np

    scipy_module.get_array_module = _get_array_module  # type: ignore[attr-defined]

    def _fftconvolve_stub(*args, **kwargs):  # pragma: no cover - safety net only
        raise RuntimeError("cupyx.scipy.signal.fftconvolve is unavailable in tests")

    signal_module.fftconvolve = _fftconvolve_stub  # type: ignore[attr-defined]

    scipy_module.signal = signal_module  # type: ignore[attr-defined]
    cupyx_module.scipy = scipy_module  # type: ignore[attr-defined]

    sys.modules.setdefault("cupyx", cupyx_module)
    sys.modules["cupyx.scipy"] = scipy_module
    sys.modules["cupyx.scipy.signal"] = signal_module


if cp is None:
    _install_cupyx_stub()
else:
    try:
        import cupyx.scipy.signal  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on runtime GPU libs
        if "cublas" in str(exc).lower():
            _install_cupyx_stub()
        else:
            raise

CP_AVAILABLE = False
if cp is not None:
    try:
        cp.zeros((1,))
    except Exception:  # pragma: no cover - depends on runtime GPU libs
        CP_AVAILABLE = False
    else:
        CP_AVAILABLE = True

import magtrack


class TestBinMean(unittest.TestCase):
    xp_modules = (np, cp) if CP_AVAILABLE and cp is not None else (np,)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        backend_names = ", ".join(xp.__name__ for xp in cls.xp_modules)
        warnings.warn(
            f"binmean tests exercising backends: {backend_names}",
            category=UserWarning,
            stacklevel=0,
        )

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
        if xp is np:
            np.testing.assert_allclose(result, expected)
        else:
            cp.testing.assert_allclose(result, expected)

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

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                result = magtrack.binmean(x.copy(), weights, n_bins)

            self.assertFalse(
                any(w.category is RuntimeWarning for w in caught),
                "binmean emitted a RuntimeWarning despite empty bins",
            )

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
