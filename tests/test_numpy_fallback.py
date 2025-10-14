import importlib
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("tifffile", types.ModuleType("tifffile"))


class TestNumPyFallback(unittest.TestCase):
    modules_to_reload = (
        "magtrack._xp",
        "magtrack.utils",
        "magtrack.core",
        "magtrack",
    )

    def setUp(self):
        super().setUp()
        self._orig_env = os.environ.get("MAGTRACK_FORCE_NUMPY")
        os.environ["MAGTRACK_FORCE_NUMPY"] = "1"

        for module_name in self.modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)

        self.magtrack = sys.modules["magtrack"]

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop("MAGTRACK_FORCE_NUMPY", None)
        else:
            os.environ["MAGTRACK_FORCE_NUMPY"] = self._orig_env

        for module_name in self.modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

        super().tearDown()

    def test_backend_forced_to_numpy(self):
        self.assertFalse(self.magtrack._xp.is_cupy_backend())

    def test_binmean_matches_numpy_reference(self):
        x = np.array(
            [
                [0, 1, 2],
                [1, 1, 0],
                [1, 2, 1],
                [2, 0, 2],
                [2, 2, 2],
            ],
            dtype="int64",
        )
        weights = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 1.0],
                [2.0, 3.0, 2.0],
                [3.0, 4.0, 3.0],
                [4.0, 5.0, 4.0],
            ],
            dtype="float64",
        )

        expected = self._reference_binmean(x, weights, n_bins=4)

        result = self.magtrack.binmean(x.copy(), weights, n_bins=4)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_split_gpu_apply_runs_on_numpy_backend(self):
        stack = np.arange(2 * 2 * 5, dtype="float64").reshape(2, 2, 5)
        scale = np.array(2.0, dtype="float64")

        def reducer(substack, gain):
            return substack.sum(axis=(0, 1)) * gain

        result = self.magtrack.utils.split_gpu_apply(
            stack,
            n=2,
            func=reducer,
            splitargs=[],
            fullargs=[scale],
        )

        expected_chunks = []
        for start in range(0, stack.shape[2], 2):
            chunk = stack[:, :, start : start + 2]
            expected_chunks.append(chunk.sum(axis=(0, 1)) * scale)
        expected = np.concatenate(expected_chunks)

        np.testing.assert_allclose(result, expected)
        self.assertIsInstance(result, np.ndarray)

    def test_stack_to_xyzp_returns_numpy_arrays(self):
        stack = np.zeros((5, 5, 3), dtype="float64")
        stack[2, 2, :] = 10.0
        stack[1, 2, :] = 5.0
        stack[2, 1, :] = 5.0

        x, y, z, profiles = self.magtrack.stack_to_xyzp(
            stack,
            zlut=None,
            **{"n auto_conv_multiline_para_fit": 0},
        )

        for array in (x, y, z, profiles):
            self.assertIsInstance(array, np.ndarray)

        self.assertEqual(x.shape, (3,))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(z.shape, (3,))
        self.assertEqual(profiles.shape[1], stack.shape[2])
        self.assertTrue(np.isnan(z).all())

    @staticmethod
    def _reference_binmean(x: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
        expected = np.full((n_bins, x.shape[1]), np.nan, dtype=weights.dtype)
        for dataset_idx in range(x.shape[1]):
            dataset_x = x[:, dataset_idx]
            dataset_w = weights[:, dataset_idx]
            for bin_idx in range(n_bins):
                mask = dataset_x == bin_idx
                if mask.any():
                    expected[bin_idx, dataset_idx] = dataset_w[mask].mean()
        return expected


if __name__ == "__main__":  # pragma: no cover - unittest main entry point
    unittest.main()
