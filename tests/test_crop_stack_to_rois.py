import unittest

import cupy as cp
import numpy as np

from tests import conftests  # noqa: F401  # Ensure test package path setup
import magtrack


class TestCropStackToRois(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    def _create_stack(self, xp):
        width = 6
        height = 6
        n_frames = 3
        return xp.arange(width * height * n_frames, dtype=xp.int64).reshape(width, height, n_frames)

    def test_crop_stack_to_rois_extracts_expected_regions(self):
        rois = np.array(
            [
                [1, 4, 0, 3],
                [2, 5, 2, 5],
                [0, 3, 3, 6],
            ],
            dtype=int,
        )

        for xp in self.xp_modules:
            stack = self._create_stack(xp)
            result = magtrack.crop_stack_to_rois(stack, rois)

            expected = xp.stack(
                [stack[left:right, top:bottom, :] for left, right, top, bottom in rois],
                axis=-1,
            )

            xp.testing.assert_array_equal(result, expected)
            self.assertEqual(result.shape, (3, 3, stack.shape[2], rois.shape[0]))

    def test_crop_stack_to_rois_preserves_dtype_and_backend(self):
        rois = np.array([[0, 4, 0, 4]], dtype=int)

        for xp in self.xp_modules:
            stack = xp.linspace(0.0, 1.0, 4 * 4 * 2, dtype=xp.float32).reshape(4, 4, 2)
            result = magtrack.crop_stack_to_rois(stack, rois)

            self.assertEqual(result.dtype, stack.dtype)
            self.assertEqual(type(result), type(stack))
            xp.testing.assert_array_equal(result[:, :, :, 0], stack)

    def test_crop_stack_to_rois_requires_square_rois(self):
        rois = np.array([[0, 3, 0, 4]], dtype=int)

        for xp in self.xp_modules:
            stack = xp.zeros((5, 5, 2), dtype=xp.float64)
            with self.assertRaises(ValueError):
                magtrack.crop_stack_to_rois(stack, rois)

    def test_crop_stack_to_rois_requires_consistent_roi_sizes(self):
        rois = np.array(
            [
                [0, 2, 0, 2],
                [1, 4, 1, 4],
            ],
            dtype=int,
        )

        for xp in self.xp_modules:
            stack = xp.zeros((5, 5, 2), dtype=xp.float64)
            with self.assertRaises(ValueError):
                magtrack.crop_stack_to_rois(stack, rois)


if __name__ == "__main__":
    unittest.main()
