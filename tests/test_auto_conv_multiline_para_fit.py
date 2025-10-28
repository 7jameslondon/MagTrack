import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp
from magtrack.simulation import simulate_beads


class TestAutoConvMultilineParaFit(unittest.TestCase):
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
        base = cls.size_px / 2.0
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

    def _initial_guesses(self, xp, expected_x, expected_y):
        offsets_x = np.linspace(-0.3, 0.3, expected_x.shape[0])
        offsets_y = np.linspace(0.25, -0.25, expected_y.shape[0])
        guess_x = expected_x + self._to_xp(xp, offsets_x)
        guess_y = expected_y + self._to_xp(xp, offsets_y)
        return guess_x, guess_y

    def _assert_within_tolerance(self, xp, offsets, tolerance):
        offsets_np = self._to_numpy(xp, offsets)
        max_offset = float(offsets_np.max())
        self.assertTrue(
            bool(np.all(offsets_np <= tolerance)),
            msg=f"Maximum offset {max_offset:.3f} exceeded tolerance {tolerance}",
        )

    def test_auto_conv_multiline_para_fit_reaches_subpixel_accuracy(self):
        tolerance = 0.1
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np)
            expected_x = self._to_xp(xp, self.expected_x_np)
            expected_y = self._to_xp(xp, self.expected_y_np)
            guess_x, guess_y = self._initial_guesses(xp, expected_x, expected_y)

            centers_x, centers_y = magtrack.auto_conv_multiline_para_fit(
                stack, guess_x, guess_y
            )
            offsets = self._compute_offsets(xp, centers_x, centers_y, expected_x, expected_y)

            self._assert_within_tolerance(xp, offsets, tolerance)

    def test_auto_conv_multiline_para_fit_improves_over_auto_conv_multiline(self):
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np)
            expected_x = self._to_xp(xp, self.expected_x_np)
            expected_y = self._to_xp(xp, self.expected_y_np)
            guess_x, guess_y = self._initial_guesses(xp, expected_x, expected_y)

            refined_x, refined_y = magtrack.auto_conv_multiline_para_fit(
                stack, guess_x, guess_y
            )
            coarse_x, coarse_y = magtrack.auto_conv_multiline(stack, guess_x, guess_y)

            refined_offsets = self._compute_offsets(
                xp, refined_x, refined_y, expected_x, expected_y
            )
            coarse_offsets = self._compute_offsets(
                xp, coarse_x, coarse_y, expected_x, expected_y
            )

            refined_offsets_np = self._to_numpy(xp, refined_offsets)
            coarse_offsets_np = self._to_numpy(xp, coarse_offsets)

            self.assertLess(
                float(refined_offsets_np.mean()),
                float(coarse_offsets_np.mean()),
            )
            self.assertTrue(
                bool(np.all(refined_offsets_np <= coarse_offsets_np + 1e-8)),
                msg="Refined offsets are not uniformly better than coarse offsets",
            )

    def test_auto_conv_multiline_para_fit_does_not_modify_inputs(self):
        for xp in self.xp_modules:
            stack = self._to_xp(xp, self.stack_np.copy())
            expected_x = self._to_xp(xp, self.expected_x_np)
            expected_y = self._to_xp(xp, self.expected_y_np)
            guess_x, guess_y = self._initial_guesses(xp, expected_x, expected_y)

            stack_copy = stack.copy()
            guess_x_copy = guess_x.copy()
            guess_y_copy = guess_y.copy()

            magtrack.auto_conv_multiline_para_fit(stack, guess_x, guess_y)

            stack_diff = self._to_numpy(xp, stack - stack_copy)
            guess_x_diff = self._to_numpy(xp, guess_x - guess_x_copy)
            guess_y_diff = self._to_numpy(xp, guess_y - guess_y_copy)

            self.assertTrue(np.allclose(stack_diff, 0.0))
            self.assertTrue(np.allclose(guess_x_diff, 0.0))
            self.assertTrue(np.allclose(guess_y_diff, 0.0))


if __name__ == "__main__":
    unittest.main()
