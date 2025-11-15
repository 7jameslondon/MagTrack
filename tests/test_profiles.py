import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp, check_cupy


class ProfileTestBase(unittest.TestCase):
    if check_cupy():
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

    def _assert_allclose(self, xp, result, expected, rtol=1e-7, atol=1e-12):
        xp.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

    def _compute_expected_radial_profile(self, xp, stack, x, y, oversample):
        width = stack.shape[0]
        n_images = stack.shape[2]
        n_bins = (width // 2) * oversample

        grid = xp.indices((width, width), dtype=xp.float32)
        r = xp.round(
            xp.hypot(
                grid[1][:, :, xp.newaxis] - x,
                grid[0][:, :, xp.newaxis] - y,
            )
            * oversample
        )
        r = r.astype(xp.uint16).reshape(-1, n_images)

        flat_stack = stack.reshape(width * width, n_images)
        return magtrack.binmean(r, flat_stack, n_bins)

    def _compute_expected_fft_profile(
        self, xp, stack, oversample, rmin, rmax
    ):
        n_images = stack.shape[2]
        width = stack.shape[0]
        center = width // 2
        n_bins = int(round(center * rmax * oversample))
        n_start = int(round(center * rmin * oversample))

        grid = xp.indices((width, center + 1), dtype=xp.float32)
        r_int = xp.round(
            xp.hypot(grid[1], grid[0] - center) * oversample
        ).astype(xp.uint16)
        r = xp.tile(r_int.reshape(-1, 1), (1, n_images))

        fft_cpx = xp.fft.fftshift(
            xp.fft.rfft2(stack, axes=(0, 1)), axes=(0,)
        )
        fft = xp.abs(fft_cpx).reshape(-1, n_images)

        profile = magtrack.binmean(r, fft, n_bins)
        return profile[n_start:]

    def _compute_expected_fft_profile_with_center(
        self, xp, stack, x, y, oversample, rmin, rmax, gaus_factor
    ):
        n_images = stack.shape[2]
        width = stack.shape[0]
        center = width // 2
        n_bins = int(round(center * rmax * oversample))
        n_start = int(round(center * rmin * oversample))

        grid = xp.indices((width, center + 1), dtype=xp.float32)
        r_int = xp.round(
            xp.hypot(grid[1], grid[0] - center) * oversample
        ).astype(xp.uint16)
        r = xp.tile(r_int.reshape(-1, 1), (1, n_images))

        coords = xp.arange(width, dtype=xp.float32)
        weights = magtrack.gaussian_2d(coords, coords, x, y, width / gaus_factor)
        weighted_stack = stack * weights

        fft_cpx = xp.fft.fftshift(
            xp.fft.rfft2(weighted_stack, axes=(0, 1)), axes=(0,)
        )
        fft = xp.abs(fft_cpx).reshape(-1, n_images)

        profile = magtrack.binmean(r, fft, n_bins)
        return profile[n_start:]


class TestRadialProfile(ProfileTestBase):
    def test_radial_profile_matches_manual_binning(self):
        image0 = np.arange(25, dtype=np.float32).reshape(5, 5)
        image1 = np.flipud(image0)
        stack_np = np.stack((image0, image1), axis=-1)

        x_np = np.array([2.0, 2.5], dtype=np.float32)
        y_np = np.array([2.0, 1.5], dtype=np.float32)
        oversample = 2

        for xp in self.xp_modules:
            stack = self._to_xp(xp, stack_np)
            x = self._to_xp(xp, x_np)
            y = self._to_xp(xp, y_np)

            result = magtrack.radial_profile(stack, x, y, oversample=oversample)
            expected = self._compute_expected_radial_profile(
                xp, stack, x, y, oversample
            )

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.dtype, stack.dtype)

    def test_radial_profile_supports_multiple_images(self):
        stack_np = np.ones((7, 7, 3), dtype=np.float32)
        stack_np[:, :, 1] *= 2.0
        stack_np[:, :, 2] *= np.linspace(1.0, 3.0, 7, dtype=np.float32)[:, None]

        x_np = np.array([3.0, 3.2, 4.0], dtype=np.float32)
        y_np = np.array([3.0, 3.5, 2.5], dtype=np.float32)
        oversample = 3

        for xp in self.xp_modules:
            stack = self._to_xp(xp, stack_np)
            x = self._to_xp(xp, x_np)
            y = self._to_xp(xp, y_np)

            result = magtrack.radial_profile(stack, x, y, oversample=oversample)
            expected = self._compute_expected_radial_profile(
                xp, stack, x, y, oversample
            )

            self._assert_allclose(xp, result, expected)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.shape[1], stack.shape[2])


class TestFFTProfile(ProfileTestBase):
    def test_fft_profile_matches_manual_computation(self):
        image = np.zeros((4, 4), dtype=np.float32)
        image[1, 1] = 4.0
        image[2, 1] = 1.0
        stack_np = np.stack((image, image.T), axis=-1)

        oversample = 4
        rmin = 0.0
        rmax = 0.75

        for xp in self.xp_modules:
            stack = self._to_xp(xp, stack_np)

            result = magtrack.fft_profile(
                stack.copy(), oversample=oversample, rmin=rmin, rmax=rmax
            )
            expected = self._compute_expected_fft_profile(
                xp, stack, oversample, rmin, rmax
            )

            self._assert_allclose(xp, result, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.shape[1], stack.shape[2])

    def test_fft_profile_respects_rmin_and_rmax(self):
        image = np.arange(36, dtype=np.float32).reshape(6, 6)
        stack_np = np.stack((image, image[::-1, :]), axis=-1)

        oversample = 2
        rmin = 0.25
        rmax = 0.5

        for xp in self.xp_modules:
            stack = self._to_xp(xp, stack_np)

            result = magtrack.fft_profile(
                stack.copy(), oversample=oversample, rmin=rmin, rmax=rmax
            )
            expected = self._compute_expected_fft_profile(
                xp, stack, oversample, rmin, rmax
            )

            self._assert_allclose(xp, result, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.shape[1], stack.shape[2])


class TestFFTProfileWithCenter(ProfileTestBase):
    def test_fft_profile_with_center_matches_manual_computation(self):
        image = np.zeros((4, 4), dtype=np.float32)
        image[1, 1] = 4.0
        image[2, 1] = 1.0
        stack_np = np.stack((image, image.T), axis=-1)

        x_np = np.array([1.5, 2.0], dtype=np.float32)
        y_np = np.array([1.5, 1.0], dtype=np.float32)
        oversample = 4
        rmin = 0.0
        rmax = 0.75
        gaus_factor = 5.0

        for xp in self.xp_modules:
            stack = self._to_xp(xp, stack_np)
            x = self._to_xp(xp, x_np)
            y = self._to_xp(xp, y_np)

            result = magtrack.fft_profile_with_center(
                stack.copy(),
                x,
                y,
                oversample=oversample,
                rmin=rmin,
                rmax=rmax,
                gaus_factor=gaus_factor,
            )
            expected = self._compute_expected_fft_profile_with_center(
                xp, stack, x, y, oversample, rmin, rmax, gaus_factor
            )

            self._assert_allclose(xp, result, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.shape[1], stack.shape[2])

    def test_fft_profile_with_center_respects_rmin_and_rmax(self):
        image = np.arange(36, dtype=np.float32).reshape(6, 6)
        stack_np = np.stack((image, image[::-1, :]), axis=-1)

        x_np = np.array([3.0, 2.5], dtype=np.float32)
        y_np = np.array([3.0, 3.5], dtype=np.float32)
        oversample = 2
        rmin = 0.25
        rmax = 0.5
        gaus_factor = 4.0

        for xp in self.xp_modules:
            stack = self._to_xp(xp, stack_np)
            x = self._to_xp(xp, x_np)
            y = self._to_xp(xp, y_np)

            result = magtrack.fft_profile_with_center(
                stack.copy(),
                x,
                y,
                oversample=oversample,
                rmin=rmin,
                rmax=rmax,
                gaus_factor=gaus_factor,
            )
            expected = self._compute_expected_fft_profile_with_center(
                xp, stack, x, y, oversample, rmin, rmax, gaus_factor
            )

            self._assert_allclose(xp, result, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.shape[1], stack.shape[2])


if __name__ == "__main__":
    unittest.main()
