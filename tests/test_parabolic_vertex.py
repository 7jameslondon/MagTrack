import unittest

import cupy as cp
import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack


class TestParabolicVertex(unittest.TestCase):
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

    def _assert_allclose(self, xp, result, expected, rtol=1e-7, atol=1e-12):
        if xp is cp:
            cp.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
        else:
            np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

    def _compute_expected_numpy(self, data_np, vertex_est_np, n_local, weighted):
        n_local_half = n_local // 2
        vertex_int = np.rint(vertex_est_np).astype(np.int64)

        index_min = n_local_half
        index_max = data_np.shape[1] - n_local_half - 1
        vertex_int = np.clip(vertex_int, index_min, index_max)

        n_datasets = data_np.shape[0]
        rel_idx = np.arange(-n_local_half, n_local_half + 1, dtype=np.int64)
        idx = rel_idx + vertex_int[:, np.newaxis]
        y = data_np[np.arange(n_datasets)[:, np.newaxis], idx].T
        x = np.arange(n_local, dtype=np.float64)

        if weighted:
            w = n_local_half - np.abs(np.arange(n_local) - n_local_half) + 1
            p = np.polyfit(x, y, 2, w=w)
        else:
            p = np.polyfit(x, y, 2)

        vertex = -p[1, :] / (2.0 * p[0, :]) + vertex_int - n_local // 2
        vertex[vertex_int == index_min] = np.nan
        vertex[vertex_int == index_max] = np.nan

        return vertex

    def test_parabolic_vertex_matches_reference_values_weighted(self):
        true_vertices = [10.4, 23.25, 41.6]
        curvatures = [0.8, 1.2, 0.5]
        offsets = [2.0, -1.5, 0.75]
        vertex_offsets = [0.3, -0.4, 0.6]

        n_datapoints = 64
        n_local = 7

        for xp in self.xp_modules:
            coords = xp.arange(n_datapoints, dtype=xp.float64)
            rows = []
            for vertex, curvature, offset in zip(true_vertices, curvatures, offsets):
                vertex_val = xp.float64(vertex)
                curvature_val = xp.float64(curvature)
                offset_val = xp.float64(offset)
                rows.append(curvature_val * (coords - vertex_val) ** 2 + offset_val)
            data = xp.stack(rows, axis=0)

            vertex_est = xp.array(
                [vertex + delta for vertex, delta in zip(true_vertices, vertex_offsets)],
                dtype=xp.float64,
            )

            result = magtrack.parabolic_vertex(data, vertex_est, n_local, weighted=True)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, data),
                self._to_numpy(xp, vertex_est),
                n_local,
                weighted=True,
            )
            expected = self._to_xp(xp, expected_numpy)

            self._assert_allclose(xp, result, expected)
            self._assert_allclose(xp, result, self._to_xp(xp, np.array(true_vertices, dtype=float)), rtol=1e-5)
            self.assertEqual(result.shape, vertex_est.shape)

    def test_parabolic_vertex_matches_reference_values_unweighted(self):
        true_vertices = [12.1, 28.4]
        curvatures = [1.0, 0.6]
        offsets = [0.5, -2.0]
        vertex_offsets = [-0.45, 0.25]

        n_datapoints = 72
        n_local = 9

        for xp in self.xp_modules:
            coords = xp.arange(n_datapoints, dtype=xp.float64)
            rows = []
            for vertex, curvature, offset in zip(true_vertices, curvatures, offsets):
                vertex_val = xp.float64(vertex)
                curvature_val = xp.float64(curvature)
                offset_val = xp.float64(offset)
                rows.append(curvature_val * (coords - vertex_val) ** 2 + offset_val)
            data = xp.stack(rows, axis=0)

            vertex_est = xp.array(
                [vertex + delta for vertex, delta in zip(true_vertices, vertex_offsets)],
                dtype=xp.float64,
            )

            result = magtrack.parabolic_vertex(data, vertex_est, n_local, weighted=False)

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, data),
                self._to_numpy(xp, vertex_est),
                n_local,
                weighted=False,
            )
            expected = self._to_xp(xp, expected_numpy)

            self._assert_allclose(xp, result, expected)
            self._assert_allclose(xp, result, self._to_xp(xp, np.array(true_vertices, dtype=float)), rtol=1e-5)
            self.assertEqual(result.shape, vertex_est.shape)

    def test_parabolic_vertex_returns_nan_for_vertices_at_limits(self):
        true_vertices = [2.4, 18.6, 10.5]
        curvatures = [1.1, 0.9, 0.7]
        offsets = [0.0, 1.0, -0.5]
        vertex_estimates = [0.2, 19.8, 10.2]

        n_datapoints = 21
        n_local = 5

        for xp in self.xp_modules:
            coords = xp.arange(n_datapoints, dtype=xp.float64)
            rows = []
            for vertex, curvature, offset in zip(true_vertices, curvatures, offsets):
                vertex_val = xp.float64(vertex)
                curvature_val = xp.float64(curvature)
                offset_val = xp.float64(offset)
                rows.append(curvature_val * (coords - vertex_val) ** 2 + offset_val)
            data = xp.stack(rows, axis=0)

            vertex_est = xp.array(vertex_estimates, dtype=xp.float64)

            result = magtrack.parabolic_vertex(data, vertex_est, n_local, weighted=True)

            self.assertTrue(bool(xp.isnan(result[0])))
            self.assertTrue(bool(xp.isnan(result[1])))

            expected_numpy = self._compute_expected_numpy(
                self._to_numpy(xp, data),
                self._to_numpy(xp, vertex_est),
                n_local,
                weighted=True,
            )
            expected = self._to_xp(xp, expected_numpy)

            self.assertTrue(bool(xp.isnan(expected[0])))
            self.assertTrue(bool(xp.isnan(expected[1])))

            valid_expected = expected[2]
            valid_result = result[2]
            self.assertFalse(bool(xp.isnan(valid_result)))
            self._assert_allclose(xp, valid_result, valid_expected, rtol=1e-7, atol=1e-10)
            self._assert_allclose(xp, valid_result, xp.float64(true_vertices[2]), rtol=1e-4)

    def test_parabolic_vertex_output_dtype_is_float64(self):
        n_datapoints = 17
        n_local = 5

        for xp in self.xp_modules:
            coords = xp.arange(n_datapoints, dtype=xp.float32)
            vertex = xp.float32(8.4)
            data = (coords - vertex) ** 2
            data = data[xp.newaxis, :].astype(xp.float32)

            vertex_est = xp.array([8.0], dtype=xp.float32)

            result = magtrack.parabolic_vertex(data, vertex_est, n_local, weighted=True)

            self.assertEqual(result.dtype, xp.float64)
            self.assertEqual(result.shape, (1,))


if __name__ == "__main__":
    unittest.main()
