import numpy as np

from magtrack.core import qi_refine_xy


def _paraboloid(size, center_x, center_y):
    y = np.arange(size, dtype=np.float64)
    x = np.arange(size, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    values = 10.0 - (xx - center_x) ** 2 - (yy - center_y) ** 2
    return values


def test_qi_refine_xy_recovers_quadratic_center():
    size = 9
    true_x = np.array([3.4, 5.1], dtype=np.float64)
    true_y = np.array([4.2, 2.7], dtype=np.float64)

    stack = np.stack(
        [
            _paraboloid(size, cx, cy)
            for cx, cy in zip(true_x, true_y)
        ],
        axis=-1,
    )

    x_old = np.floor(true_x)
    y_old = np.floor(true_y)

    refined_x, refined_y = qi_refine_xy(stack, x_old, y_old)

    assert np.allclose(refined_x, true_x, atol=1e-6)
    assert np.allclose(refined_y, true_y, atol=1e-6)
