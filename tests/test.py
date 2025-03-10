import unittest

import numpy as np
import cupy as cp
from cupy.testing import assert_array_almost_equal

import magtrack

# TODO: Add more tests
class Test(unittest.TestCase):
    def test_binmean(self):
        for xp in [np, cp]:
            x = xp.array([[0, 0],
                          [1, 2],
                          [3, 2],
                          [3, 2]], dtype='int')
            weights = xp.array([[0.0, 0.1],
                                [0.1, 0.1],
                                [0.1, 0.1],
                                [0.1, 0.4]], dtype='float')
            n_bins = 5
            return_value = magtrack.binmean(x, weights, n_bins)
            expected_value = xp.array([[0.0, 0.1],
                                       [0.1, xp.nan],
                                       [xp.nan, 0.2],
                                       [0.1, xp.nan],
                                       [xp.nan, xp.nan]], dtype='float')
            assert_array_almost_equal(return_value, expected_value)


if __name__ == '__main__':
    unittest.main()
