import unittest

import numpy as np

import tests.conftests  # noqa: F401  # Ensure test package path setup
import magtrack
from magtrack._cupy import cp


class TestCenterOfMass(unittest.TestCase):
    if magtrack.utils.check_cupy():
        xp_modules = (np, cp)
    else:
        xp_modules = (np,)

    def test_center_of_mass_not_implemented(self):
        self.skipTest("Center of mass tests have not been implemented yet")


if __name__ == "__main__":
    unittest.main()
