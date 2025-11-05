import importlib
import unittest


class TestMagtrackImports(unittest.TestCase):
    def test_stack_to_xyzp_is_exposed(self):
        module = importlib.import_module("magtrack")
        self.assertIsNotNone(getattr(module, "stack_to_xyzp", None))


if __name__ == "__main__":
    unittest.main()
