import importlib


def test_stack_to_xyzp_is_exposed():
    module = importlib.import_module("magtrack")
    assert getattr(module, "stack_to_xyzp") is not None
