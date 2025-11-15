from __future__ import annotations

import importlib
import sys


def test_confbenchmarks_alias_is_registered(monkeypatch):
    monkeypatch.syspath_prepend("/non-existent")
    for name in ["confbenchmarks", "benchmarks", "benchmarks.confbenchmarks"]:
        sys.modules.pop(name, None)

    pkg = importlib.import_module("benchmarks")
    assert "benchmarks.confbenchmarks" in sys.modules
    assert "confbenchmarks" in sys.modules
    assert sys.modules["confbenchmarks"] is sys.modules["benchmarks.confbenchmarks"]
    assert pkg is sys.modules["benchmarks"]
