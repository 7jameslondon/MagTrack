"""Utility package for MagTrack benchmarks.

Speed benchmarks live under :mod:`benchmarks.speed` and retain the existing
runtime benchmarking harness. A companion :mod:`benchmarks.accuracy`
subpackage will eventually house localization accuracy suites.
"""

from __future__ import annotations

__all__ = ["speed", "accuracy"]
