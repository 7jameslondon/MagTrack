"""Programmatic API for localization accuracy benchmarks."""

from __future__ import annotations

from benchmarks.accuracy.runner import (
    get_latest_csv,
    plot_accuracy_results,
    run_full_accuracy_benchmark,
)
from benchmarks.accuracy.xy_accuracy import (
    AccuracySweepConfig,
    AccuracySweepResults,
    DEFAULT_CAMERA_PIXEL_SIZE_NM,
    DEFAULT_SWEEP_CONFIG,
    MethodFunc,
    format_summary,
    register_method,
    run_accuracy_sweep,
    summarize_accuracy,
)

__all__ = [
    "AccuracySweepConfig",
    "AccuracySweepResults",
    "DEFAULT_CAMERA_PIXEL_SIZE_NM",
    "DEFAULT_SWEEP_CONFIG",
    "MethodFunc",
    "format_summary",
    "get_latest_csv",
    "plot_accuracy_results",
    "register_method",
    "run_accuracy_sweep",
    "run_full_accuracy_benchmark",
    "summarize_accuracy",
]
