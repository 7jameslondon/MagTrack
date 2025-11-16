"""Programmatic API for localization accuracy benchmarks."""

from __future__ import annotations

from benchmarks.accuracy.montage import (
    BeadImage,
    generate_accuracy_benchmark_images,
    show_accuracy_montage,
)
from benchmarks.accuracy.runner import get_latest_csv, run_full_accuracy_benchmark
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
    "BeadImage",
    "DEFAULT_CAMERA_PIXEL_SIZE_NM",
    "DEFAULT_SWEEP_CONFIG",
    "MethodFunc",
    "generate_accuracy_benchmark_images",
    "format_summary",
    "get_latest_csv",
    "register_method",
    "run_accuracy_sweep",
    "run_full_accuracy_benchmark",
    "show_accuracy_montage",
    "summarize_accuracy",
]
