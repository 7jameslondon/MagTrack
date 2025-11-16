"""Compatibility wrapper for :mod:`benchmarks.speed.run_all`."""

from __future__ import annotations

from benchmarks.speed.run_all import (
    BenchmarkFunc,
    BenchmarkRecorder,
    discover_benchmark_functions,
    discover_benchmark_modules,
    main as _speed_main,
    run_benchmarks,
)

__all__ = [
    "BenchmarkFunc",
    "BenchmarkRecorder",
    "discover_benchmark_functions",
    "discover_benchmark_modules",
    "main",
    "run_benchmarks",
]


def main(argv: list[str] | None = None) -> int:
    """Delegate execution to :func:`benchmarks.speed.run_all.main`."""

    return _speed_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
