"""Benchmarking utilities for ``magtrack.binmean`` across CPU and GPU backends.

This module provides a command line interface that benchmarks ``magtrack.binmean``
for several combinations of input sizes.  The script relies on ``cupyx``'s
``benchmark`` helper so it can time both CPU and GPU executions using a single
code path.  When a CUDA compatible device is unavailable the GPU benchmarks are
skipped gracefully.

Example
-------
Run the default sweep of benchmarks and write the results to ``binmean.json``::

    python benchmarks/benchmark_binmean.py --output binmean.json

Restrict the benchmark sweep to CPU executions and custom input sizes::

    python benchmarks/benchmark_binmean.py \
        --cpu-only \
        --n-values 1000 10000 \
        --n-datasets 1 4 \
        --n-bins 32 256

The resulting table printed to ``stdout`` summarises the mean, standard
deviation, minimum, and maximum execution time (in milliseconds) for each test
case.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from typing import Dict, Iterable, List, Sequence

import cupy as cp
import numpy as np
from cupyx.profiler import benchmark as cupy_benchmark

import magtrack


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    """Description of a single binmean benchmark scenario."""

    n_values: int
    n_datasets: int
    n_bins: int
    device: str  # "cpu" or "gpu"

    def label(self) -> str:
        return (
            f"values={self.n_values:,} "
            f"datasets={self.n_datasets:,} "
            f"bins={self.n_bins:,} "
            f"device={self.device}"
        )


@dataclasses.dataclass
class BenchmarkResult:
    """Execution statistics in milliseconds for a single benchmark case."""

    mean: float
    stddev: float
    minimum: float
    maximum: float
    n_repeat: int

    @classmethod
    def from_timings(cls, times: Sequence[float]) -> "BenchmarkResult":
        # ``statistics.pstdev`` provides a stable population standard deviation.
        return cls(
            mean=float(statistics.fmean(times)),
            stddev=float(statistics.pstdev(times)),
            minimum=float(min(times)),
            maximum=float(max(times)),
            n_repeat=len(times),
        )


def _binmean_wrapper(x, weights, n_bins):
    """Call ``magtrack.binmean`` while protecting the input array from mutation."""
    # ``binmean`` clips its first argument in-place.  Copying here keeps the
    # input data identical across benchmark repetitions.
    return magtrack.binmean(x.copy(), weights, n_bins)


def _collect_timings(result) -> np.ndarray:
    """Convert ``cupyx`` benchmark results to total execution times in ms."""
    # CPU executions only populate ``cpu_times`` whereas GPU runs include both
    # host and device contributions.  ``cupyx`` reports timings in seconds.
    cpu_times = np.asarray(result.cpu_times, dtype=np.float64)
    gpu_times = np.asarray(getattr(result, "gpu_times", ()), dtype=np.float64)

    if gpu_times.size:
        times = cpu_times + gpu_times
    else:
        times = cpu_times

    # Convert seconds to milliseconds for easier interpretation.
    return times * 1_000.0


def run_case(case: BenchmarkCase, repeats: int, warmup: int) -> BenchmarkResult:
    """Execute ``magtrack.binmean`` for a single benchmark configuration."""
    rng = np.random.default_rng(seed=12345)
    x_cpu = rng.integers(
        low=0,
        high=case.n_bins,
        size=(case.n_values, case.n_datasets),
        dtype=np.int32,
    )
    weights_cpu = rng.random((case.n_values, case.n_datasets), dtype=np.float32)

    if case.device == "gpu":
        x = cp.asarray(x_cpu)
        weights = cp.asarray(weights_cpu)
    else:
        x = x_cpu
        weights = weights_cpu

    result = cupy_benchmark(
        _binmean_wrapper,
        args=(x, weights, case.n_bins),
        kwargs={},
        n_repeat=repeats,
        n_warmup=warmup,
        max_duration=10.0,
    )
    return BenchmarkResult.from_timings(_collect_timings(result))


def _available_devices(request_gpu: bool) -> List[str]:
    devices = ["cpu"]
    if request_gpu:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                devices.append("gpu")
        except cp.cuda.runtime.CUDARuntimeError:
            # CUDA is not available in the current environment.
            pass
    return devices


def _iter_cases(
    n_values: Sequence[int],
    n_datasets: Sequence[int],
    n_bins: Sequence[int],
    devices: Sequence[str],
) -> Iterable[BenchmarkCase]:
    for device in devices:
        for values in n_values:
            for datasets in n_datasets:
                for bins in n_bins:
                    yield BenchmarkCase(values, datasets, bins, device)


def _format_table(results: Dict[BenchmarkCase, BenchmarkResult]) -> str:
    lines = []
    header = (
        "n_values",
        "n_datasets",
        "n_bins",
        "device",
        "mean [ms]",
        "stdev [ms]",
        "min [ms]",
        "max [ms]",
        "repeats",
    )
    column_widths = [len(col) for col in header]

    # First pass: compute maximum column widths.
    for case, result in results.items():
        values = (
            f"{case.n_values}",
            f"{case.n_datasets}",
            f"{case.n_bins}",
            case.device,
            f"{result.mean:.3f}",
            f"{result.stddev:.3f}",
            f"{result.minimum:.3f}",
            f"{result.maximum:.3f}",
            f"{result.n_repeat}",
        )
        column_widths = [max(width, len(value)) for width, value in zip(column_widths, values)]

    # Header line.
    header_line = " | ".join(col.ljust(width) for col, width in zip(header, column_widths))
    separator = "-+-".join("-" * width for width in column_widths)
    lines.append(header_line)
    lines.append(separator)

    # Rows sorted for deterministic output.
    for case in sorted(
        results.keys(),
        key=lambda c: (c.device, c.n_values, c.n_datasets, c.n_bins),
    ):
        result = results[case]
        values = (
            f"{case.n_values}",
            f"{case.n_datasets}",
            f"{case.n_bins}",
            case.device,
            f"{result.mean:.3f}",
            f"{result.stddev:.3f}",
            f"{result.minimum:.3f}",
            f"{result.maximum:.3f}",
            f"{result.n_repeat}",
        )
        lines.append(" | ".join(value.ljust(width) for value, width in zip(values, column_widths)))

    return "\n".join(lines)


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{parsed} must be positive")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=_parse_positive_int,
        default=[10_000, 100_000, 1_000_000],
        help="Sequence of input lengths (rows) to benchmark.",
    )
    parser.add_argument(
        "--n-datasets",
        nargs="+",
        type=_parse_positive_int,
        default=[1, 4, 16],
        help="Sequence of dataset counts (columns) to benchmark.",
    )
    parser.add_argument(
        "--n-bins",
        nargs="+",
        type=_parse_positive_int,
        default=[32, 256, 1024],
        help="Sequence of bin counts to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=_parse_positive_int,
        default=30,
        help="Number of timed repetitions for each benchmark case.",
    )
    parser.add_argument(
        "--warmup",
        type=_parse_positive_int,
        default=5,
        help="Number of warmup iterations before timing begins.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU benchmarks even if a CUDA device is available.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to a JSON file where raw results will be stored.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    devices = _available_devices(request_gpu=not args.cpu_only)
    if not devices:
        parser.error("No valid execution devices available for benchmarking.")

    cases = list(
        _iter_cases(
            n_values=args.n_values,
            n_datasets=args.n_datasets,
            n_bins=args.n_bins,
            devices=devices,
        )
    )

    results: Dict[BenchmarkCase, BenchmarkResult] = {}
    for case in cases:
        print(f"Running {case.label()} ...", flush=True)
        result = run_case(case, repeats=args.repeats, warmup=args.warmup)
        results[case] = result

    print()
    print(_format_table(results))

    if args.output is not None:
        serialisable = {
            dataclasses.asdict(case): dataclasses.asdict(result)
            for case, result in results.items()
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
