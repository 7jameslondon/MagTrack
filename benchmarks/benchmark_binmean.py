"""Benchmarking script for magtrack.binmean."""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence
import importlib
import importlib.util

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_cp_spec = importlib.util.find_spec("cupy")
cp = None
cupy_benchmark = None
_gpu_backend_available = False
if _cp_spec is not None:
    with contextlib.suppress(ImportError, ModuleNotFoundError):
        cp = importlib.import_module("cupy")
        cupyx_profiler = importlib.import_module("cupyx.profiler")
        cupy_benchmark = getattr(cupyx_profiler, "benchmark")
        _gpu_backend_available = True

_magtrack_binmean: Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = None
_magtrack_import_attempted = False


def _load_magtrack_binmean() -> Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]]:
    global _magtrack_binmean, _magtrack_import_attempted
    if _magtrack_import_attempted:
        return _magtrack_binmean

    _magtrack_import_attempted = True
    with contextlib.suppress(ImportError, ModuleNotFoundError, AttributeError):
        module = importlib.import_module("magtrack")
        _magtrack_binmean = getattr(module, "binmean")

    return _magtrack_binmean


@dataclass
class BenchmarkResult:
    """Container for benchmark summary statistics."""

    backend: str
    timings: List[float]
    mean: float
    stdev: float

    @classmethod
    def from_timings(cls, backend: str, timings: Sequence[float]) -> "BenchmarkResult":
        """Create a result from a sequence of timing measurements."""

        native_timings = [float(value) for value in timings]
        mean = statistics.fmean(native_timings)
        stdev = statistics.pstdev(native_timings) if len(native_timings) > 1 else 0.0
        return cls(backend=backend, timings=native_timings, mean=mean, stdev=stdev)

    @property
    def n_samples(self) -> int:
        return len(self.timings)


def _numpy_binmean(x: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Pure NumPy fallback implementation of ``magtrack.binmean``."""

    clipped = np.minimum(x, n_bins, out=x.copy())
    n_datasets = clipped.shape[1]
    indices = np.arange(n_datasets, dtype=np.min_scalar_type(n_datasets))
    indices = np.broadcast_to(indices, clipped.shape)

    bin_means = np.zeros((n_bins + 1, n_datasets), dtype=weights.dtype)
    np.add.at(bin_means, (clipped, indices), weights)

    bin_counts = np.zeros((n_bins + 1, n_datasets), dtype=np.uint32)
    np.add.at(bin_counts, (clipped, indices), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        bin_means /= bin_counts

    return bin_means[:-1, :]


def _generate_inputs(
    n_values: int, n_datasets: int, n_bins: int, xp_module
):
    """Generate deterministic inputs for the benchmark."""

    rng = np.random.default_rng(seed=12345)
    base_x = rng.integers(0, n_bins, size=(n_values, n_datasets), dtype=np.int32)
    base_weights = rng.random(size=(n_values, n_datasets), dtype=np.float64)

    if xp_module is np:
        return base_x, base_weights

    return xp_module.asarray(base_x), xp_module.asarray(base_weights, dtype=xp_module.float64)


def _collect_timings(
    func,
    args: Sequence,
    use_gpu: bool,
    *,
    max_duration: float,
    n_repeat: int,
    n_warmup: int,
) -> List[float]:
    """Collect timing measurements for ``func``.

    When ``use_gpu`` is True the timings are gathered via ``cupy_benchmark`` and the
    GPU and CPU components are combined before being flattened into a simple list of
    native ``float`` objects. For CPU timings a manual loop with ``time.perf_counter``
    is used.
    """

    if use_gpu and cupy_benchmark is not None:
        benchmark_result = cupy_benchmark(
            func,
            args=args,
            kwargs={},
            max_duration=max_duration,
            n_repeat=n_repeat,
            n_warmup=n_warmup,
        )

        cpu_times = np.asarray(benchmark_result.cpu_times, dtype=np.float64)
        gpu_times = np.asarray(benchmark_result.gpu_times, dtype=np.float64)
        if gpu_times.size:
            gpu_times = gpu_times.reshape(cpu_times.shape)
            combined = cpu_times + gpu_times
        else:
            combined = cpu_times

        return np.asarray(combined, dtype=np.float64).ravel().tolist()

    # CPU path
    timings: List[float] = []
    for _ in range(max(n_warmup, 0)):
        func(*args)

    for _ in range(max(n_repeat, 1)):
        start = time.perf_counter()
        func(*args)
        stop = time.perf_counter()
        timings.append(float(stop - start))

    return timings


def _format_results(results: Iterable[BenchmarkResult]) -> str:
    lines = []
    header = f"{'Backend':<8} {'Samples':>7} {'Mean (ms)':>12} {'Std (ms)':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        lines.append(
            f"{result.backend:<8} {result.n_samples:>7d} "
            f"{result.mean * 1_000:>12.3f} {result.stdev * 1_000:>11.3f}"
        )
    return "\n".join(lines)


def run_benchmark(
    n_values: int,
    n_datasets: int,
    n_bins: int,
    use_gpu: bool,
    max_duration: float,
    n_repeat: int,
    n_warmup: int,
) -> BenchmarkResult:
    xp_module = cp if use_gpu and _gpu_backend_available and cp is not None else np
    x, weights = _generate_inputs(n_values, n_datasets, n_bins, xp_module)

    binmean_callable = _load_magtrack_binmean()
    if use_gpu:
        if not (_gpu_backend_available and cp is not None and cupy_benchmark is not None):
            raise RuntimeError("GPU benchmarking requires CuPy support.")
        if binmean_callable is None:
            raise RuntimeError("magtrack.binmean is unavailable; cannot run GPU benchmark.")
    backend_callable: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
    if binmean_callable is not None and (use_gpu or cp is not None):
        backend_callable = binmean_callable
    else:
        backend_callable = _numpy_binmean

    def _binmean_callable(x_array, weights_array, bins: int):
        return backend_callable(x_array.copy(), weights_array, bins)

    args = (x, weights, n_bins)
    timings = _collect_timings(
        _binmean_callable,
        args,
        use_gpu and _gpu_backend_available and cp is not None,
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup,
    )
    backend_name = "GPU" if use_gpu and _gpu_backend_available and cp is not None else "CPU"
    return BenchmarkResult.from_timings(backend_name, timings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark magtrack.binmean")
    parser.add_argument("--n-values", type=int, default=100_000, help="Number of values")
    parser.add_argument("--n-datasets", type=int, default=8, help="Number of datasets")
    parser.add_argument("--n-bins", type=int, default=64, help="Number of bins")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        help="Maximum duration for each benchmark measurement (GPU only)",
    )
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=10,
        help="Number of timing repetitions",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run CPU benchmark only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = [
        run_benchmark(
            args.n_values,
            args.n_datasets,
            args.n_bins,
            use_gpu=False,
            max_duration=args.max_duration,
            n_repeat=args.n_repeat,
            n_warmup=args.n_warmup,
        )
    ]

    if not args.cpu_only:
        if not (_gpu_backend_available and cupy_benchmark is not None):
            print("CuPy is not available. Skipping GPU benchmark.")
        elif _load_magtrack_binmean() is None:
            print("magtrack.binmean could not be imported. Skipping GPU benchmark.")
        else:
            results.append(
                run_benchmark(
                    args.n_values,
                    args.n_datasets,
                    args.n_bins,
                    use_gpu=True,
                    max_duration=args.max_duration,
                    n_repeat=args.n_repeat,
                    n_warmup=args.n_warmup,
                )
            )

    print(_format_results(results))


if __name__ == "__main__":
    main()
