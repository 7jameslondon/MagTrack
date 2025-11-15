"""Performance benchmark for magtrack.binmean."""

from __future__ import annotations

import numpy as np

from benchmarks import confbenchmarks as confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.cpu_benchmark import cpu_benchmark
from magtrack._cupy import cp, check_cupy


def _generate_inputs(xp, n_values: int, n_datasets: int, n_bins: int):
    """Create random test data on the requested array module."""
    x = xp.random.randint(0, n_bins, size=(n_values, n_datasets)).astype(xp.int64)
    weights = xp.random.random((n_values, n_datasets)).astype(xp.float64)
    return x, weights


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_binmean(
    n_values: int = 1_000_000,
    n_datasets: int = 16,
    n_bins: int = 64,
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.binmean`."""

    print('Benchmarking: magtrack.binmean')
    print(f"n_values: {n_values}, n_datasets: {n_datasets}, n_bins: {n_bins}")
    print(f"n_repeat: {n_repeat}, n_warmup_cpu: {n_warmup_cpu}, n_warmup_gpu: {n_warmup_gpu}, max_duration: {max_duration}")

    # CPU benchmark
    x_cpu, weights_cpu = _generate_inputs(np, n_values, n_datasets, n_bins)
    cpu_results = cpu_benchmark(
        magtrack.binmean,
        args=(x_cpu.copy(), weights_cpu, n_bins),
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_cpu,
    )
    _print_summary("CPU", cpu_results.cpu_times)

    # GPU benchmark
    if not check_cupy():
        print("CuPy with GPU support is not available; skipping GPU benchmark.")
        return

    from cupyx.profiler import benchmark as cupy_benchmark  # type: ignore

    x_gpu, weights_gpu = _generate_inputs(cp, n_values, n_datasets, n_bins)
    gpu_results = cupy_benchmark(
        magtrack.binmean,
        args=(x_gpu.copy(), weights_gpu, n_bins),
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_gpu,
    )
    gpu_times = cp.asnumpy(gpu_results.gpu_times).squeeze()
    gpu_cpu_times = np.asarray(gpu_results.cpu_times).squeeze()
    _print_summary("GPU", gpu_times + gpu_cpu_times)


if __name__ == "__main__":
    benchmark_binmean()
