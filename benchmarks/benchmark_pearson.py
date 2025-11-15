"""Performance benchmark for magtrack.pearson."""

from __future__ import annotations

import numpy as np

from benchmarks import confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.cpu_benchmark import cpu_benchmark
from magtrack._cupy import cp, check_cupy


def _generate_inputs(xp, n_bins: int, n_refs: int, n_profiles: int):
    """Create random test data on the requested array module."""
    x = xp.random.random((n_bins, n_refs)).astype(xp.float64)
    y = xp.random.random((n_bins, n_profiles)).astype(xp.float64)
    return x, y


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_pearson(
    n_bins: int = 100,
    n_refs: int = 200,
    n_profiles: int = 1000,
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.pearson`."""

    print('Benchmarking: magtrack.pearson')
    print(f"n_bins: {n_bins}, n_refs: {n_refs}, n_profiles: {n_profiles}")
    print(f"n_repeat: {n_repeat}, n_warmup_cpu: {n_warmup_cpu}, n_warmup_gpu: {n_warmup_gpu}, max_duration: {max_duration}")

    # CPU benchmark
    x_cpu, y_cpu = _generate_inputs(np, n_bins, n_refs, n_profiles)
    cpu_results = cpu_benchmark(
        magtrack.pearson,
        args=(x_cpu.copy(), y_cpu.copy()),
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

    x_gpu, y_gpu = _generate_inputs(cp, n_bins, n_refs, n_profiles)
    gpu_results = cupy_benchmark(
        magtrack.pearson,
        args=(x_gpu.copy(), y_gpu.copy()),
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_gpu,
    )
    gpu_times = cp.asnumpy(gpu_results.gpu_times).squeeze()
    gpu_cpu_times = np.asarray(gpu_results.cpu_times).squeeze()
    _print_summary("GPU", gpu_times + gpu_cpu_times)


if __name__ == "__main__":
    benchmark_pearson()
