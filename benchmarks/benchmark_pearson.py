"""Performance benchmark for magtrack.pearson."""

from __future__ import annotations

import numpy as np
import cupy as cp
from cupyx.profiler import benchmark as cupy_benchmark

import confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.cpu_benchmark import cpu_benchmark


def _generate_inputs(xp, n_samples: int, n_features: int):
    """Create random test data on the requested array module."""
    x = xp.random.random((n_samples, n_features)).astype(xp.float64)
    y = xp.random.random(n_samples).astype(xp.float64)
    return x, y


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_pearson(
    n_samples: int = 1_000_000,
    n_features: int = 16,
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.pearson`."""
    # CPU benchmark
    x_cpu, y_cpu = _generate_inputs(np, n_samples, n_features)
    cpu_results = cpu_benchmark(
        magtrack.pearson,
        args=(x_cpu.copy(), y_cpu.copy()),
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_cpu,
    )
    _print_summary("CPU", cpu_results.cpu_times)

    # GPU benchmark
    if not magtrack.utils.check_cupy():
        print("CuPy with GPU support is not available; skipping GPU benchmark.")
        return

    x_gpu, y_gpu = _generate_inputs(cp, n_samples, n_features)
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
