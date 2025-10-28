"""Performance benchmark for magtrack.gaussian."""

from __future__ import annotations

import numpy as np

import confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.cpu_benchmark import cpu_benchmark
from magtrack._cupy import cp


def _generate_inputs(
    xp,
    n_values: int,
    dtype,
    sigma_min: float,
):
    """Create random test data on the requested array module."""

    xp_dtype = xp.dtype(dtype)
    x = xp.random.standard_normal(n_values).astype(xp_dtype)
    mu = xp.random.standard_normal(n_values).astype(xp_dtype)
    sigma = xp.random.random(n_values).astype(xp_dtype)
    sigma = xp.maximum(sigma, xp.asarray(sigma_min, dtype=xp_dtype))
    return x, mu, sigma


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_gaussian(
    n_values: int = 1_000_000,
    dtype=np.float64,
    sigma_min: float = 1e-3,
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.gaussian`."""

    print('Benchmarking: magtrack.gaussian')
    dtype = np.dtype(dtype)
    print(f"n_values: {n_values}, dtype: {dtype.name}")
    print(
        f"n_repeat: {n_repeat}, n_warmup_cpu: {n_warmup_cpu}, "
        f"n_warmup_gpu: {n_warmup_gpu}, max_duration: {max_duration}"
    )
    print('')

    # CPU benchmark
    x_cpu, mu_cpu, sigma_cpu = _generate_inputs(
        np,
        n_values,
        dtype,
        sigma_min,
    )
    cpu_results = cpu_benchmark(
        magtrack.gaussian,
        args=(x_cpu, mu_cpu, sigma_cpu),
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_cpu,
    )
    _print_summary("CPU", cpu_results.cpu_times)

    # GPU benchmark
    if not magtrack.utils.check_cupy():
        print("CuPy with GPU support is not available; skipping GPU benchmark.")
        return

    from cupyx.profiler import benchmark as cupy_benchmark  # type: ignore

    x_gpu, mu_gpu, sigma_gpu = _generate_inputs(
        cp,
        n_values,
        dtype,
        sigma_min,
    )
    gpu_results = cupy_benchmark(
        magtrack.gaussian,
        args=(x_gpu, mu_gpu, sigma_gpu),
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_gpu,
    )
    gpu_times = cp.asnumpy(gpu_results.gpu_times).squeeze()
    gpu_cpu_times = np.asarray(gpu_results.cpu_times).squeeze()
    _print_summary("GPU", gpu_times + gpu_cpu_times)


if __name__ == "__main__":
    benchmark_gaussian()
