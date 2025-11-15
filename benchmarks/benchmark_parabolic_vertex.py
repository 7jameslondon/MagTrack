"""Performance benchmark for magtrack.parabolic_vertex."""

from __future__ import annotations

import numpy as np

from benchmarks import confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.cpu_benchmark import cpu_benchmark
from magtrack._cupy import cp, check_cupy


def _generate_inputs(
    xp,
    n_datasets: int,
    n_datapoints: int,
    n_local: int,
):
    """Create synthetic parabolic datasets on the requested array module."""

    if n_local < 3 or n_local % 2 == 0:
        raise ValueError("n_local must be an odd integer >= 3")
    if n_datapoints <= n_local:
        raise ValueError("n_datapoints must exceed n_local")

    n_local_half = n_local // 2
    vertex_min = n_local_half
    vertex_max = n_datapoints - n_local_half - 1
    if vertex_max <= vertex_min:
        raise ValueError("Configuration leaves no room for a valid vertex range")

    vertex_est = xp.asarray(
        vertex_min
        + (vertex_max - vertex_min) * xp.random.random(n_datasets),
        dtype=xp.float64,
    )

    positions = xp.arange(n_datapoints, dtype=xp.float64)[xp.newaxis, :]
    centers = vertex_est[:, xp.newaxis]

    curvatures = xp.asarray(
        0.5 + 1.5 * xp.random.random((n_datasets, 1)),
        dtype=xp.float64,
    )
    slopes = xp.asarray(
        (xp.random.random((n_datasets, 1)) - 0.5) * 0.4,
        dtype=xp.float64,
    )
    amplitudes = xp.asarray(
        1.0 + xp.random.random((n_datasets, 1)),
        dtype=xp.float64,
    )
    noise = xp.asarray(
        xp.random.standard_normal((n_datasets, n_datapoints)),
        dtype=xp.float64,
    )

    profile = amplitudes - curvatures * (positions - centers) ** 2
    profile += slopes * (positions - centers)

    data = profile + 0.05 * noise
    return data, vertex_est


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_parabolic_vertex(
    n_datasets: int = 2048,
    n_datapoints: int = 512,
    n_local: int = 7,
    weighted: bool = True,
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.parabolic_vertex`."""

    print("Benchmarking: magtrack.parabolic_vertex")
    print(
        f"n_datasets: {n_datasets}, n_datapoints: {n_datapoints}, "
        f"n_local: {n_local}, weighted: {weighted}"
    )
    print(
        f"n_repeat: {n_repeat}, n_warmup_cpu: {n_warmup_cpu}, "
        f"n_warmup_gpu: {n_warmup_gpu}, max_duration: {max_duration}"
    )

    # CPU benchmark
    data_cpu, vertex_cpu = _generate_inputs(np, n_datasets, n_datapoints, n_local)
    cpu_results = cpu_benchmark(
        magtrack.parabolic_vertex,
        args=(data_cpu.copy(), vertex_cpu.copy(), n_local),
        kwargs={"weighted": weighted},
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

    data_gpu, vertex_gpu = _generate_inputs(cp, n_datasets, n_datapoints, n_local)
    gpu_results = cupy_benchmark(
        magtrack.parabolic_vertex,
        args=(data_gpu.copy(), vertex_gpu.copy(), n_local),
        kwargs={"weighted": weighted},
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_gpu,
    )
    gpu_times = cp.asnumpy(gpu_results.gpu_times).squeeze()
    gpu_cpu_times = np.asarray(gpu_results.cpu_times).squeeze()
    _print_summary("GPU", gpu_times + gpu_cpu_times)


if __name__ == "__main__":
    benchmark_parabolic_vertex()
