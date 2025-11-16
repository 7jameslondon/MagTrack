"""Performance benchmark for :func:`magtrack.center_of_mass`."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmarks.speed import confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.speed.cpu_benchmark import cpu_benchmark
from magtrack._cupy import cp, check_cupy
from magtrack.simulation import simulate_beads


_BASE_LAYOUT_NM = (
    np.array(
        [
            [-0.02, 0.015, -0.05],
            [0.018, -0.022, 0.02],
            [0.005, 0.007, -0.01],
            [-0.012, -0.018, 0.04],
            [0.02, 0.012, -0.03],
            [-0.015, 0.022, 0.0],
            [0.017, -0.005, 0.01],
            [-0.018, 0.009, -0.02],
            [0.008, -0.01, 0.03],
            [-0.01, 0.0, -0.04],
        ],
        dtype=np.float64,
    )
    * 1e3
)


def _generate_inputs(
    xp: Any,
    *,
    n_images: int,
    nm_per_px: float,
    size_px: int,
    seed: int,
) -> Any:
    """Generate a bead stack for benchmarking."""

    if n_images <= 0:
        raise ValueError("n_images must be a positive integer")

    rng = np.random.default_rng(seed)
    base = _BASE_LAYOUT_NM
    repeats = math.ceil(n_images / base.shape[0])
    xyz_nm = np.tile(base, (repeats, 1))[:n_images].copy()

    # Introduce small random perturbations to avoid identical images while
    # preserving the overall bead layout used in the tests.
    jitter_scale = np.array([50.0, 50.0, 75.0], dtype=np.float64)
    jitter = rng.normal(scale=jitter_scale, size=xyz_nm.shape)
    xyz_nm += jitter

    stack_np = simulate_beads(
        xyz_nm,
        nm_per_px=nm_per_px,
        size_px=size_px,
    ).astype(np.float64, copy=False)

    if xp is np:
        return stack_np
    if xp is cp:
        return cp.asarray(stack_np)
    raise TypeError("xp must be either numpy or cupy")


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_center_of_mass(
    *,
    n_images: int = 1000,
    nm_per_px: float = 100.0,
    size_px: int = 64,
    background: str = "mean",
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
    seed: int = 12345,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.center_of_mass`."""

    print("Benchmarking: magtrack.center_of_mass")
    print(
        "n_images: {n_images}, nm_per_px: {nm_per_px}, size_px: {size_px}, "
        "background: {background}".format(
            n_images=n_images,
            nm_per_px=nm_per_px,
            size_px=size_px,
            background=background,
        )
    )
    print(
        "n_repeat: {n_repeat}, n_warmup_cpu: {n_warmup_cpu}, "
        "n_warmup_gpu: {n_warmup_gpu}, max_duration: {max_duration}, "
        "seed: {seed}".format(
            n_repeat=n_repeat,
            n_warmup_cpu=n_warmup_cpu,
            n_warmup_gpu=n_warmup_gpu,
            max_duration=max_duration,
            seed=seed,
        )
    )

    stack_cpu = _generate_inputs(
        np,
        n_images=n_images,
        nm_per_px=nm_per_px,
        size_px=size_px,
        seed=seed,
    )

    cpu_results = cpu_benchmark(
        magtrack.center_of_mass,
        args=(stack_cpu,),
        kwargs={"background": background},
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_cpu,
    )
    _print_summary("CPU", cpu_results.cpu_times)

    if not check_cupy():
        print("CuPy with GPU support is not available; skipping GPU benchmark.")
        return

    from cupyx.profiler import benchmark as cupy_benchmark  # type: ignore

    stack_gpu = _generate_inputs(
        cp,
        n_images=n_images,
        nm_per_px=nm_per_px,
        size_px=size_px,
        seed=seed,
    )

    gpu_results = cupy_benchmark(
        magtrack.center_of_mass,
        args=(stack_gpu,),
        kwargs={"background": background},
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_gpu,
    )
    gpu_times = cp.asnumpy(gpu_results.gpu_times).squeeze()
    gpu_cpu_times = np.asarray(gpu_results.cpu_times).squeeze()
    _print_summary("GPU", gpu_times + gpu_cpu_times)


if __name__ == "__main__":
    benchmark_center_of_mass()
