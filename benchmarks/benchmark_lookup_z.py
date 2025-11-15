"""Performance benchmark for :func:`magtrack.lookup_z`."""

from __future__ import annotations

import numpy as np

from benchmarks import confbenchmarks as confbenchmarks  # noqa: F401  # Ensures repository root on sys.path
import magtrack
from benchmarks.cpu_benchmark import cpu_benchmark
from magtrack._cupy import cp, check_cupy
from magtrack.simulation import simulate_beads


def _generate_reference_zlut(
    *,
    nm_per_px: float,
    roi_px: int,
    z_min_nm: float,
    z_max_nm: float,
    z_step_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a radial-profile lookup table spanning the requested z range."""

    z_reference_nm = np.arange(z_min_nm, z_max_nm + 0.5 * z_step_nm, z_step_nm, dtype=np.float64)

    xyz_reference_nm = np.column_stack(
        [
            np.zeros_like(z_reference_nm),
            np.zeros_like(z_reference_nm),
            z_reference_nm,
        ]
    )

    reference_stack = simulate_beads(
        xyz_reference_nm,
        size_px=roi_px,
        nm_per_px=nm_per_px,
    ).astype(np.float64, copy=False)

    center = roi_px / 2.0
    centers = np.full(z_reference_nm.shape, center, dtype=np.float64)
    reference_profiles = magtrack.radial_profile(reference_stack, centers, centers)

    zlut_np = np.vstack([z_reference_nm, reference_profiles.astype(np.float64, copy=False)])
    return z_reference_nm, zlut_np


def _generate_eval_profiles(
    *,
    nm_per_px: float,
    roi_px: int,
    n_profiles: int,
    amplitude_nm: float,
    z_min_nm: float,
    z_max_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate evaluation profiles and their corresponding z positions."""

    z_true_nm = np.linspace(z_min_nm, z_max_nm, n_profiles, dtype=np.float64)
    oscillation = amplitude_nm * np.sin(np.linspace(0.0, 6.0 * np.pi, n_profiles, dtype=np.float64))
    z_true_nm = z_true_nm + oscillation

    xyz_eval_nm = np.column_stack(
        [
            np.zeros_like(z_true_nm),
            np.zeros_like(z_true_nm),
            z_true_nm,
        ]
    )

    eval_stack = simulate_beads(
        xyz_eval_nm,
        size_px=roi_px,
        nm_per_px=nm_per_px,
    ).astype(np.float64, copy=False)

    center = roi_px / 2.0
    centers = np.full(z_true_nm.shape, center, dtype=np.float64)
    eval_profiles_np = magtrack.radial_profile(eval_stack, centers, centers).astype(
        np.float64,
        copy=False,
    )

    return z_true_nm, eval_profiles_np


def _to_xp(xp_module, array: np.ndarray):
    if xp_module is cp:
        return cp.asarray(array)
    if xp_module is np:
        return array
    raise TypeError("xp_module must be numpy or cupy")


def _print_summary(label: str, times: np.ndarray) -> None:
    times = np.asarray(times, dtype=float).squeeze()
    mean = float(times.mean())
    std = float(times.std())
    print(f"{label}: mean {mean:.6f}s ± {std:.6f}s over {times.size} runs")


def benchmark_lookup_z(
    *,
    n_profiles: int = 60,
    nm_per_px: float = 100.0,
    roi_px: int = 64,
    z_min_nm: float = -3000.0,
    z_max_nm: float = 3000.0,
    zlut_min_nm: float = -10000.0,
    zlut_max_nm: float = 10000.0,
    zlut_step_nm: float = 200.0,
    oscillation_amplitude_nm: float = 800.0,
    n_local: int = 5,
    n_repeat: int = 100,
    n_warmup_cpu: int = 10,
    n_warmup_gpu: int = 10,
    max_duration: float = 30.0,
) -> None:
    """Run CPU and GPU benchmarks for :func:`magtrack.lookup_z`."""

    print("Benchmarking: magtrack.lookup_z")
    print(
        "n_profiles: {n_profiles}, nm_per_px: {nm_per_px}, roi_px: {roi_px}, "
        "n_local: {n_local}".format(
            n_profiles=n_profiles,
            nm_per_px=nm_per_px,
            roi_px=roi_px,
            n_local=n_local,
        )
    )
    print(
        "zlut range: [{zlut_min}, {zlut_max}] nm step {zlut_step} nm; profile z range: "
        "[{z_min}, {z_max}] nm with oscillation amplitude {amp} nm".format(
            zlut_min=zlut_min_nm,
            zlut_max=zlut_max_nm,
            zlut_step=zlut_step_nm,
            z_min=z_min_nm,
            z_max=z_max_nm,
            amp=oscillation_amplitude_nm,
        )
    )
    print(
        "n_repeat: {n_repeat}, n_warmup_cpu: {n_warmup_cpu}, n_warmup_gpu: {n_warmup_gpu}, "
        "max_duration: {max_duration}".format(
            n_repeat=n_repeat,
            n_warmup_cpu=n_warmup_cpu,
            n_warmup_gpu=n_warmup_gpu,
            max_duration=max_duration,
        )
    )

    _, zlut_np = _generate_reference_zlut(
        nm_per_px=nm_per_px,
        roi_px=roi_px,
        z_min_nm=zlut_min_nm,
        z_max_nm=zlut_max_nm,
        z_step_nm=zlut_step_nm,
    )
    _, eval_profiles_np = _generate_eval_profiles(
        nm_per_px=nm_per_px,
        roi_px=roi_px,
        n_profiles=n_profiles,
        amplitude_nm=oscillation_amplitude_nm,
        z_min_nm=z_min_nm,
        z_max_nm=z_max_nm,
    )

    # CPU benchmark
    profiles_cpu = eval_profiles_np
    zlut_cpu = zlut_np
    cpu_results = cpu_benchmark(
        magtrack.lookup_z,
        args=(profiles_cpu, zlut_cpu),
        kwargs={"n_local": n_local},
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

    profiles_gpu = _to_xp(cp, eval_profiles_np)
    zlut_gpu = _to_xp(cp, zlut_np)
    gpu_results = cupy_benchmark(
        magtrack.lookup_z,
        args=(profiles_gpu, zlut_gpu),
        kwargs={"n_local": n_local},
        max_duration=max_duration,
        n_repeat=n_repeat,
        n_warmup=n_warmup_gpu,
    )
    gpu_times = cp.asnumpy(gpu_results.gpu_times).squeeze()
    gpu_cpu_times = np.asarray(gpu_results.cpu_times).squeeze()
    _print_summary("GPU", gpu_times + gpu_cpu_times)


if __name__ == "__main__":
    benchmark_lookup_z()
