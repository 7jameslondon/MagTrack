"""
Benchmark MagTrack core functions on CPU and GPU using synthetic Airy bead stacks.
"""

import numpy as np
import pandas as pd
from time import time
from scipy.special import j1
from cpu_benchmark import cpu_benchmark
from cupyx.profiler import benchmark as cupy_benchmark
import cupy as cp
import magtrack


# ==========================================================
#  Airy bead image generator
# ==========================================================
def bead_airy_px(
    size=512,
    r1_frac=0.0456,
    amp=1.0,
    bg=0.0,
    center=None,
    gamma=0.2,
    shot_scale=0.0,
    rng=None,
):
    """
    Generate an Airy diffraction pattern with sub-pixel center placement,
    gamma correction, and optional Poisson shot noise.
    """
    if center is None:
        center = (size / 2.0, size / 2.0)
    cx, cy = map(float, center)
    if rng is None:
        rng = np.random.default_rng()

    r1_px = r1_frac * size
    x = np.arange(size, dtype=np.float64) - cx
    y = np.arange(size, dtype=np.float64) - cy
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)

    xarg = 3.83170597 * (R / max(r1_px, 1e-6))
    I = np.ones_like(R, dtype=np.float64)
    nz = xarg != 0
    I[nz] = (2.0 * j1(xarg[nz]) / xarg[nz]) ** 2

    I /= I.max() + 1e-12
    I = I ** gamma
    img = amp * I + bg

    if shot_scale > 0:
        photons = img * shot_scale
        noisy = rng.poisson(np.clip(photons, 0, None)).astype(np.float64) / shot_scale
        img = noisy

    return img.astype(np.float64)


# ==========================================================
#  Synthetic stack generator
# ==========================================================
def bead_stack(
    n_images=32,
    size=256,
    r1_frac=0.0456,
    amp=1.0,
    bg=0.0,
    gamma=0.2,
    shot_scale=500.0,
    center_jitter_frac=0.2,
    rng=None,
):
    """Generate stack (height, width, frames) with random bead centers."""
    if rng is None:
        rng = np.random.default_rng()

    half = size / 2.0
    max_offset = size * center_jitter_frac / 2.0
    stack = np.empty((n_images, size, size), dtype=np.float64)

    for i in range(n_images):
        dx = rng.uniform(-max_offset, max_offset)
        dy = rng.uniform(-max_offset, max_offset)
        cx = half + dx
        cy = half + dy

        img = bead_airy_px(
            size=size,
            r1_frac=r1_frac,
            amp=amp,
            bg=bg,
            center=(cx, cy),
            gamma=gamma,
            shot_scale=shot_scale,
            rng=rng,
        )
        stack[i] = img

    # transpose to (height, width, frames)
    stack = np.transpose(stack, (1, 2, 0))
    return stack


# ==========================================================
#  Benchmark setup
# ==========================================================
BENCH_FUNCS = [
    # "auto_conv_para_fit",
    # "radial_profile",
    # "fft_profile",
    # "lookup_z_para_fit",
    "stack_to_xyzp",
]

# IMAGE_SIZES = [16, 32, 64, 128, 256]
# N_IMAGES = np.round(10**np.linspace(1, 6, 100)).astype('int')
# N_REPEAT = 100

IMAGE_SIZES = [16, 64, 256]
N_IMAGES = np.round(10**np.linspace(1, 6, 10)).astype('int')
N_REPEAT = 10

# ==========================================================
#  Benchmark loop (CPU + GPU)
# ==========================================================
records = []
rng = np.random.default_rng(42)

cp.cuda.set_allocator(None)

start_time = time()

for func_name in BENCH_FUNCS:
    func = getattr(magtrack, func_name)
    print(f"Benchmarking {func_name}")

    for size in IMAGE_SIZES:
        for n_img in N_IMAGES:
            if n_img * size * size * 8 > 2e9:
                continue

            # create synthetic bead stack
            stack_cpu = bead_stack(
                n_images=n_img,
                size=size,
                shot_scale=800.0,
                rng=rng,
            )
            kwargs = {}

            # choose arguments
            match func_name:
                case 'center_of_mass':
                    args_cpu = (stack_cpu, )
                case 'auto_conv' | "auto_conv_para_fit" | "radial_profile" | "fft_profile":
                    x = size/2 * np.ones((n_img, ), dtype=np.float64)
                    y = size/2 * np.ones((n_img,), dtype=np.float64)
                    args_cpu = (stack_cpu, x, y)
                case 'lookup_z_para_fit':
                    profiles = np.random.uniform(0.001, 1.3, size=((size//4) * 1, n_img)).astype(np.float64)
                    zlut = np.random.uniform(0.001, 1.3, size=(1 + (size // 4) * 1, 200)).astype(np.float64)
                    args_cpu = (profiles, zlut)
                case 'stack_to_xyzp':
                    zlut = np.random.uniform(0.001, 1.3, size=(1 + (size // 4) * 1, 200)).astype(np.float64)
                    args_cpu = (stack_cpu, zlut)

            # ---------- CPU benchmark ----------
            result_cpu = cpu_benchmark(func, args=args_cpu, kwargs=kwargs, n_repeat=N_REPEAT)
            mean_cpu = float(np.nanmean(result_cpu.cpu_times))
            sd_cpu = float(np.nanstd(result_cpu.cpu_times))

            # ---------- GPU benchmark ----------
            stack_gpu = cp.asarray(stack_cpu)
            if func_name == 'center_of_mass':
                args_gpu = (stack_gpu,)
            elif func_name in ['auto_conv', "auto_conv_para_fit", "radial_profile", "fft_profile"]:
                x_gpu = cp.asarray(x)
                y_gpu = cp.asarray(y)
                args_gpu = (stack_gpu, x_gpu, y_gpu)
            elif func_name == 'lookup_z_para_fit':
                profiles_gpu = cp.asarray(profiles)
                zlut_gpu = cp.asarray(zlut)
                args_gpu = (profiles_gpu, zlut_gpu)
            elif func_name == 'stack_to_xyzp':
                zlut_gpu = cp.asarray(zlut)
                args_gpu = (stack_gpu, zlut_gpu)
            result_gpu = cupy_benchmark(func, args=args_gpu, kwargs=kwargs, n_repeat=N_REPEAT, n_warmup=10)
            mean_gpu = float(np.nanmean(result_gpu.gpu_times))
            sd_gpu = float(np.nanstd(result_gpu.gpu_times))

            records.append({
                "function": func_name,
                "size": size,
                "n_images": n_img,
                "processor": "CPU",
                "mean": mean_cpu,
                "sd": sd_cpu,
                "n_repeat": N_REPEAT,
            })

            records.append({
                "function": func_name,
                "size": size,
                "n_images": n_img,
                "processor": "GPU",
                "mean": mean_gpu,
                "sd": sd_gpu,
                "n_repeat": N_REPEAT,
            })

print(f'Benchmarking took: {time() - start_time:.2f}s')

# ==========================================================
#  Save results and display
# ==========================================================
df = pd.DataFrame.from_records(records)
df.to_csv("benchmark_results.csv", index=False)
print("\nBenchmark summary:")
print(df.head())