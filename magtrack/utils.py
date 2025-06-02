import cupy as cp
import numpy as np
import scipy as sp


def split_gpu_apply(stack, n, func, *args, **kwargs):
    n_images = stack.shape[2]
    n_splits = n_images // n
    n_mod = n_images % n

    # First split
    gpu_substack = cp.asarray(stack[:, :, 0:n]).astype('float64')
    gpu_args = []
    for arg in args:
        gpu_args.append(cp.asarray(arg[0:n]))
    results = func(gpu_substack, *gpu_args, **kwargs)
    if not isinstance(results, tuple):
        results = (results,)
    results = list(results)
    for i in range(len(results)):
        results[i] = cp.asnumpy(results[i])

    # Middle splits
    for s in range(1, n_splits):
        gpu_substack = cp.asarray(stack[:, :, (s * n):((s + 1) * n)]
                                  ).astype('float64')
        gpu_args = []
        for arg in args:
            gpu_args.append(cp.asarray(arg[(s * n):((s + 1) * n)]))
        sub_results = func(gpu_substack, *gpu_args, **kwargs)
        if not isinstance(sub_results, tuple):
            sub_results = (sub_results,)
        for i in range(len(results)):
            results[i] = np.concatenate((results[i], cp.asnumpy(sub_results[i])), axis=-1)

    # Last split
    if n_mod > 0:
        gpu_substack = cp.asarray(stack[:, :, (n_splits * n):]).astype('float64')
        gpu_args = []
        for arg in args:
            gpu_args.append(cp.asarray(arg[(n_splits * n):]))
        sub_results = func(gpu_substack, *gpu_args, **kwargs)
        if not isinstance(sub_results, tuple):
            sub_results = (sub_results,)
        for i in range(len(results)):
            results[i] = np.concatenate((results[i], cp.asnumpy(sub_results[i])), axis=-1)

    if len(results) == 1:
        return results[0]
    else:
        return results

def airy_disk(size=512, radius=50, wavelength=1.0):
    """ Generate an Airy disk pattern """
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    r = np.where(r == 0, 1e-10, r)  # Avoid division by zero

    k = 2 * np.pi / wavelength
    kr = k * r / radius
    intensity = (2 * sp.special.j1(kr) / kr) ** 0.8

    intensity[r >= radius*4] = 0

    return intensity