import cupy as cp
import numpy as np


def split_gpu_apply(stack, n, func, *args, **kwargs):
    n_images = stack.shape[2]
    n_splits = n_images // n
    n_mod = n_images % n

    # First split
    gpu_substack = cp.asarray(stack[:, :, 0:n]).astype('float32')
    gpu_args = []
    for arg in args:
        gpu_args.append(cp.asarray(arg[0:n]))
    results = list(func(gpu_substack, *gpu_args, **kwargs))
    for i in range(len(results)):
        results[i] = cp.asnumpy(results[i])

    # Middle splits
    for s in range(1, n_splits):
        gpu_substack = cp.asarray(stack[:, :, (s * n):((s + 1) * n)]
                                  ).astype('float32')
        gpu_args = []
        for arg in args:
            gpu_args.append(cp.asarray(arg[(s * n):((s + 1) * n)]))
        sub_results = func(gpu_substack, *gpu_args, **kwargs)
        for i in range(len(results)):
            results[i] = np.append(results[i], cp.asnumpy(sub_results[i]))

    # Last split
    if n_mod > 0:
        gpu_substack = cp.asarray(stack[:, :,
                                        (n_splits * n):]).astype('float32')
        gpu_args = []
        for arg in args:
            gpu_args.append(cp.asarray(arg[(n_splits * n):]))
        sub_results = func(gpu_substack, *gpu_args, **kwargs)
        for i in range(len(results)):
            results[i] = np.append(results[i], cp.asnumpy(sub_results[i]))

    return results
