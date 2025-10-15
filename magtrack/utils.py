import cupy as cp
import numpy as np
import scipy as sp
import os
import re

def check_cupy():
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except: # noqa E722
        return False

def split_gpu_apply(stack, n, func, splitargs, fullargs, **kwargs):
    n_images = stack.shape[2]
    n_splits = n_images // n
    n_mod = n_images % n

    full_gpu_args = []
    for arg in fullargs:
        full_gpu_args.append(cp.asarray(arg))

    # First split
    gpu_substack = cp.asarray(stack[:, :, 0:n]).astype('float64')
    gpu_args = full_gpu_args.copy()
    for arg in splitargs:
        gpu_args.append(cp.asarray(arg[0:n]))
    results = func(gpu_substack, *gpu_args, **kwargs)
    if not isinstance(results, tuple):
        results = (results, )
    results = list(results)
    for i in range(len(results)):
        results[i] = cp.asnumpy(results[i])

    # Middle splits
    for s in range(1, n_splits):
        gpu_substack = cp.asarray(stack[:, :, (s * n):((s + 1) * n)]
                                  ).astype('float64')
        gpu_args = full_gpu_args.copy()
        for arg in splitargs:
            gpu_args.append(cp.asarray(arg[(s * n):((s + 1) * n)]))
        sub_results = func(gpu_substack, *gpu_args, **kwargs)
        if not isinstance(sub_results, tuple):
            sub_results = (sub_results, )
        for i in range(len(results)):
            results[i] = np.concatenate(
                (results[i], cp.asnumpy(sub_results[i])), axis=-1
            )

    # Last split
    if n_mod > 0:
        gpu_substack = cp.asarray(stack[:, :,
                                        (n_splits * n):]).astype('float64')
        gpu_args = full_gpu_args.copy()
        for arg in splitargs:
            gpu_args.append(cp.asarray(arg[(n_splits * n):]))
        sub_results = func(gpu_substack, *gpu_args, **kwargs)
        if not isinstance(sub_results, tuple):
            sub_results = (sub_results, )
        for i in range(len(results)):
            results[i] = np.concatenate(
                (results[i], cp.asnumpy(sub_results[i])), axis=-1
            )

    if len(results) == 1:
        return results[0]
    else:
        return results


def airy_disk(size=512, radius=50, wavelength=1.0):
    """ Generate an Airy disk pattern """
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    r = np.where(r == 0, 1e-10, r)  # Avoid division by zero

    k = 2 * np.pi / wavelength
    kr = k * r / radius
    intensity = (2 * sp.special.j1(kr) / kr)**0.8

    intensity[r >= radius * 4] = 0

    return intensity

def join_videos(dir_path, pattern=None, output_filename=None):
    """ Join TIFF files from a directory into a single video file

    Args:
        dir_path: Path to directory containing TIFF files
        output_filename: Name of output video to be saved in directory file. If None, no file is saved.

    Returns:
        joined video: Numpy array containing the joined video frames
    """

    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required to use this function. Please install it via pip or conda.")

    filenames = os.listdir(dir_path)
    filenames = [f for f in filenames if f.lower().endswith(('.tif', '.tiff'))]
    if pattern is not None:
        pattern = re.compile(pattern)
        filenames = [f for f in filenames if pattern.match(f)]
    filenames = sorted(filenames)

    if not filenames:
        raise ValueError("No TIFF files found in the specified directory")

    # Read all TIFF files and store them in a list
    video_frames = []
    for filename in filenames:
        filepath = os.path.join(dir_path, filename)
        frames = tifffile.imread(filepath)
        if frames.ndim == 2:
            frames = frames[np.newaxis, ...]
        video_frames.append(frames)

    # Concatenate all frames
    joined_video = np.concatenate(video_frames, axis=0)

    # Save the joined video
    if output_filename is not None:
        tifffile.imwrite(os.path.join(dir_path, output_filename), joined_video)

    return joined_video
