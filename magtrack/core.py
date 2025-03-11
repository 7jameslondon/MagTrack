from typing import Any

import cupy as cp
import cupyx.scipy.signal

FLOAT = cp.float32
INT = cp.int32

def binmean(
        x: cp.ndarray[tuple[int, int], INT],
        weights: cp.ndarray[tuple[int, int], FLOAT],
        n_bins: int,
) -> cp.ndarray[tuple[int, int], FLOAT]:
    """
    Similar to numpy.bincount but for mean of 2D arrays

    Parameters
    ----------
    x : 2D int array, shape (n_values, n_datasets)
        Input array
    weights : optional, 2D float array, shape (n_values, n_datasets)
        Weights, same shape as x
    n_bins : int
        The number of bins to be used. Values will be binned as integers
        between 0 and n_bins.

    Returns
    ----------
    bin_means : 2D float array, shape (n_bins, n_datasets)
        Binned average values of weights.
    """
    n_datasets = x.shape[1]

    # GPU or CPU?
    xp = cp.get_array_module(x)

    # Clip the maximum x value to nbins (we will discard them latter)
    xp.minimum(x, n_bins, out=x)

    # Create an index to keep track of each row/dataset of x
    i_base = xp.arange(x.shape[1], dtype=cp.uint16)
    i = xp.broadcast_to(i_base, x.shape)

    # Binning
    bin_means = xp.zeros((n_bins + 1, n_datasets), dtype=FLOAT)  # Pre-allocate
    xp.add.at(bin_means, (x, i), weights)

    bin_counts = xp.zeros((n_bins + 1, n_datasets), dtype=FLOAT)  # Pre-allocate
    xp.add.at(bin_counts, (x, i), 1)

    bin_means /= bin_counts

    return bin_means[:-1, :]  # Return without the overflow row

def crop_stack_to_rois(
        stack: cp.ndarray[tuple[int, int, int], Any],
        rois: cp.ndarray[tuple[int, 4], cp.integer]
) -> cp.ndarray[tuple[int, int, int, int], Any]:
    """
    Takes a 3D image-stack and crops it to the region of interests (ROIs).

    Given a 3D image-stack and a list of ROIs, this function will crop around
    each ROI and return a 4D array. Note the input stack must contain square
    images and the ROIs must be squares. This can run on both CPU and GPU. The
    function will use the GPU if the stack is on the GPU. Otherwise, the CPU.
    However, it is recommended to use the CPU and then transfer the result to
    the GPU and perform downstream analysis on the GPU.

    Parameters
    ----------
    stack : 3D array of any type, shape (width, width, n_images)
        Note the images must be square.
    rois : 2D int array, shape (n_roi, 4)
        Each row is an ROI. The columns are [left, right, top, bottom].

    Returns
    ----------
    cropped_stack : 4D array, shape (width, width, n_images, n_roi)
        Same type as input stack
    """
    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Pre-allocate space for cropped stack
    n_images = stack.shape[2]
    n_rois = rois.shape[0]
    width = rois[0,1] - rois[0,0]
    shape = (width, width, n_images, n_rois)
    cropped_stack = xp.ndarray(shape, dtype=stack.dtype)  # width, width, frames, rois

    # Crop
    for i in range(n_rois):
        cropped_stack[:, :, :, i] = (
            stack[rois[i,0]:rois[i,1], rois[i,2]:rois[i,3], :])

    return cropped_stack

def parabolic_vertex(
        data: cp.ndarray[tuple[int, int], FLOAT],
        vertex_est: cp.ndarray[int, FLOAT],
        n_local: int
) -> cp.ndarray[int, FLOAT]:
    """
    Refines a local min/max by parabolic interpolation.

    Given an estimated location of a local min/max, this function will find a
    more precise location of the vertex by fitting the local datapoints to a
    parabola and interpolating the vertex. The code is agnostic of CPU and GPU
    usage. If data is on the GPU the computation and result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    data : 2D cupy float array, shape (n_datasets, n_datapoints)
        The data to be fit where each row is a dataset and columns are the data
        points.
    vertex_est : 1D cupy float, shape (n_datasets)
        The estimated location of the vertex.
    n_local : int
        The number of local datapoints to be fit. Must be an odd int >=3.

    Returns
    ----------
    vertex : 1D cupy float array
        The precise location of the vertex.
    """

    # GPU or CPU?
    xp = cp.get_array_module(data)

    # Setup
    n_local_half = (n_local // 2)

    # Convert the estimated vertex to an int for use as an index
    vertex_int = vertex_est.round().astype(INT)

    # Force index to be with the limits
    index_min = n_local_half
    index_max = data.shape[1] - n_local_half - 1
    xp.clip(vertex_int, index_min, index_max, out=vertex_int)

    # Get the local data to be fit
    n_datasets = data.shape[0]
    rel_idx = xp.arange(-n_local_half, n_local_half + 1, dtype=INT)
    idx = rel_idx + vertex_int[:, xp.newaxis]
    y = data[xp.arange(n_datasets)[:, xp.newaxis], idx].T
    x = xp.arange(n_local, dtype=FLOAT)

    # Fit to parabola
    p = xp.polyfit(x, y, 2)

    # Calculate the location of the vertex
    vertex = -p[1, :] / (2. * p[0, :]) + vertex_int - n_local // 2.  # -b/2a

    # Exclude points outside limits
    vertex[vertex_int == index_min] = xp.nan
    vertex[vertex_int == index_max] = xp.nan

    return vertex

def center_of_mass(
        stack: cp.ndarray[tuple[int, int, int], FLOAT]
) -> tuple[cp.ndarray[int, FLOAT], cp.ndarray[int, FLOAT]]:
    """
    Calculates the center-of-mass of each 2D image from a 3D image-stack.

    For each 2D image of a 3D image-stack: the mean background is
    subtracted, the absolute value is taken and then the center-of-mass in
    along the x- and y-axis is calculated. The code is agnostic of CPU and
    GPU usage. If the stack is on the GPU the computation and result will
    be on the GPU. Otherwise, the computation/result will be on the CPU.
    Benchmarking show this function is faster than the version from scipy
    or cupyx.scipy

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. Note, the images must be square.

    Returns
    ----------
    x : 1D float array, shape (n_images,)
        The x coordinates of the center of mass.
    y : 1D float array, shape (n_images,)
        The y coordinates of the center of mass.
    """

    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Correct background of each image
    stack_norm = stack.copy()
    xp.subtract(stack_norm, xp.mean(stack, axis=(0, 1)), out=stack_norm)
    xp.absolute(stack_norm, out=stack_norm)

    # Calculate center-of-mass
    index = xp.arange(stack_norm.shape[0], dtype=FLOAT)[:, xp.newaxis]
    total_mass = xp.sum(stack_norm, axis=(0, 1))
    x = xp.sum(index * xp.sum(stack_norm, axis=0), axis=0) / total_mass
    y = xp.sum(index * xp.sum(stack_norm, axis=1), axis=0) / total_mass

    return x, y

# TODO: Add documentation
def auto_conv(stack, x_old, y_old):
    # GPU or CPU?
    xp = cp.get_array_module(stack)
    xpx = cupyx.scipy.get_array_module(stack)

    # Get the "signal" from the row and column of the last x & y
    index = xp.arange(stack.shape[2], dtype=INT)
    x_arr = stack[:, xp.round(x_old).astype(INT), index]
    y_arr = stack[xp.round(y_old).astype(INT), :, index]

    # Subtract mean from signals
    x_arr -= xp.mean(x_arr, axis=0, keepdims=True)
    y_arr -= xp.mean(y_arr, axis=1, keepdims=True)

    # Convolve the signals
    x_con = xpx.signal.fftconvolve(x_arr, x_arr, 'same', axes=0)
    y_con = xpx.signal.fftconvolve(y_arr, y_arr, 'same', axes=1)

    # Find the convolution maxima
    y_con_max = xp.argmax(y_con, axis=1)
    x_con_max = xp.argmax(x_con, axis=0)

    # Use the maximum of the convolution to find center of the beads
    radius = (stack.shape[0] - 1) // 2
    x = radius - ((radius - y_con_max) / 2)
    y = radius - ((radius - x_con_max) / 2)

    return (x, y), (x_con_max, y_con_max, x_con, y_con)

# TODO: Add documentation
def auto_conv_para_fit(stack, x_old, y_old):
    _, (x_con_max, y_con_max, x_con, y_con) = auto_conv(stack, x_old, y_old)

    x = parabolic_vertex(y_con, y_con_max, 11)
    y = parabolic_vertex(x_con.T, x_con_max, 11)

    radius = (stack.shape[0] - 1) // 2
    x = radius - ((radius - x) / 2)
    y = radius - ((radius - y) / 2)

    return x, y

# TODO: Add documentation
def radial_profile(
        stack: cp.ndarray[tuple[int, int, int], FLOAT],
        x: cp.ndarray[int, FLOAT],
        y: cp.ndarray[int, FLOAT]
) -> cp.ndarray[tuple[int, int], FLOAT]:

    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Setup
    width = stack.shape[0]
    n_images = stack.shape[2]
    n_bins = stack.shape[0] // 4
    grid = xp.indices((width, width), dtype=FLOAT)
    flat_stack = stack.reshape((width*width, n_images))  # flatten spatial dimensions

    # Calculate distance of each pixel from x and y
    r = xp.hypot(
        grid[1][:, :, xp.newaxis] - x,
        grid[0][:, :, xp.newaxis] - y
    ).reshape(-1, n_images).round().astype(cp.uint16)

    # Calculate profile by summing
    profiles = binmean(r, flat_stack, n_bins)

    return profiles

# TODO: Add documentation
def lookup_z_para_fit(
        profiles: cp.ndarray[tuple[int, int], FLOAT],
        zlut: cp.ndarray[tuple[int, int], FLOAT]
) -> cp.ndarray[int, FLOAT]:
    ref_z = zlut[0, :]
    ref_profiles = zlut[1:, :]
    n_images = profiles.shape[1]
    n_ref = ref_profiles.shape[1]

    # Calculate the total difference between Z-LUT and current profiles
    dif = cp.empty((n_images, n_ref), dtype=FLOAT)
    for i in range(n_images):
        dif[i, :] = cp.sum(cp.abs(ref_profiles - profiles[:, i:i+1]), axis=0)

    # Find index of the minimum difference
    z_int_idx = cp.argmin(dif, axis=1).astype(FLOAT)

    # Find the sub-planar index of the minimum difference
    z_idx = parabolic_vertex(dif, z_int_idx, 3)

    # Interpolate z from reference index
    z = cp.interp(z_idx, cp.arange(n_ref), ref_z, left=cp.nan, right=cp.nan)

    return z

# TODO: Add documentation
def stack_to_xyzp(stack, zlut):
    g_stack = cp.asarray(stack, dtype=FLOAT)

    x, y = center_of_mass(g_stack)

    for _ in range(1):
        (x, y), _ = auto_conv(g_stack, x, y)

    for _ in range(1):
        x, y = auto_conv_para_fit(g_stack, x, y)

    # If this needs to be added it should be set to cp.nan
    # width = stack.shape[0]
    # cp.clip(x, 0, width, out=x)
    # cp.clip(y, 0, width, out=y)

    profiles = radial_profile(g_stack, x, y)

    if zlut is None:
        z = x*cp.nan
    else:
        z = lookup_z_para_fit(profiles, zlut)

    return cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z), cp.asnumpy(profiles)