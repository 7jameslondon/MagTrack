import warnings
import numpy as np

from ._cupy import cp, cupyx, is_cupy_available

if is_cupy_available():
    try:  # pragma: no cover - exercised when CuPy is installed without SciPy extras
        import cupyx.scipy.signal  # type: ignore
        import cupyx.scipy.ndimage  # type: ignore
    except ImportError:
        warnings.warn(
            "GPU-acceleration with CuPy SciPy extensions is unavailable. "
            "Falling back to CPU implementations."
        )
else:
    warnings.warn(
        "GPU-acceleration with CuPy is not available. Will use CPU only."
    )

np.seterr(divide='ignore', invalid='ignore')

# ---------- Helper functions ---------- #

def binmean(x, weights, n_bins: int):
    """Compute mean values per bin for 2D arrays, similar to ``numpy.bincount``.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    The input ``x`` is clipped *in place* so that values above ``n_bins`` fall
    back within the valid bin range; entries clipped to ``n_bins`` are ignored
    when returning the binned means.

    Parameters
    ----------
    x : 2D int array, shape (n_values, n_datasets)
        Input array to bin.
    weights : 2D float array, shape (n_values, n_datasets)
        Weights associated with ``x``; should be floating point to allow
        averaging.
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
    i_base = xp.arange(x.shape[1], dtype=xp.min_scalar_type(x.shape[1]))
    i = xp.broadcast_to(i_base, x.shape)

    # Binning
    bin_means = xp.zeros(
        (n_bins + 1, n_datasets), dtype=weights.dtype
    )  # Pre-allocate
    xp.add.at(bin_means, (x, i), weights)

    bin_counts = xp.zeros(
        (n_bins + 1, n_datasets), dtype=xp.uint32
    )  # Pre-allocate
    xp.add.at(bin_counts, (x, i), 1)

    bin_means /= bin_counts

    # Return without the overflow row
    return bin_means[:-1, :]


def pearson(x, y):
    """
    Calculate the Pearson correlation coefficient between each row of x and y.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    x : array, shape (n, m)
        2D array whose columns are correlated with the columns of ``y``.
    y : array, shape (n, k)
        2D array whose columns are correlated with the columns of ``x``.

    Returns
    -------
    r : array, shape (k, m)
        Pearson correlation coefficients between each column of ``y`` and each
        column of ``x``.
    """

    # GPU or CPU?
    xp = cp.get_array_module(x)

    X = x - xp.nanmean(x, axis=0, keepdims=True)
    Y = y - xp.nanmean(y, axis=0, keepdims=True)
    X = xp.nan_to_num(X, copy=False)
    Y = xp.nan_to_num(Y, copy=False)
    sx = xp.sqrt((X * X).sum(axis=0))  # (m,)
    sy = xp.sqrt((Y * Y).sum(axis=0))  # (k,)
    num = Y.T @ X  # (k,m)
    den = sy[:, None] * sx[None, :]  # (k,m)

    r = num / den

    return r


def gaussian(x, mu, sigma):
    """
    Calculate a 1D Gaussian function.

    Parameters
    ----------
    x : array_like
        x coordinates where to evaluate the gaussian
    mu : float
        Mean (center)
    sigma : float
        Standard deviation

    Returns
    -------
    array_like
        1D array containing the gaussian evaluated at x coordinates
    """

    # GPU or CPU?
    xp = cp.get_array_module(x)

    return xp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def gaussian_2d(x, y, mu_x, mu_y, sigma):
    """
    Calculate a 2D Gaussian image.

    Calculates a 2D Gaussian image for each center (mu_x, mu_y) provided along
    the grid (x, y) all sharing the same sigma in x and y (sigma).
    across images.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    x : 1D array
        x coordinates where to evaluate the gaussian
    y : 1D array
        y coordinates where to evaluate the gaussian
    mu_x : 1D array
        Mean (center) in x direction for each image (one center per image)
    mu_y : 1D array
        Mean (center) in y direction for each image (one center per image)
    sigma : float
        Standard deviation in x and y direction

    Returns
    -------
    array
        3D array of shape ``(len(x), len(y), n_images)`` containing the gaussian
        evaluated at ``(x, y)`` coordinates for each image center
    """
    # GPU or CPU?
    xp = cp.get_array_module(x)

    return xp.exp(-((x[:, xp.newaxis, xp.newaxis] - mu_x[xp.newaxis, xp.newaxis, :]) ** 2 / (2 * sigma ** 2) +
                    (y[xp.newaxis, :, xp.newaxis] - mu_y[xp.newaxis, xp.newaxis, :]) ** 2 / (2 * sigma ** 2)))


def crop_stack_to_rois(stack, rois):
    """
    Takes a 3D image-stack and crops it to the region of interests (ROIs).

    Given a 3D image-stack and a list of ROIs, this function will crop around
    each ROI and return a 4D array. Note the ROIs must be squares.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU. However, it is
    recommended to use the CPU and then transfer the result to the GPU and
    perform downstream analysis on the GPU.

    Parameters
    ----------
    stack : 3D ndarray of any type, shape (stack_width, stack_height, n_images)
        Note the images must be square.
    rois : 2D int ndarray, shape (n_roi, 4)
        Each row is an ROI. The columns are [left, right, top, bottom].

    Returns
    ----------
    cropped_stack : 4D ndarray, shape (cropped_width, cropped_width, n_images, n_roi)
        Same type as input stack
    """
    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Pre-allocate space for cropped stack
    n_images = stack.shape[2]
    n_rois = rois.shape[0]
    width = rois[0, 1] - rois[0, 0]
    shape = (width, width, n_images, n_rois)
    cropped_stack = xp.ndarray(
        shape, dtype=stack.dtype
    )  # width, width, frames, rois

    # Crop
    for i in range(n_rois):
        cropped_stack[:, :, :, i] = (
            stack[rois[i, 0]:rois[i, 1], rois[i, 2]:rois[i, 3], :]
        )

    return cropped_stack


def parabolic_vertex(data, vertex_est, n_local: int, weighted=True):
    """Refine local min/max using parabolic interpolation.

    Given an estimated location of a local minimum or maximum, this function
    fits the surrounding datapoints to a parabola and interpolates the vertex.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    data : array of float, shape (n_datasets, n_datapoints)
        Sequence of datasets arranged row-wise for fitting.
    vertex_est : array of float, shape (n_datasets,)
        Initial vertex estimates corresponding to each dataset.
    n_local : int
        The number of local datapoints to be fit. Must be an odd integer >= 3.
    weighted : bool, optional
        Whether to apply a simple weighting procedure to emphasize the more
        central points in the fit. Default is True.

    Returns
    -------
    vertex : array of float, shape (n_datasets,)
        Refined vertex locations
    """

    # GPU or CPU?
    xp = cp.get_array_module(data)

    # Setup
    n_local_half = (n_local // 2)

    # Convert the estimated vertex to an int for use as an index
    vertex_int = vertex_est.round().astype(xp.int64)

    # Force index to be with the limits
    index_min = n_local_half
    index_max = data.shape[1] - n_local_half - 1
    xp.clip(vertex_int, index_min, index_max, out=vertex_int)

    # Get the local data to be fit
    n_datasets = data.shape[0]
    rel_idx = xp.arange(-n_local_half, n_local_half + 1, dtype=xp.int64)
    idx = rel_idx + vertex_int[:, xp.newaxis]
    y = data[xp.arange(n_datasets)[:, xp.newaxis], idx].T
    x = xp.arange(n_local, dtype=xp.float64)

    # Fit to parabola
    if weighted:
        w = n_local_half - xp.abs(xp.arange(n_local) - n_local_half) + 1
        p = xp.polyfit(x, y, 2, w=w)
    else:
        p = xp.polyfit(x, y, 2)

    # Calculate the location of the vertex
    vertex = -p[1, :] / (2. * p[0, :]) + vertex_int - n_local // 2.  # -b/2a

    # Exclude points outside limits
    vertex[vertex_int == index_min] = xp.nan
    vertex[vertex_int == index_max] = xp.nan

    return vertex

# ---------- QI functions ---------- #

def _qi_sample_axis_profiles(stack, x, y, axis):
    """Sample 1D profiles along ``axis`` using quadratic interpolation support.

    Parameters
    ----------
    stack : 3D array_like
        Image stack where the first two axes correspond to ``y`` and ``x`` and
        the third axis indexes frames.
    x, y : 1D array_like
        Approximate center coordinates for each frame. They must have the same
        length as the number of frames in ``stack``.
    axis : int
        Axis along which to collect the three-point profile. ``0`` samples the
        column profile (varying ``y``); ``1`` samples the row profile (varying
        ``x``).

    Returns
    -------
    array_like
        Samples of shape ``(n_frames, 3)`` corresponding to offsets of ``-1``,
        ``0`` and ``+1`` pixels along the chosen axis.
    """

    xp = cp.get_array_module(stack)
    xpx = cupyx.scipy.get_array_module(stack)

    stack_xp = stack
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)

    n_frames = stack.shape[2]
    offsets = xp.array([-1.0, 0.0, 1.0], dtype=xp.float64)

    frame_coords = xp.arange(n_frames, dtype=xp.float64)[:, xp.newaxis]
    frame_coords = xp.broadcast_to(frame_coords, (n_frames, offsets.size))

    if axis == 0:
        primary = (y[:, xp.newaxis] + offsets[xp.newaxis, :])
        secondary = xp.broadcast_to(x[:, xp.newaxis], primary.shape)
    elif axis == 1:
        primary = xp.broadcast_to(y[:, xp.newaxis], (n_frames, offsets.size))
        secondary = (x[:, xp.newaxis] + offsets[xp.newaxis, :])
    else:
        raise ValueError("axis must be 0 (column) or 1 (row)")

    coords = xp.stack(
        (
            primary.reshape(-1),
            secondary.reshape(-1),
            frame_coords.reshape(-1),
        ),
        axis=0,
    )

    samples = xpx.ndimage.map_coordinates(
        stack_xp,
        coords,
        order=2,
        mode="nearest",
    )

    return samples.reshape(n_frames, offsets.size)


def _qi_quadratic_offsets(samples):
    """Compute quadratic-interpolation offsets from three-point samples."""

    xp = cp.get_array_module(samples)
    values = xp.asarray(samples, dtype=xp.float64)

    left = values[:, 0]
    center = values[:, 1]
    right = values[:, 2]

    denom = left - 2.0 * center + right
    eps = xp.finfo(values.dtype).eps
    valid = xp.abs(denom) > eps

    offsets = xp.full(left.shape, xp.nan, dtype=values.dtype)
    offsets = xp.where(
        valid,
        0.5 * (left - right) / denom,
        offsets,
    )

    return offsets


def qi(stack, x_old, y_old):
    """Refine centers using quadratic interpolation along x and y axes.

    This routine samples the intensity profiles along the horizontal and
    vertical axes through the supplied center estimates and performs quadratic
    interpolation to recover sub-pixel offsets.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU. Intermediate values
    remain on the originating device and the function respects the caller's
    backend.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        Image stack containing square frames to refine.
    x_old : 1D float array, shape (n_images)
        Initial estimates of the x coordinates.
    y_old : 1D float array, shape (n_images)
        Initial estimates of the y coordinates.

    Returns
    -------
    tuple of array_like
        Refined ``(x, y)`` coordinates with sub-pixel precision.
    """

    xp = cp.get_array_module(stack)
    x_old = xp.asarray(x_old, dtype=xp.float64)
    y_old = xp.asarray(y_old, dtype=xp.float64)

    row_samples = _qi_sample_axis_profiles(stack, x_old, y_old, axis=1)
    col_samples = _qi_sample_axis_profiles(stack, x_old, y_old, axis=0)

    dx = _qi_quadratic_offsets(row_samples)
    dy = _qi_quadratic_offsets(col_samples)

    return x_old + dx, y_old + dy

# ---------- Center-of-Mass functions ---------- #

def center_of_mass(stack, background='none'):
    """
    Calculate x and y by center-of-mass

    For each 2D image of a 3D image-stack compute the center-of-mass along the
    x- and y-axes. To avoid bias from the images' background, a pre-processing
    step can be taken to remove the background with the ``background`` keyword
    argument. The default, ``background='none'`` leaves the data unchanged.
    When ``background='mean'`` the per-frame mean is subtracted before
    calculating the center-of-mass. Likewise, ``background='median'``
    subtracts the per-frame median. This function is faster than the version
    from ``scipy`` or ``cupyx.scipy``.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. The images must be square.
    background : str, optional
        Background pre-processing. ``'none'`` (default) uses the raw data.
        ``'mean'`` subtracts the per-image mean. ``'median'`` subtracts the
        per-image median.

    Returns
    ----------
    x : 1D float array, shape (n_images,)
        The x coordinates of the center
    y : 1D float array, shape (n_images,)
        The y coordinates of the center
    """

    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Correct background of each image
    if background == 'none':
        stack_norm = stack.view()
    elif background == 'mean':
        stack_norm = stack.copy()
        xp.subtract(stack_norm, xp.mean(stack, axis=(0, 1)), out=stack_norm)
        xp.absolute(stack_norm, out=stack_norm)
    elif background == 'median':
        stack_norm = stack.copy()
        xp.subtract(stack_norm, xp.median(stack, axis=(0, 1)), out=stack_norm)
        xp.absolute(stack_norm, out=stack_norm)
    else:
        raise NotImplementedError

    # Calculate total
    total_mass = xp.sum(stack_norm, axis=(0, 1))
    # Prevent divide by zero
    total_mass = xp.where(total_mass == 0., xp.nan, total_mass)

    # Calculate center
    index = xp.arange(stack_norm.shape[0], dtype=xp.float64)[:, xp.newaxis]
    x = xp.sum(index * xp.sum(stack_norm, axis=0), axis=0) / total_mass
    y = xp.sum(index * xp.sum(stack_norm, axis=1), axis=0) / total_mass

    return x, y

# ---------- Auto-convolution functions ---------- #

def auto_conv(stack, x_old, y_old, return_conv=False):
    """
    Re-calculate center of symmetric object by auto-convolution

    For each 2D image of a 3D image-stack: use the previous center to select
    the central row and column. Convolve these against reversed versions of
    themselves (auto-convolution). Then take the maximum as the new center.
    Optionally, by setting return_conv to True the convolution results can be
    returned directly. This is useful for sub-pixel fitting.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. The images must be square.
    x_old : 1D float array, shape (n_images)
        Estimated x coordinates near the true centers.
    y_old : 1D float array, shape (n_images)
        Estimated y coordinates near the true centers.
    return_conv : bool, optional
        Whether to return the convolution or return the new center.
        The default is False.

    Returns
    ----------
    tuple
        see information below
    If return_conv is False:
        x : 1D float array, shape (n_images,)
            The x coordinates of the center
        y : 1D float array, shape (n_images,)
            The y coordinates of the center
    If return_conv is True:
        col_max : 1D float array, shape (n_images,)
            The index of the maximum of the column convolution
        row_max : 1D float array, shape (n_images,)
            The index of the maximum of the row convolution
        col_con : 2D float array, shape (n_pixels, n_images)
            The column convolution
        row_con : 2D float array, shape (n_pixels, n_images)
            The row convolution
    """
    # GPU or CPU?
    xp = cp.get_array_module(stack)
    xpx = cupyx.scipy.get_array_module(stack)

    # Get the row and column slices corresponding to the previous x & y
    frame_idx = xp.arange(stack.shape[2], dtype=xp.int64)
    x_idx = xp.round(x_old).astype(xp.int64)
    y_idx = xp.round(y_old).astype(xp.int64)
    cols = stack[:, x_idx, frame_idx]
    rows = stack[y_idx, :, frame_idx]

    # Subtract means
    cols -= xp.mean(cols, axis=0, keepdims=True)
    rows -= xp.mean(rows, axis=1, keepdims=True)

    # Apply gaussian weights to reduce edge effects
    width = stack.shape[0]
    px_idx = xp.arange(width)
    cols *= gaussian(px_idx[:, xp.newaxis], y_old[xp.newaxis, :], width/6.)
    rows *= gaussian(px_idx[xp.newaxis, :], x_old[:, xp.newaxis], width/6.)

    # Convolve the signals
    col_con = xpx.signal.fftconvolve(cols, cols, 'same', axes=0)
    row_con = xpx.signal.fftconvolve(rows, rows, 'same', axes=1)

    # Find the convolution maxima
    col_max = xp.argmax(col_con, axis=0)
    row_max = xp.argmax(row_con, axis=1)

    if return_conv:
        return col_max, row_max, col_con, row_con
    else:
        # Use the maximum of the convolution to find center of the beads
        radius = (stack.shape[0] - 1) // 2
        x = radius - ((radius - row_max) / 2)
        y = radius - ((radius - col_max) / 2)
        return x, y


def auto_conv_sub_pixel(stack, x_old, y_old, n_local=5):
    """
    Re-calculate center of symmetric object by auto-convolution sub-pixel fit

    For each 2D image of a 3D image-stack: use the previous center to select
    the central row and column. Convolve these against themselves. Use several
    points around the maximum of the convolution to fit a parabola and use the
    vertex of the parabola as the center to find the sub-pixel coordinates.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the
    parameters are on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. Note, the images must be square.
    x_old : 1D float array, shape (n_images)
        Estimated x coordinates near the true centers.
    y_old : 1D float array, shape (n_images)
        Estimated y coordinates near the true centers.
    n_local : int
        The number of local points around the vertex to be used in parabolic
        fitting. Must be an odd int >=3.

    Returns
    ----------
    x : 1D float array, shape (n_images,)
        The x coordinates of the center.
    y : 1D float array, shape (n_images,)
        The y coordinates of the center.
    """
    col_max, row_max, col_con, row_con = auto_conv(
        stack, x_old, y_old, return_conv=True
    )

    x = parabolic_vertex(row_con, row_max, n_local)
    y = parabolic_vertex(col_con.T, col_max, n_local)

    radius = (stack.shape[0] - 1) // 2
    x = radius - ((radius - x) / 2)
    y = radius - ((radius - y) / 2)

    return x, y


def auto_conv_multiline(stack, x_old, y_old, line_ratio=0.05, return_conv=False):
    """
    Re-calculate center of symmetric object by multi-line auto-convolution

    For each 2D image of a 3D image-stack: use the previous center to select
    the central row and column. Convolve these against reversed versions of
    themselves (auto-convolution). Then take the maximum as the new center.
    Optionally, by setting return_conv to True the convolution results can be
    returned directly. This is useful for sub-pixel fitting.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. The images must be square.
    x_old : 1D float array, shape (n_images)
        Estimated x coordinates near the true centers.
    y_old : 1D float array, shape (n_images)
        Estimated y coordinates near the true centers.
    line_ratio : float, optional
        Fraction of the frame width that determines how many neighbouring
        lines are averaged before convolution.
    return_conv : bool, optional
        Whether to return the convolution or return the new center.
        The default is False.

    Returns
    ----------
    tuple
        see information below
    If return_conv is False:
        x : 1D float array, shape (n_images,)
            The x coordinates of the center
        y : 1D float array, shape (n_images,)
            The y coordinates of the center
    If return_conv is True:
        col_max : 1D int array, shape (n_images,)
            The index of the maximum of the column convolution
        row_max : 1D int array, shape (n_images,)
            The index of the maximum of the row convolution
        col_con : 2D float array, shape (n_pixels, n_images)
            The column convolution
        row_con : 2D float array, shape (n_images, n_pixels)
            The row convolution
    """
    # GPU or CPU?
    xp = cp.get_array_module(stack)
    xpx = cupyx.scipy.get_array_module(stack)

    # Get the row and column slices corresponding to the previous x & y
    half_n_lines = int(stack.shape[0] * line_ratio // 2)
    n_lines = half_n_lines * 2 + 1
    line_idx = xp.arange(-half_n_lines, half_n_lines + 1)
    width = stack.shape[0]
    n_images = stack.shape[2]
    frame_idx = xp.arange(n_images, dtype=xp.int64)
    frame_idx = xp.repeat(frame_idx, n_lines)
    x_idx = xp.round(xp.nan_to_num(x_old)).astype(xp.int64)
    y_idx = xp.round(xp.nan_to_num(y_old)).astype(xp.int64)
    x_idx = x_idx[:, xp.newaxis] + line_idx
    y_idx = y_idx[:, xp.newaxis] + line_idx
    x_idx = x_idx.flatten()
    y_idx = y_idx.flatten()
    cols = stack[:, x_idx, frame_idx]
    rows = stack[y_idx, :, frame_idx]

    # Average multi-lines
    cols = cols.reshape(width, n_images, n_lines)
    rows = rows.reshape(n_images, n_lines, width)
    cols = xp.mean(cols, axis=2)
    rows = xp.mean(rows, axis=1)

    # Subtract means
    cols -= xp.mean(cols, axis=0, keepdims=True)
    rows -= xp.mean(rows, axis=1, keepdims=True)

    # Apply gaussian filter to reduce edge effects
    width = stack.shape[0]
    px_idx = xp.arange(width)
    cols *= gaussian(px_idx[:, xp.newaxis], y_old[xp.newaxis, :], width / 6.)
    rows *= gaussian(px_idx[xp.newaxis, :], x_old[:, xp.newaxis], width / 6.)

    # Convolve the signals
    col_con = xpx.signal.fftconvolve(cols, cols, 'same', axes=0)
    row_con = xpx.signal.fftconvolve(rows, rows, 'same', axes=1)

    # Find the convolution maxima
    col_max = xp.argmax(col_con, axis=0)
    row_max = xp.argmax(row_con, axis=1)

    if return_conv:
        return col_max, row_max, col_con, row_con
    else:
        # Use the maximum of the convolution to find center of the beads
        radius = (stack.shape[0] - 1) // 2
        x = radius - ((radius - row_max) / 2)
        y = radius - ((radius - col_max) / 2)
        return x, y


def auto_conv_multiline_sub_pixel(stack, x_old, y_old, line_ratio=0.1, n_local=5):
    """
    Re-calculate center of symmetric object by multi-line auto-convolution with sub-pixel fit

    For each 2D image of a 3D image-stack: use the previous center to select
    multiple rows and columns (determined by ``line_ratio``). Average the
    resulting signals, convolve them against themselves (auto-convolution) and
    use several points around the maximum to fit a parabola. The vertex of the
    parabola is used to determine the sub-pixel coordinates of the center.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. The images must be square.
    x_old : 1D float array, shape (n_images)
        Estimated x coordinates near the true centers.
    y_old : 1D float array, shape (n_images)
        Estimated y coordinates near the true centers.
    line_ratio : float, optional
        The ratio relative to the total image width of lines to be used in the
        convolutions.
    n_local : int, optional
        The number of local points around the vertex to be used in parabolic
        fitting. Must be an odd int >=3.

    Returns
    ----------
    x : 1D float array, shape (n_images,)
        The x coordinates of the center.
    y : 1D float array, shape (n_images,)
        The y coordinates of the center.
    """
    col_max, row_max, col_con, row_con = auto_conv_multiline(
        stack, x_old, y_old, return_conv=True, line_ratio=line_ratio
    )

    x = parabolic_vertex(row_con, row_max, n_local)
    y = parabolic_vertex(col_con.T, col_max, n_local)

    radius = (stack.shape[0] - 1) // 2
    x = radius - ((radius - x) / 2)
    y = radius - ((radius - y) / 2)

    return x, y

# ---------- Radial profile functions ---------- #

def radial_profile(stack, x, y, oversample=1):
    """
    Calculate the average radial profile about a center

    For each 2D image of a 3D image-stack: calculate the average radial profile
    about the corresponding center (x and y). The profile is calculated by
    binning. For each pixel in an image the Euclidean distance from the center
    is calculated. The distance is then used to bin each pixel. When
    ``oversample`` equals 1 the bin widths are 1 pixel wide; higher values split
    each native bin into finer ``1 / oversample`` pixel slices. The bins are
    then normalized by the number of pixels in each bin to find the average
    intensity in each bin. The number of bins (n_bins) is
    ((stack.shape[0] // 2) * oversample).

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. Note, the images must be square.
    x : 1D float array, shape (n_images)
        x-coordinates of the center.
    y : 1D float array, shape (n_images)
        y-coordinates of the center.
    oversample : int, optional
        Oversampling factor applied to the radial distances before binning.
        Increasing the factor multiplies the number of radial bins and thus the
        resolution of the profile by the same amount. Must be an integer
        greater than or equal to 1.

    Returns
    ----------
    profiles : 2D float array, shape (n_bins, n_images)
        The average radial profile of each image about the center
    """
    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Setup
    width = stack.shape[0]
    n_images = stack.shape[2]
    n_bins = (width // 2) * oversample
    grid = xp.indices((width, width), dtype=xp.float32)
    flat_stack = stack.reshape((width * width, n_images))

    # Calculate the distance of each pixel from x and y
    # cast to uint16 because min and max r for 1024x1024 would be 0 and 1449
    r = xp.round(xp.hypot(grid[1][:, :, xp.newaxis] - x, grid[0][:, :, xp.newaxis] - y) * oversample)
    r = r.astype(xp.uint16).reshape(-1, n_images)

    # Calculate profiles
    profiles = binmean(r, flat_stack, n_bins)

    return profiles


def fft_profile(stack, x, y, oversample=4, rmin=0.0, rmax=0.5, gaus_factor=6.):
    """
    oversample: int, >=1

    input images must be square and have an even width
    """

    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # Radial distance
    n_images = stack.shape[2]
    width = stack.shape[0]
    center = width // 2
    n_bins = int(round(center * rmax * oversample))
    n_start = int(round(center * rmin * oversample))
    grid = xp.indices((width, center+1), dtype=xp.float32)
    # cast to uint16 because min and max r for 1024x1024 would be 0 and 1449
    r_int = xp.round(xp.hypot(grid[1], grid[0] - center) * oversample).astype(xp.uint16).reshape(-1, 1)
    r = xp.tile(r_int, (1, n_images))

    # Gaussian weights
    w = gaussian_2d(xp.arange(width), xp.arange(width), x, y, width/gaus_factor)
    stack *= w

    # FFT
    fft_cpx = xp.fft.fftshift(xp.fft.rfft2(stack, axes=(0, 1)), axes=(0,))
    fft = xp.abs(fft_cpx).reshape(-1, n_images)

    # Profile
    profile = binmean(r, fft, n_bins)[n_start:]

    return profile

# ---------- Z-Lookup functions ---------- #

def lookup_z(profiles, zlut, n_local=5):
    """
    Calculate the corresponding sub-planar z-coordinate of each profile by LUT

    For each image's profile in ``profiles``: find the best matching profile in
    the Z-LUT (lookup table). The lookup table stores a leading row of
    z-coordinates and radial profiles below; the first radial bin corresponds to
    the central pixel and is ignored during correlation (hence ``zlut[2:, :]``).
    Fits the local points around the best matching profile to find sub-planar
    fit in between columns of the LUT.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    profiles : 2D float array, shape (n_bins, n_images)
        The average radial profile of each image about the center
    zlut : 2D float array, shape (1+n_bins, n_ref)
        The reference radial profiles and corresponding z-coordinates. The
        first row (``zlut[0, :]``) holds the z-axis values. The remaining rows
        contain the reference radial profiles produced by
        :func:`radial_profile`; their first bin (``zlut[1, :]``) is skipped to
        avoid the central pixel when matching.
    n_local : int, optional
        The number of local points around the vertex to be used in parabolic
        fitting. Must be an odd int >=3. Default is 5.

    Returns
    ----------
    z : 1D float array, shape (n_images)
        z-coordinates
    """
    # GPU or CPU?
    xp = cp.get_array_module(profiles)

    ref_z = zlut[0, :]
    ref_profiles = zlut[2:, :]  # Skip the first pixel
    n_ref = ref_profiles.shape[1]

    # Calculate the pearson correlation coefficient between Z-LUT and current profiles.
    # This (likely) needs to be done in a loop to prevent the
    # operation from taking too much memory at once.
    # Skip the first pixel
    r = pearson(ref_profiles, profiles[1:, :])

    # Find index of the max
    z_int_idx = xp.argmax(r, axis=1).astype(xp.float64)

    # Find the sub-planar index of the max
    z_idx = parabolic_vertex(r, z_int_idx, n_local)

    # Interpolate z from the reference index
    z = xp.interp(z_idx, xp.arange(n_ref), ref_z, left=xp.nan, right=xp.nan)

    return z


# ---------- Complete pipeline functions ---------- #

def stack_to_xyzp_advanced(stack, zlut=None, **kwargs):
    """
    Calculate image-stack XYZ and profiles (Z is None if Z-LUT is None)

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. Note, the images must be square. It is expected it is
        in the regular CPU memory. It will be transferred to the GPU.
    zlut : 2D float array, shape (1+n_bins, n_ref), optional
        The reference radial profiles and corresponding z-coordinates. The
        first row (zlut[0, :]) are the z-coordinates. The rest of the rows are
        the corresponding profiles as generated by radial_profile. It is
        expected it is already in the GPU memory. The defualt is None.
    **kwargs : dict, optional
        Additional keyword arguments controlling individual processing stages.
        The following keys are recognised:

        * ``"center_of_mass"`` (dict, default ``{}``): forwarded to
          :func:`center_of_mass`.
        * ``"auto_conv"`` (dict, default ``{}``): forwarded to :func:`auto_conv`.
        * ``"n auto_conv_multiline_sub_pixel"`` (int, default ``5``): number of
          :func:`auto_conv_multiline_sub_pixel` refinement iterations.
        * ``"auto_conv_multiline_sub_pixel"`` (dict, default ``{}``): forwarded
          to :func:`auto_conv_multiline_sub_pixel`.
        * ``"use fft_profile"`` (bool, default ``True``): when ``True`` compute
          profiles with :func:`fft_profile`; when ``False`` compute profiles
          with :func:`radial_profile`.
        * ``"fft_profile"`` (dict, default ``{}``): forwarded to
          :func:`fft_profile` when ``use fft_profile`` is ``True``.
        * ``"radial_profile"`` (dict, default ``{}``): forwarded to
          :func:`radial_profile` when ``use fft_profile`` is ``False``.
        * ``"lookup_z"`` (dict, default ``{}``): forwarded to :func:`lookup_z`
          when ``zlut`` is provided.

    Returns
    ----------
    x : 1D float array, shape (n_images)
        x-coordinates
    y : 1D float array, shape (n_images)
        y-coordinates
    z : 1D float array, shape (n_images)
        z-coordinates
    profiles : 2D float array, shape (n_bins, n_images)
        The average radial profile of each image about the center
    """
    # Move stack to GPU memory
    gpu_stack = cp.asarray(stack, dtype=cp.float64)

    x, y = center_of_mass(gpu_stack, **kwargs.get('center_of_mass', {}))

    x, y = auto_conv(gpu_stack, x, y, **kwargs.get('auto_conv', {}))

    for _ in range(kwargs.get('n auto_conv_multiline_sub_pixel', 5)):
        x, y = auto_conv_multiline_sub_pixel(
            gpu_stack, x, y, **kwargs.get('auto_conv_multiline_sub_pixel', {})
        )

    if kwargs.get('use fft_profile', True):
        profiles = fft_profile(gpu_stack, x, y, **kwargs.get('fft_profile', {}))
    else:
        profiles = radial_profile(gpu_stack, x, y, **kwargs.get('radial_profile', {}))

    if zlut is None:
        z = x * cp.nan
    else:
        z = lookup_z(
            profiles, zlut, **kwargs.get('lookup_z', {})
        )

    # Return and move back to regular memory
    return cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z), cp.asnumpy(profiles)


def stack_to_xyzp(stack, zlut=None):
    """Estimate XYZ coordinates and radial profiles from an image stack.

    This convenience wrapper orchestrates the CPU/GPU-agnostic pipeline used
    throughout MagTrack: the x and y are first estimated with
    :func:`center_of_mass`, refined with :func:`auto_conv`, further refined by
    five iterations of :func:`auto_conv_multiline_sub_pixel`, and then
    converted into radial profiles via :func:`radial_profile`. When a Z-look-up
    table is provided, :func:`lookup_z` translates those profiles into axial
    coordinates; otherwise, NaNs are returned for the z.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : array-like, shape (n_pixels, n_pixels, n_images)
        3-D image stack containing square images. The array can reside on the
        CPU (NumPy) or GPU (CuPy).
    zlut : array-like, shape (1 + n_bins, n_ref), optional
        Radial-profile look-up table whose first row stores the reference
        z-positions and remaining rows contain the corresponding template
        profiles. If omitted, the axial coordinate output is filled with NaNs.

    Returns
    -------
    x : 1D float array, shape (n_images)
        x-coordinates
    y : 1D float array, shape (n_images)
        y-coordinates
    z : 1D float array, shape (n_images)
        z-coordinates or a NaN array when ``zlut`` is None
    profiles : 2D float array, shape (n_bins, n_images)
        The average radial profile of each image about the center
    """
    # GPU or CPU?
    xp = cp.get_array_module(stack)

    # XY
    x, y = center_of_mass(stack)
    x, y = auto_conv(stack, x, y)
    for _ in range(5): # Repeat
        x, y = auto_conv_multiline_sub_pixel(stack, x, y)

    # Z
    profiles = radial_profile(stack, x, y)
    if zlut is None:
        z = x * xp.nan
    else:
        z = lookup_z(profiles, zlut)

    return x, y, z, profiles
