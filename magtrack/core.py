import cupy as cp
import cupyx.scipy.signal


def binmean(x, weights, n_bins: int):
    """
    Similar to numpy.bincount but for mean of 2D arrays

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

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
    i_base = xp.arange(x.shape[1], dtype=xp.uint16)
    i = xp.broadcast_to(i_base, x.shape)

    # Binning
    bin_means = xp.zeros(
        (n_bins + 1, n_datasets), dtype=xp.float32
    )  # Pre-allocate
    xp.add.at(bin_means, (x, i), weights)

    bin_counts = xp.zeros(
        (n_bins + 1, n_datasets), dtype=xp.float32
    )  # Pre-allocate
    xp.add.at(bin_counts, (x, i), 1)

    bin_means /= bin_counts

    return bin_means[:-1, :]  # Return without the overflow row


def crop_stack_to_rois(stack, rois):
    """
    Takes a 3D image-stack and crops it to the region of interests (ROIs).

    Given a 3D image-stack and a list of ROIs, this function will crop around
    each ROI and return a 4D array. Note the input stack must contain square
    images and the ROIs must be squares.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU. However, it is
    recommended to use the CPU and then transfer the result to the GPU and
    perform downstream analysis on the GPU.

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


def parabolic_vertex(data, vertex_est, n_local: int):
    """
    Refines a local min/max by parabolic interpolation.

    Given an estimated location of a local min/max, this function will find a
    more precise location of the vertex by fitting the local datapoints to a
    parabola and interpolating the vertex.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
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
    vertex_int = vertex_est.round().astype(xp.int32)

    # Force index to be with the limits
    index_min = n_local_half
    index_max = data.shape[1] - n_local_half - 1
    xp.clip(vertex_int, index_min, index_max, out=vertex_int)

    # Get the local data to be fit
    n_datasets = data.shape[0]
    rel_idx = xp.arange(-n_local_half, n_local_half + 1, dtype=xp.int32)
    idx = rel_idx + vertex_int[:, xp.newaxis]
    y = data[xp.arange(n_datasets)[:, xp.newaxis], idx].T
    x = xp.arange(n_local, dtype=xp.float32)

    # Fit to parabola
    p = xp.polyfit(x, y, 2)

    # Calculate the location of the vertex
    vertex = -p[1, :] / (2. * p[0, :]) + vertex_int - n_local // 2.  # -b/2a

    # Exclude points outside limits
    vertex[vertex_int == index_min] = xp.nan
    vertex[vertex_int == index_max] = xp.nan

    return vertex


def center_of_mass(stack):
    """
    Calculate center-of-mass

    For each 2D image of a 3D image-stack: the mean background is
    subtracted, the absolute value is taken and then the center-of-mass in
    along the x- and y-axis is calculated. This function is faster than the
    version from scipy or cupyx.scipy.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    stack : 3D float array, shape (n_pixels, n_pixels, n_images)
        The image-stack. The images must be square.

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
    stack_norm = stack.copy()
    xp.subtract(stack_norm, xp.mean(stack, axis=(0, 1)), out=stack_norm)
    xp.absolute(stack_norm, out=stack_norm)

    # Calculate center-of-mass
    index = xp.arange(stack_norm.shape[0], dtype=xp.float32)[:, xp.newaxis]
    total_mass = xp.sum(stack_norm, axis=(0, 1))
    x = xp.sum(index * xp.sum(stack_norm, axis=0), axis=0) / total_mass
    y = xp.sum(index * xp.sum(stack_norm, axis=1), axis=0) / total_mass

    return x, y


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
    frame_idx = xp.arange(stack.shape[2], dtype=xp.int32)
    x_idx = xp.round(x_old).astype(xp.int32)
    y_idx = xp.round(y_old).astype(xp.int32)
    cols = stack[:, x_idx, frame_idx]
    rows = stack[y_idx, :, frame_idx]

    # Subtract means
    cols -= xp.mean(cols, axis=0, keepdims=True)
    rows -= xp.mean(rows, axis=1, keepdims=True)

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


def auto_conv_para_fit(stack, x_old, y_old, n_local=11):
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


def radial_profile(stack, x, y):
    """
    Calculate the average radial profile about a center

    For each 2D image of a 3D image-stack: calculate the average radial profile
    about the corresponding center (x and y). The profile is calculated by
    binning. For each pixel in an image the Euclidean distance from the center
    is calculated. The distance is then used to bin each pixel. The bin widths
    are 1 pixel wide. The bins are then normalized by the number of pixels in
    each bin to find the average intensity in each bin. The number of bins
    (n_bins) is stack.shape[0] // 4.

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
    n_bins = stack.shape[0] // 4
    grid = xp.indices((width, width), dtype=xp.float32)
    flat_stack = stack.reshape((width * width, n_images))

    # Calculate distance of each pixel from x and y
    r = xp.hypot(grid[1][:, :, xp.newaxis] - x, grid[0][:, :, xp.newaxis] -
                 y).reshape(-1, n_images).round().astype(xp.uint16)

    # Calculate profiles
    profiles = binmean(r, flat_stack, n_bins)

    return profiles


def lookup_z_para_fit(profiles, zlut, n_local=3):
    """
    Calculate the corresponding sub-planar z-coordinate of each profile by LUT

    For each image's profile in profiles: find the best matching profile in the
    Z-LUT (lookup table). Fits the local points around the best matching
    profile to find sub-planar fit in between columns of the LUT.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    profiles : 2D float array, shape (n_bins, n_images)
        The average radial profile of each image about the center
    zlut : 2D float array, shape (1+n_bins, n_ref)
        The reference radial profiles and corresponding z-coordinates. The
        first row (zlut[0, :]) are the z-coordinates. The rest of the rows are
        the corresponding profiles as generated by radial_profile.
    n_local : int
        The number of local points around the vertex to be used in parabolic
        fitting. Must be an odd int >=3.

    Returns
    ----------
    z : 1D float array, shape (n_images)
        z-coordinates
    """
    # GPU or CPU?
    xp = cp.get_array_module(profiles)

    ref_z = zlut[0, :]
    ref_profiles = zlut[1:, :]
    n_images = profiles.shape[1]
    n_ref = ref_profiles.shape[1]

    # Calculate the total difference between Z-LUT and current profiles.
    # This (likely) needs to be done in a loop to prevent the subtraction
    # operation from taking too much memory at once.
    dif = xp.empty((n_images, n_ref), dtype=xp.float32)
    for i in range(n_images):
        dif[i, :] = xp.sum(xp.abs(ref_profiles - profiles[:, i:i + 1]), axis=0)

    # Find index of the minimum difference
    z_int_idx = xp.argmin(dif, axis=1).astype(xp.float32)

    # Find the sub-planar index of the minimum difference
    z_idx = parabolic_vertex(dif, z_int_idx, n_local)

    # Interpolate z from reference index
    z = xp.interp(z_idx, xp.arange(n_ref), ref_z, left=xp.nan, right=xp.nan)

    return z


# TODO: Add documentation
def stack_to_xyzp(stack, zlut=None):
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

    Returns
    ----------
    z : 1D float array, shape (n_images)
        z-coordinates
    """
    # Move stack to GPU memory
    gpu_stack = cp.asarray(stack, dtype=cp.float32)

    x, y = center_of_mass(gpu_stack)

    for _ in range(1):
        x, y = auto_conv(gpu_stack, x, y)

    for _ in range(1):
        x, y = auto_conv_para_fit(gpu_stack, x, y)

    profiles = radial_profile(gpu_stack, x, y)

    if zlut is None:
        z = x * cp.nan
    else:
        z = lookup_z_para_fit(profiles, zlut)

    # Return and move back to regular memory
    return cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z), cp.asnumpy(profiles)
