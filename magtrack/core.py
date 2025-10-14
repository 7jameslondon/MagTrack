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

    return bin_means[:-1, :]  # Return without the overflow row


def pearson(x, y):
    """
    Calculate the Pearson correlation coefficient.

    Note: CPU or GPU: The code is agnostic of CPU and GPU usage. If the first
    parameter is on the GPU the computation/result will be on the GPU.
    Otherwise, the computation/result will be on the CPU.

    Parameters
    ----------
    x : 2D array, shape (n, m)
        Input array
    y : 1D array, shape (n,)
        Input array

    Returns
    ----------
    r : 1D array, shape (m,)
        Pearson correlation coefficient
    """

    # GPU or CPU?
    xp = cp.get_array_module(x)

    mean_x = xp.nanmean(x, axis=0)
    mean_y = xp.nanmean(y, axis=0)
    dif_x = x - mean_x
    dif_y = y - mean_y
    dif_x2 = dif_x**2
    dif_y2 = dif_y**2
    r = xp.nansum(dif_x * dif_y, axis=0) / xp.sqrt(
        xp.nansum(dif_x2, axis=0) * xp.nansum(dif_y2, axis=0)
    )

    return r


def gaussian(x, mu, sigma):
    # GPU or CPU?
    xp = cp.get_array_module(x)

    return xp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def gaussian_2d(x, y, mu_x, mu_y, sigma):
    """
    Calculate a 2D Gaussian function.

    Parameters
    ----------
    x : array_like
        x coordinates where to evaluate the gaussian
    y : array_like  
        y coordinates where to evaluate the gaussian
    mu_x : float
        Mean (center) in x direction
    mu_y : float
        Mean (center) in y direction 
    sigma : float
        Standard deviation in x and y direction

    Returns
    -------
    array_like
        2D array containing the gaussian evaluated at x,y coordinates
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
    weighted : bool, optional
        Whether to apply a simple weighting procedure to emphasize the more cental points in the fit. Default is True.

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


def center_of_mass(stack, background='none'):
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
    background : str, optional
        The method to subtract background. 'mean' subtracts the mean. 'median'
        subtracts the median. Otherwise, no background is subtracted. Default
        is 'mean'.

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


def auto_conv_para_fit(stack, x_old, y_old, n_local=5):
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


def auto_conv_multiline(
    stack, x_old, y_old, line_ratio=0.05, return_conv=False
):
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
    line_ratio : int, optional
        The ratio relative to the total image width of lines to be used in the
        convolutions.
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
    half_n_lines = int(stack.shape[0] * line_ratio // 2)
    n_lines = half_n_lines * 2 + 1
    line_idx = xp.arange(-half_n_lines, half_n_lines + 1)
    width = stack.shape[0]
    n_images = stack.shape[2]
    frame_idx = xp.arange(n_images, dtype=xp.int64)
    frame_idx = xp.repeat(frame_idx, n_lines)
    x_idx = xp.round(x_old).astype(xp.int64)
    y_idx = xp.round(y_old).astype(xp.int64)
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


# TODO: Add documentation
def auto_conv_multiline_para_fit(
    stack, x_old, y_old, line_ratio=0.05, n_local=5
):
    col_max, row_max, col_con, row_con = auto_conv_multiline(
        stack, x_old, y_old, return_conv=True, line_ratio=line_ratio
    )

    x = parabolic_vertex(row_con, row_max, n_local)
    y = parabolic_vertex(col_con.T, col_max, n_local)

    radius = (stack.shape[0] - 1) // 2
    x = radius - ((radius - x) / 2)
    y = radius - ((radius - y) / 2)

    return x, y


def radial_profile(stack, x, y, oversample=4):
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
    n_bins = (width // 4) * oversample
    grid = xp.indices((width, width), dtype=xp.float32)
    flat_stack = stack.reshape((width * width, n_images))

    # Calculate distance of each pixel from x and y
    # cast to uint16 because min and max r for 1024x1024 would be 0 and 1449
    r = xp.round(
        xp.hypot(grid[1][:, :, xp.newaxis] - x, grid[0][:, :, xp.newaxis] - y) *
        oversample
    ).astype(xp.uint16).reshape(-1, n_images)

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


def lookup_z_para_fit(profiles, zlut, n_local=5):
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
    ref_profiles = zlut[2:, :]  # Skip the first pixel
    n_images = profiles.shape[1]
    n_ref = ref_profiles.shape[1]

    # Calculate the pearson correlation coefficient between Z-LUT and current profiles.
    # This (likely) needs to be done in a loop to prevent the
    # operation from taking too much memory at once.
    r = xp.empty((n_images, n_ref), dtype=xp.float64)
    for i in range(n_images):
        # Skip the first pixel
        r[i, :] = pearson(ref_profiles, profiles[1:, i:i + 1])

    # Find index of the max
    z_int_idx = xp.argmax(r, axis=1).astype(xp.float64)

    # Find the sub-planar index of the max
    z_idx = parabolic_vertex(r, z_int_idx, n_local)

    # Interpolate z from reference index
    z = xp.interp(z_idx, xp.arange(n_ref), ref_z, left=xp.nan, right=xp.nan)

    return z


def stack_to_xyzp(stack, zlut=None, **kwargs):
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

    for _ in range(kwargs.get('n auto_conv_multiline_para_fit', 5)):
        x, y = auto_conv_multiline_para_fit(
            gpu_stack, x, y, **kwargs.get('auto_conv_multiline_para_fit', {})
        )

    if kwargs.get('use fft_profile', True):
        profiles = fft_profile(gpu_stack, x, y, **kwargs.get('fft_profile', {}))
    else:
        profiles = radial_profile(gpu_stack, x, y, **kwargs.get('radial_profile', {}))

    if zlut is None:
        z = x * cp.nan
    else:
        z = lookup_z_para_fit(
            profiles, zlut, **kwargs.get('lookup_z_para_fit', {})
        )

    # Return and move back to regular memory
    return cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z), cp.asnumpy(profiles)
