"""Simulations for generating realistic synthetic images and Z-LUTs of beads."""

import numpy as np
import scipy as sp

from magtrack._cupy import cp, cupyx
from magtrack.core import radial_profile

def simulate_beads(
    xyz_nm, # array of [x_nm, y_nm, z_nm]
    nm_per_px: float = 100.0,
    size_px: int = 64,
    radius_nm: float = 1500.0,
    wavelength_nm: float = 550.0, # Advanced
    n_sphere: float = 1.59, # Advanced
    n_medium: float = 1.33, # Advanced
    absorption_per_nm: float = 0.0, # Advanced
    background_level=0.4, # Advanced
    contrast_scale=1.0, # Advanced
    pad_factor=2.0, # Advanced
):
    """Simulate brightfield images of spherical beads.

    Generates a 3D stack of normalized brightfield images for beads located at
    the provided coordinates in nanometers. The simulation models refraction and
    absorption through a spherical object, applies a sampling-limited pupil
    defined by the imaging wavelength and medium, and returns images
    cropped to the requested pixel size.

    Parameters
    ----------
    xyz_nm : array_like, shape (n_beads, 3)
        Nanometer coordinates ``[x_nm, y_nm, z_nm]`` for each bead.
    nm_per_px : float, optional
        Nanometers per pixel. Defaults to 100.0 nm/px.
    size_px : int, optional
        Output image width and height in pixels. Defaults to 64.
    radius_nm : float, optional
        Bead radius in nanometers. Defaults to 1500.0 nm.
    wavelength_nm : float, optional
        Illumination wavelength in nanometers. Defaults to 550.0 nm.
    n_sphere : float, optional
        Refractive index of the bead. Defaults to 1.59.
    n_medium : float, optional
        Refractive index of the surrounding medium. Defaults to 1.33.
    absorption_per_nm : float, optional
        Absorption coefficient per nanometer. Defaults to 0.0.
    background_level : float, optional
        Baseline background intensity scaling. Defaults to 0.4.
    contrast_scale : float, optional
        Contrast scaling relative to the background. Defaults to 1.0.
    pad_factor : float, optional
        Factor for zero-padding the pupil function to reduce edge effects.
        Defaults to 2.0.

    Returns
    -------
    stack : ndarray, shape (size_px, size_px, n_beads)
        Normalized simulated bead images with intensities clipped to ``[0, 1]``.
    """
    # ========== Parameters ==========
    xyz_nm = np.asarray(xyz_nm, dtype=np.float64)

    # ========== Setup ==========
    dx_nm = float(nm_per_px)

    # NA:
    NA_eff = 1.3

    N = int(size_px)
    ax = (np.arange(N) - N//2) * dx_nm
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    R = np.sqrt(X**2 + Y**2).astype(np.float64)

    a = float(radius_nm)
    t = np.zeros_like(R, dtype=np.float64)
    inside = R <= a
    t[inside] = 2.0 * np.sqrt(a*a - R[inside]**2)

    opd = (n_sphere - n_medium) * t
    phi = 2.0 * np.pi * opd / wavelength_nm
    A = np.exp(-absorption_per_nm * t).astype(np.float64)

    T_small = np.ones((N, N), dtype=np.complex128)
    T_small[inside] = (A[inside].astype(np.complex128)) * np.exp(1j * phi[inside]).astype(np.complex128)

    # ========== Padding ==========
    Npad = int(np.ceil(pad_factor * N))
    if Npad % 2:
        Npad += 1
    T = np.ones((Npad, Npad), dtype=np.complex128)
    i0 = Npad//2 - N//2
    T[i0:i0+N, i0:i0+N] = T_small

    # ========== Pre-compute ==========
    fx = np.fft.fftfreq(Npad, d=dx_nm)  # cycles/nm
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    k = 2.0 * np.pi * n_medium / wavelength_nm
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY
    kxy2 = kx**2 + ky**2
    kz = np.sqrt(np.maximum(k**2 - kxy2, 0.0)).astype(np.float64)

    # Pupil with sampling-limited effective NA
    f_cut = NA_eff / wavelength_nm
    pupil = (np.sqrt(FX ** 2 + FY ** 2) <= f_cut).astype(np.complex128)

    def _sinc(x):
        out = np.ones_like(x)
        nz = x != 0
        out[nz] = np.sin(x[nz]) / x[nz]
        return out

    M_pixel = _sinc(np.pi * dx_nm * FX) * _sinc(np.pi * dx_nm * FY)
    pupil *= M_pixel

    U0 = np.fft.fft2(T)

    # ========== Finally make images ==========
    stack = []
    bg_width = max(size_px // 16, 1)
    for x, y, z in xyz_nm:
        H = np.exp(1j * z * kz) * pupil
        phase = np.exp(-2j * np.pi * (FX * x + FY * y))  # subpixel shift
        Uz_shift = np.fft.ifft2(U0 * H * phase)
        crop = Uz_shift[i0:i0+N, i0:i0+N]
        I = (np.abs(crop)**2).astype(np.float64)
        bg = float(I[:bg_width, :bg_width].mean()) or 1.0
        I = background_level * 2 * (1.0 + contrast_scale * (I / bg - 1.0))
        stack.append(I)

    stack = np.stack(stack, axis=2)
    stack /= 2
    stack = np.clip(stack, 0, 1)
    return stack


def simulate_zlut(
    z_nm,
    nm_per_px: float = 100.0,
    size_px: int = 64,
    oversample: int = 1,
    **simulate_beads_kwargs,
):
    """Simulate a Z look-up table (Z-LUT) from synthetic bead images.

    This convenience function renders a stack of bead images at the supplied
    axial positions using :func:`simulate_beads`, converts each frame into a
    radial profile via :func:`magtrack.core.radial_profile`, and assembles the
    results into a Z-LUT where the first row stores the z positions.

    Parameters
    ----------
    z_nm : array_like, shape (n_planes,)
        Z coordinates in nanometers for the reference stack.
    nm_per_px : float, optional
        Nanometers per pixel used for the simulation. Defaults to 100.0.
    size_px : int, optional
        Width/height in pixels of the simulated frames. Defaults to 64.
    oversample : int, optional
        Oversampling factor passed to :func:`magtrack.core.radial_profile`.
        Defaults to 1.
    **simulate_beads_kwargs : dict, optional
        Additional keyword arguments forwarded to :func:`simulate_beads` (for
        example ``radius_nm`` or ``wavelength_nm``).

    Returns
    -------
    zlut : ndarray, shape (1 + n_bins, n_planes)
        Z look-up table where ``zlut[0, :]`` contains the reference z positions
        and the remaining rows contain the corresponding radial profiles.
    """

    z_nm = np.asarray(z_nm, dtype=np.float64)
    n_planes = z_nm.shape[0]

    xyz_nm = np.column_stack([
        np.zeros_like(z_nm, dtype=np.float64),
        np.zeros_like(z_nm, dtype=np.float64),
        z_nm,
    ])

    stack = simulate_beads(
        xyz_nm, nm_per_px=nm_per_px, size_px=size_px, **simulate_beads_kwargs
    )

    center = np.full(n_planes, size_px / 2.0, dtype=np.float64)
    profiles = radial_profile(stack, center, center, oversample=oversample)

    zlut = np.vstack([z_nm, profiles])
    return zlut
