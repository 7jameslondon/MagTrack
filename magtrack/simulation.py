""" Various methods of simulating images of beads for testing """

import numpy as np
import scipy as sp
from magtrack._cupy import cp, cupyx

def airy_disk(xp=np, size=64, radius=10, wavelength=1.0):
    """ Generate an Airy disk pattern """
    xpx = sp if xp is np else cupyx.scipy
    x = xp.linspace(-size / 2, size / 2, size)
    y = xp.linspace(-size / 2, size / 2, size)
    xx, yy = xp.meshgrid(x, y)
    r = xp.sqrt(xx**2 + yy**2)
    r = xp.where(r == 0, 1e-10, r)  # Avoid division by zero

    k = 2 * xp.pi / wavelength
    kr = k * r / radius
    intensity = 2 * xpx.special.j1(kr) / kr

    intensity[r >= radius * 4] = 0

    return intensity

def simulate_beads(
    xyz_um, # array of [x_um, y_um, z_um]
    nm_per_px: float = 100.0,
    size_px: int = 64,
    radius_um: float = 1.5,
    wavelength_nm: float = 550.0, # Advanced
    n_sphere: float = 1.59, # Advanced
    n_medium: float = 1.33, # Advanced
    NA: float = 1.3, # Advanced
    absorption_per_um: float = 0.0, # Advanced
    background_level=0.8, # Advanced
    contrast_scale=1.0, # Advanced
    pad_factor=2.0, # Advanced
):
    # ========== Parameters ==========
    xyz_um = np.asarray(xyz_um, dtype=np.float64)

    # ========== Setup ==========
    wavelength_um = wavelength_nm * 1e-3
    dx_um = nm_per_px * 1e-3

    N = int(size_px)
    ax = (np.arange(N) - N//2) * dx_um
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    R = np.sqrt(X**2 + Y**2).astype(np.float64)

    a = float(radius_um)
    t = np.zeros_like(R, dtype=np.float64)
    inside = R <= a
    t[inside] = 2.0 * np.sqrt(a*a - R[inside]**2)

    opd = (n_sphere - n_medium) * t
    phi = 2.0 * np.pi * opd / wavelength_um
    A = np.exp(-absorption_per_um * t).astype(np.float64)

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
    fx = np.fft.fftfreq(Npad, d=dx_um)  # cycles/µm
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    k = 2.0 * np.pi * n_medium / wavelength_um
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY
    kxy2 = kx**2 + ky**2
    kz = np.sqrt(np.maximum(k**2 - kxy2, 0.0)).astype(np.float64)

    if NA is not None:
        f_cut = NA / wavelength_um
        pupil = (np.sqrt(FX**2 + FY**2) <= f_cut).astype(np.float64)
    else:
        pupil = np.ones_like(FX, dtype=np.float64)

    U0 = np.fft.fft2(T)

    # ========== Finally make images ==========
    stack = []
    bg_width = max(size_px // 16, 1)
    for x, y, z in xyz_um:
        H = np.exp(1j * z * kz) * pupil
        phase = np.exp(-2j * np.pi * (FX * x + FY * y))  # subpixel shift
        Uz_shift = np.fft.ifft2(U0 * H * phase)
        crop = Uz_shift[i0:i0+N, i0:i0+N]
        I = (np.abs(crop)**2).astype(np.float64)
        bg = float(I[:bg_width, :bg_width].mean()) or 1.0
        I = background_level * (1.0 + contrast_scale * (I / bg - 1.0))
        stack.append(I)

    stack = np.stack(stack, axis=2)
    stack /= 2
    stack = np.clip(stack, 0, 1)
    return stack