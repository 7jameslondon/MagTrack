""" Various methods of simulating images of beads for testing """

import numpy as np
import scipy as sp

from ._cupy import cp, cupyx

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = airy_disk(xp=np)
    plt.imshow(image)
    plt.show(block=False)


