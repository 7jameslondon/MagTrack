Getting Started
===============

What is MagTrack?
-----------------
MagTrack is a Python package for tracking symmetric beads in single-molecule magnetic tweezers experiments. The
library focuses on fast, accurate bead localization and includes an optional GPU pipeline for labs with CUDA-capable
hardware. Out of the box it provides documented, benchmarked algorithms that return sub-pixel XYZ coordinates and the
underlying radial intensity profiles used during axial fitting.

Key features
------------

* GPU accelerated (optional, requires a CUDA GPU)
* Sub-pixel XYZ coordinates
* Python notebook included with examples
* Documented, tested, and benchmarked
* Only depends on NumPy, SciPy, and CuPy
* Actively maintained

System requirements
-------------------

MagTrack requires Python 3.9 or newer as well as recent versions of NumPy and SciPy. Optional GPU acceleration is
available through CuPy (CUDA 12.x builds). The package works on Windows, Linux, and macOS. macOS can run the CPU
algorithms, but NVIDIA GPU acceleration is only available on platforms with CUDA support.

Installation
------------

Install the latest release from PyPI. Include the ``[gpu]`` extra if you plan to use CuPy-based acceleration:

.. code-block:: console

   pip install magtrack[gpu]

To install only the CPU dependencies:

.. code-block:: console

   pip install magtrack

If you intend to use GPU acceleration, make sure that the CUDA Toolkit compatible with your GPU is installed before
installing CuPy. Refer to the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for detailed
instructions.

Verify the installation
-----------------------

After installing, import :mod:`magtrack` in a Python session and print the version string. This confirms that Python can
locate the package and that the compiled extensions were installed correctly:

.. code-block:: python

   import magtrack
   print(magtrack.__version__)

You can also execute the quick-start pipeline, which converts an image stack into lateral and axial bead coordinates:

.. code-block:: python

   import magtrack

   x, y, z, profiles = magtrack.stack_to_xyzp(stack)

Here ``stack`` is a 3D NumPy array of bead images (frames × rows × columns). The function returns the fitted lateral
coordinates ``x`` and ``y``, axial positions ``z``, and the radial profiles ``profiles`` that were used in the axial fit.

Explore the examples
--------------------

The ``examples/examples.ipynb`` notebook walks through the core functionality, including simulation, tracking, and GPU
acceleration. Download the notebook to run it locally or open it in Google Colab from the README badge. The notebook's
scenarios are mirrored throughout the :doc:`user_guide`, so reading both side-by-side provides an effective first tour of
MagTrack.
