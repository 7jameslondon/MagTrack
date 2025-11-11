Getting Started
===============

What is MagTrack?
-----------------
MagTrack is a Python package for tracking symmetric beads in single-molecule magnetic tweezers experiments. The
library focuses on fast, accurate bead localization and includes an optional GPU acceleration for labs with CUDA-capable
hardware.

Key features
------------

* Sub-pixel XYZ coordinates
* GPU accelerated (optional, requires a CUDA GPU)
* Python notebook included with `examples <https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb>`_
* `Documented <https://magtrack.readthedocs.io/en/stable/>`_, `tested <https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml>`_, and `benchmarked <https://github.com/7jameslondon/MagTrack/tree/master/benchmarks>`_
* Only depends on `NumPy <https://numpy.org/>`_, `SciPy <https://scipy.org/>`_, and `CuPy <https://cupy.dev/>`_

System requirements
-------------------

MagTrack requires Python 3.9 or newer as well as recent versions of NumPy and SciPy. Optional GPU acceleration is
available through CuPy (CUDA 11.x, 12.x, or 13.x builds). The package works on Windows, Linux, and macOS. macOS can run the CPU
algorithms, but NVIDIA GPU acceleration is only available on platforms with CUDA support.

* Python >=3.9
* NumPy >=1.26
* SciPy >=1.11.1
* (Optional) CuPy-CUDA11x >=12.0
* (Optional) CuPy-CUDA12x >=13.0
* (Optional) CuPy-CUDA13x >=13.0

Installation
------------

MagTrack supports Windows, macOS, and Linux. macOS installs are limited to CPU execution because CUDA-based GPU
acceleration is only available on Windows and Linux systems. Ensure that you have Python 3.9 or newer installed before
continuing.

Method 1: CPU only (easier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method works on macOS, Windows, or Linux and installs the CPU-only dependencies.

.. code-block:: console

   pip install magtrack[cpu]

Conda installation *coming soon*.

Method 2: CPU or GPU (a little harder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method supports Windows or Linux and allows MagTrack to use either the CPU or a CUDA-capable GPU. Confirm that
your system has an NVIDIA GPU and install the free `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ first.
Versions 11, 12, or 13 are supported. You can verify the installed version by running ``nvcc -V`` in a terminal.

After installing the CUDA Toolkit, install MagTrack using the pip command that matches your CUDA version:

.. list-table::
   :header-rows: 1

   * - CUDA Toolkit Version
     - Pip command
   * - 11.x
     - ``pip install magtrack[cu11]``
   * - 12.x
     - ``pip install magtrack[cu12]``
   * - 13.x
     - ``pip install magtrack[cu13]``

Conda installation *coming soon*.

Verify the installation
-----------------------

After installing, import MagTrack in a Python session and print the version string. This confirms that Python can
locate the package and that the compiled extensions were installed correctly:

.. code-block:: python

   import magtrack
   print(magtrack.__version__)

You can also try a function like center-of-mass, which converts an image stack into x,y bead coordinates:

.. code-block:: python

   import magtrack
   import numpy as np

   stack = np.zeros((10, 10, 3))
   x, y = magtrack.center_of_mass(stack)

   print(f'x: {x}')
   print(f'y: {y}')

Explore the examples
--------------------

The ``examples/examples.ipynb`` notebook walks through the core functionality, including simulation, tracking, and GPU
acceleration. `Download <https://github.com/7jameslondon/MagTrack/blob/master/examples/examples.ipynb>`_ the notebook to run it locally or open it in `Google Colab <https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb>`_.
