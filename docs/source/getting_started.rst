Getting Started
===============

What is MagTrack?
-----------------
`MagTrack <https://github.com/7jameslondon/MagTrack/>`_ is a free open-source Python library for tracking symmetric beads in `single-molecule magnetic tweezers <https://doi.org/10.1007/978-1-0716-3377-9_18>`_ experiments.

* Sub-pixel XYZ coordinates
* GPU accelerated (optional, requires a CUDA-capable GPU)
* Python notebook included with `examples <https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb>`_
* `Documented <https://magtrack.readthedocs.io/en/stable/>`_, `tested <https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml>`_, and `benchmarked <https://magtrack.readthedocs.io/en/stable/benchmarking.html>`_
* Only depends on `NumPy <https://numpy.org>`_, `SciPy <https://scipy.org>`_, and `CuPy <https://cupy.dev>`_
* Runs on Windows, macOS, and Linux
* `Easy installation <https://magtrack.readthedocs.io/en/stable/getting_started.html#installation>`_ with pip

System requirements
-------------------

- `Python <https://www.python.org/downloads/>`_  >=3.9
- Supports Windows, macOS, and Linux

  - macOS does not support CUDA GPU acceleration

Installation
------------


Method 1: CPU only (easier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method supports macOS, Windows, or Linux. But MagTrack will only be able to use your CPU.
You can always later use method 2 if you want to add GPU acceleration.

.. code-block:: console

   pip install magtrack[cpu]

Method 2: CPU + GPU (a little harder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method supports Windows or Linux.
This allows MagTrack to use either the CPU or a CUDA-capable GPU.

#. Confirm that your system has an NVIDIA GPU.

#. Install the free `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_.

   * Versions 11, 12, or 13 are supported.

   * You can check the installed version by running ``nvcc -V`` in a terminal such as Command Prompt.

#. Install MagTrack using the pip command that matches your CUDA version:

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
