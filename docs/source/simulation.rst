Simulation Guide
================

Use :mod:`magtrack.simulation` to generate synthetic bead stacks for tutorials, benchmarking, or validating your
analysis workflow. The examples below show the minimal inputs required to create repeatable stacks and capture both
lateral and axial motion.

Quick start
-----------

.. code-block:: python

   import numpy as np
   from magtrack.simulation import simulate_beads

   # Define ground-truth motion (x, y, z in nanometers)
   frames = 120
   x_true = np.linspace(-150, 150, frames)
   y_true = 80.0 * np.sin(np.linspace(0, 2 * np.pi, frames))
   z_true = np.zeros_like(x_true)
   xyz_true = np.column_stack([x_true, y_true, z_true])

   # Render a 96x96 pixel stack at 90 nm/px
   stack = simulate_beads(xyz_true, size_px=96, nm_per_px=90.0)

   # ``stack`` now has shape (96, 96, 120) and contains a single bead drifting and oscillating over time.

Tips for clean simulations
--------------------------

* Keep ``nm_per_px`` consistent between simulations and downstream analysis to avoid scale mismatches.
* Simulate a small axial sweep (e.g., from ``-500`` to ``500`` nm) to test Z-LUT lookups before processing long
  recordings.
* When benchmarking, reuse the same ``xyz_true`` arrays so algorithm changes can be compared frame-for-frame.

Bead resolution placeholders
----------------------------

Below are placeholders for visual references of the same bead rendered at different resolutions. Replace the image files
in ``docs/source/_static/placeholders/`` when you have the final assets.

.. figure:: _static/placeholders/bead_low.png
   :alt: Placeholder for a low-resolution bead rendering
   :align: center

   Low-resolution bead placeholder.

.. figure:: _static/placeholders/bead_medium.png
   :alt: Placeholder for a medium-resolution bead rendering
   :align: center

   Medium-resolution bead placeholder.

.. figure:: _static/placeholders/bead_high.png
   :alt: Placeholder for a high-resolution bead rendering
   :align: center

   High-resolution bead placeholder.

Z-LUT placeholders
------------------

Use these slots to showcase axial lookup tables derived from low, medium, and high-resolution reference stacks. Swap in
real Z-LUT previews when they are ready.

.. figure:: _static/placeholders/zlut_low.png
   :alt: Placeholder for a low-resolution Z-LUT preview
   :align: center

   Low-resolution Z-LUT placeholder.

.. figure:: _static/placeholders/zlut_medium.png
   :alt: Placeholder for a medium-resolution Z-LUT preview
   :align: center

   Medium-resolution Z-LUT placeholder.

.. figure:: _static/placeholders/zlut_high.png
   :alt: Placeholder for a high-resolution Z-LUT preview
   :align: center

   High-resolution Z-LUT placeholder.
