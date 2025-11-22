Simulation Guide
================

Use :mod:`magtrack.simulation` to generate synthetic bead stacks and Z-LUTs for tutorials, benchmarking, or validating your
analysis workflow. The examples below show the minimal inputs required to create stacks and capture both
lateral and axial motion.

Quick start
-----------

.. code-block:: python

   import numpy as np
   from magtrack.simulation import simulate_beads

   # Define some motion (x, y, z in nanometers)
   frames = 120
   x_true = np.linspace(-150, 150, frames)
   y_true = 80.0 * np.sin(np.linspace(0, 2 * np.pi, frames))
   z_true = np.zeros_like(x_true)
   xyz_true = np.column_stack([x_true, y_true, z_true])

   # Render a 64x64 pixel stack at 500 nm/px
   stack = simulate_beads(xyz_true, size_px=64, nm_per_px=500.0)

   # ``stack`` now has shape (64, 64, 120) and contains a single bead drifting and oscillating over time.

Tips for clean simulations
--------------------------

* Keep properties like ``nm_per_px`` consistent between simulations and downstream analysis to avoid scale mismatches.
* The simulation works best with larger image sizes (``size_px >= 64px``).
* The simulation is far from perfect. The wrong combination of parameters can unrealistic images of beads. Start with
the default simulation values and work your way towards what you want to simulate.

Examples
----------------------------

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
