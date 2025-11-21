"""Simple runner to generate and plot the default accuracy sweep."""

from __future__ import annotations

from benchmarks.accuracy.bead_simulation_sweep import BeadSimulationSweep
from benchmarks.accuracy.plot_sweep import main as plot_sweep_main


def main() -> None:
    sweep = BeadSimulationSweep("default")
    sweep.generate(overwrite=True)
    plot_sweep_main()


if __name__ == "__main__":
    main()
