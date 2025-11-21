# Benchmarks: Accuracy – Working Notes

## Overall goal
- Build an accuracy benchmark for MagTrack bead localization (x, y now; z and radial profile later) using simulated bead images so we can quantify algorithm performance on controlled test cases.
- Use the simulation sweeps under `benchmarks/accuracy/sweeps` as reusable fixtures for generating images + metadata, plotting them, and later evaluating localization error.

## Current components
- **`bead_simulation_sweep.py`**
  - Provides `BeadSimulationSweep` to generate images via `magtrack.simulation.simulate_beads`, storing `images.npz` and `metadata.json` per sweep directory. Images are float64 arrays normalized for visualization/runs.
  - `ParameterSet` defines parameter grids; default set sweeps `size_px` over `[64, 128, 256, 512]` with zero offsets and `background_level` 0.8. Keys for images follow `{set_name}__{index:04d}`.
  - Metadata captures parameter combinations and system info; raises if outputs exist unless `overwrite=True`.
- **`sweep_loader.py`**
  - `SweepData.load(name)` loads a sweep from `sweeps/<name>/images.npz` + `metadata.json`, attaching flattened parameter values to each `SweepImage` for downstream evaluation.
- **`plot_sweep.py`**
  - CLI (`python -m benchmarks.accuracy.plot_sweep --sweep <name>`) loads a sweep and displays all images in a scrollable Tk UI. Titles show key/value parameters (wrapped to width 40). Grid auto-sizes to near-square.
- **`sweeps/default`**
  - Contains the default sweep artifacts generated from the default parameter set (4 images, sizes 64–512). Serves as a sanity-check fixture for plotting/loading.

## How to generate/inspect sweeps
1) Generate (overwrites disabled by default):
   ```bash
   python -m benchmarks.accuracy.bead_simulation_sweep --sweep-name <name> --overwrite
   ```
   or call `BeadSimulationSweep(...).generate()` directly after constructing custom `ParameterSet`s.
2) Plot for quick visual QA: `python -m benchmarks.accuracy.plot_sweep --sweep <name>`.

## Maintenance
- Keep this file accurate and concise for future agents; remove outdated notes and avoid duplicating information available directly in code or metadata.
- Update with key findings, new sweeps, and evaluation steps as they are implemented so future accuracy tasks start from current context.

## Open questions / next steps
- Define evaluation routines that consume `SweepData` to measure localization error for x/y (and later z/radial profile) against known offsets in metadata.
- Decide on additional sweeps (e.g., variable noise, offsets, z, seed, GPU vs CPU) and standardize naming.
- Clarify expected output metrics/reporting format for the accuracy benchmark suite.

## Gotchas / reminders
- Sweep generation will refuse to overwrite existing artifacts unless `overwrite=True`.
- Parameter values stored in metadata are JSON-serializable; ensure custom parameters follow that pattern.
- Tk plotting requires a display; consider headless-friendly alternatives if CI visualization is needed.
