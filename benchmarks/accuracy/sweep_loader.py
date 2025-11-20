"""Utilities for loading accuracy benchmark sweeps.

This module provides an eager loader for sweep artifacts produced by
``bead_simulation_sweep.py``. Images are kept in their original dtype (expected
to be float64) and the associated metadata is returned for downstream
consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_SWEEP_ROOT = Path(__file__).resolve().parent / "sweeps"


@dataclass(frozen=True)
class SweepData:
    """Container for sweep artifacts."""

    sweep_name: str
    sweep_dir: Path
    images: dict[str, np.ndarray]
    metadata: dict[str, Any]
    combinations: dict[str, dict[str, Any]]

    def image_keys(self) -> list[str]:
        """Return the ordered image keys."""

        return list(self.images.keys())


def load_sweep(sweep_name: str, sweep_root: Path | str | None = None) -> SweepData:
    """Load a sweep by name from the ``benchmarks/accuracy/sweeps`` directory.

    Parameters
    ----------
    sweep_name
        Name of the sweep directory to load.
    sweep_root
        Optional override for the sweeps root directory.

    Raises
    ------
    FileNotFoundError
        If the sweep directory or expected files are missing.
    ValueError
        If any loaded image is not float64.
    """

    root = Path(sweep_root) if sweep_root is not None else _DEFAULT_SWEEP_ROOT
    sweep_dir = root / sweep_name
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory does not exist: '{sweep_dir}'")

    images_path = sweep_dir / "images.npz"
    metadata_path = sweep_dir / "metadata.json"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing images file: '{images_path}'")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: '{metadata_path}'")

    with np.load(images_path) as npz_file:
        images = {key: npz_file[key] for key in npz_file.files}

    for key, image in images.items():
        if image.dtype != np.float64:
            raise ValueError(
                f"Image '{key}' has dtype {image.dtype}; expected float64.",
            )

    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    combinations: dict[str, dict[str, Any]] = {}
    for param_set in metadata.get("parameter_sets", []):
        for combo in param_set.get("combinations", []):
            key = combo.get("key")
            if key is not None:
                combinations[key] = combo.get("values", {})

    return SweepData(
        sweep_name=sweep_name,
        sweep_dir=sweep_dir,
        images=images,
        metadata=metadata,
        combinations=combinations,
    )


__all__ = ["SweepData", "load_sweep"]
