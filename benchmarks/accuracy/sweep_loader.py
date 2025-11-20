"""Utilities for loading accuracy benchmark sweeps from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_DEFAULT_SWEEP_ROOT = Path(__file__).resolve().parent / "sweeps"


@dataclass(slots=True)
class SweepImage:
    """Single sweep image and its associated parameter values."""

    key: str
    image: np.ndarray
    parameters: Mapping[str, Any]


@dataclass(slots=True)
class SweepData:
    """Eagerly loaded sweep images and metadata."""

    sweep_name: str
    sweep_dir: Path
    images: list[SweepImage]
    metadata: Mapping[str, Any]

    @classmethod
    def load(cls, sweep_name: str = "default", *, sweep_root: Path | str | None = None) -> "SweepData":
        """Load sweep artifacts eagerly, raising when any files are missing."""

        root = Path(sweep_root) if sweep_root is not None else _DEFAULT_SWEEP_ROOT
        sweep_dir = root / sweep_name
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Sweep directory '{sweep_dir}' does not exist.")

        images_path = sweep_dir / "images.npz"
        metadata_path = sweep_dir / "metadata.json"
        if not images_path.exists():
            raise FileNotFoundError(f"Missing images archive at '{images_path}'.")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata JSON at '{metadata_path}'.")

        metadata = json.loads(metadata_path.read_text())
        parameter_lookup = _build_parameter_lookup(metadata)

        with np.load(images_path) as npz:
            image_keys: Iterable[str] = list(npz.files)
            images = [
                SweepImage(
                    key=key,
                    image=np.array(npz[key]),
                    parameters=parameter_lookup.get(key, {}),
                )
                for key in image_keys
            ]

        return cls(
            sweep_name=sweep_name,
            sweep_dir=sweep_dir,
            images=images,
            metadata=metadata,
        )


def _build_parameter_lookup(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Flatten parameter combinations by image key for quick lookup."""

    lookup: dict[str, Mapping[str, Any]] = {}
    for param_set in metadata.get("parameter_sets", []):
        for combination in param_set.get("combinations", []):
            key = combination.get("key")
            if key:
                lookup[key] = combination.get("values", {})
    return lookup


__all__ = ["SweepData", "SweepImage"]
