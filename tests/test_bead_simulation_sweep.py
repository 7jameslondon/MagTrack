from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmarks.accuracy.bead_simulation_sweep import (
    BeadSimulationSweep,
    ParameterSet,
)


def test_generate_creates_npz_and_metadata(tmp_path: Path) -> None:
    parameter_set = ParameterSet(
        name="default",
        parameters={
            "x_offset": [-1.0, 1.0],
            "y_offset": [-2.0, 2.0],
            "z_offset": [-3.0, 3.0],
            "background": [0.0, 5.0],
            "seed": [111, 222],
        },
        image_shape=(8, 8, 1),
    )
    sweep = BeadSimulationSweep("test_sweep", [parameter_set], sweep_root=tmp_path)
    artifact = sweep.generate()

    assert artifact.images_path.is_file()
    assert artifact.metadata_path.is_file()

    loaded = np.load(artifact.images_path)
    expected_images = 2 ** 5
    assert len(loaded.files) == expected_images
    for key in loaded.files:
        array = loaded[key]
        assert array.shape == (8, 8, 1)
        assert array.dtype == np.float64

    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["image_file"] == artifact.images_path.name
    parameter_sets = metadata["parameter_sets"]
    assert len(parameter_sets) == 1
    combinations = parameter_sets[0]["combinations"]
    assert len(combinations) == expected_images
    for combo in combinations:
        assert combo["image_path"] == f"{artifact.images_path.name}::{combo['key']}"
    assert metadata["total_images"] == expected_images


def test_overwrite_flag_controls_replacement(tmp_path: Path) -> None:
    sweep = BeadSimulationSweep("test_overwrite", sweep_root=tmp_path)
    sweep.generate()

    # Second run should fail without overwrite.
    try:
        sweep.generate()
        assert False, "expected FileExistsError"
    except FileExistsError:
        pass

    # Overwrite should succeed.
    artifact = sweep.generate(overwrite=True)
    assert artifact.images_path.is_file()
    assert artifact.metadata_path.is_file()


def test_cli_generates_default_sweep(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.accuracy.bead_simulation_sweep",
            "--sweep-name",
            "cli_sweep",
            "--sweep-root",
            str(tmp_path),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    sweep_dir = tmp_path / "cli_sweep"
    images_path = sweep_dir / "images.npz"
    metadata_path = sweep_dir / "metadata.json"

    assert images_path.is_file(), result.stdout + result.stderr
    assert metadata_path.is_file(), result.stdout + result.stderr
