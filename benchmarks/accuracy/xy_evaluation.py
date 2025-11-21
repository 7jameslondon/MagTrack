"""Evaluate XY localization accuracy across predefined pipelines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from benchmarks.accuracy.sweep_loader import SweepData, SweepImage
from benchmarks.speed import log_utils
from magtrack import core

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "xy"


@dataclass(frozen=True)
class Pipeline:
    """Descriptor for an evaluation pipeline."""

    name: str
    run: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


PIPELINES: dict[str, Pipeline] = {
    "center_of_mass": Pipeline(
        name="center_of_mass",
        run=lambda stack: core.center_of_mass(stack),
    ),
    "center_of_mass_auto_conv": Pipeline(
        name="center_of_mass_auto_conv",
        run=lambda stack: core.auto_conv(stack, *core.center_of_mass(stack)),
    ),
}


def _ensure_stack(image: np.ndarray) -> np.ndarray:
    stack = np.asarray(image, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[:, :, np.newaxis]
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}.")
    return stack


def _extract_combinations(metadata: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    lookup: dict[str, Mapping[str, object]] = {}
    for param_set in metadata.get("parameter_sets", []):
        if not isinstance(param_set, Mapping):
            continue
        for combination in param_set.get("combinations", []):
            if not isinstance(combination, Mapping):
                continue
            key = combination.get("key")
            if isinstance(key, str):
                lookup[key] = combination
    return lookup


def _evaluate_image(
    sweep: SweepData,
    image: SweepImage,
    pipeline: Pipeline,
    *,
    combination_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    stack = _ensure_stack(image.image)
    predicted_x, predicted_y = pipeline.run(stack)

    predicted_x = float(np.asarray(predicted_x).ravel()[0])
    predicted_y = float(np.asarray(predicted_y).ravel()[0])

    parameters = dict(image.parameters)
    combo_metadata = combination_lookup.get(image.key, {})
    true_position_px = combo_metadata.get("true_position_px", {})
    true_position_nm = combo_metadata.get("true_position_nm", {})
    nm_per_px = float(combo_metadata.get("nm_per_px", parameters.get("nm_per_px", 1.0)))

    if isinstance(true_position_px, Mapping):
        truth_x_px = float(true_position_px.get("x", 0.0))
        truth_y_px = float(true_position_px.get("y", 0.0))
    else:
        size_px = int(combo_metadata.get("size_px", parameters.get("size_px", 0)))
        center_px = size_px // 2
        truth_x_px = center_px + float(parameters.get("x_offset", 0.0)) / nm_per_px
        truth_y_px = center_px + float(parameters.get("y_offset", 0.0)) / nm_per_px

    if isinstance(true_position_nm, Mapping):
        truth_x_nm = float(true_position_nm.get("x", 0.0))
        truth_y_nm = float(true_position_nm.get("y", 0.0))
        truth_z_nm = float(true_position_nm.get("z", 0.0))
    else:
        truth_x_nm = truth_x_px * nm_per_px
        truth_y_nm = truth_y_px * nm_per_px
        truth_z_nm = float(parameters.get("z_offset", 0.0))

    record: dict[str, object] = {
        "sweep_name": sweep.sweep_name,
        "image_key": image.key,
        "pipeline": pipeline.name,
        "predicted_x": predicted_x,
        "predicted_y": predicted_y,
        "truth_x": truth_x_px,
        "truth_y": truth_y_px,
        "truth_x_nm": truth_x_nm,
        "truth_y_nm": truth_y_nm,
        "truth_z_nm": truth_z_nm,
        "parameters": parameters,
    }

    if isinstance(combo_metadata, Mapping):
        for meta_key in ("size_px", "nm_per_px", "radius_nm"):
            if meta_key in combo_metadata:
                record[meta_key] = combo_metadata[meta_key]

    return record


def evaluate_sweeps(
    sweeps: Iterable[SweepData],
    pipeline_names: Iterable[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sweep in sweeps:
        combination_lookup = _extract_combinations(sweep.metadata)
        for image in sweep.images:
            for pipeline_name in pipeline_names:
                pipeline = PIPELINES[pipeline_name]
                record = _evaluate_image(
                    sweep,
                    image,
                    pipeline,
                    combination_lookup=combination_lookup,
                )
                records.append(record)
    return records


def _discover_sweeps(names: Iterable[str] | None) -> list[SweepData]:
    if names:
        targets = list(names)
    else:
        sweeps_root = Path(__file__).resolve().parent / "sweeps"
        targets = [path.name for path in sweeps_root.iterdir() if path.is_dir()]

    return [SweepData.load(name) for name in targets]


def _write_results(
    records: list[dict[str, object]],
    *,
    result_name: str,
    pipeline_names: list[str],
    sweep_names: list[str],
) -> Path:
    system_id, timestamp, system_metadata = log_utils.collect_system_metadata()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "result_name": result_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": {
            "id": system_id,
            "timestamp": timestamp,
            "metadata": system_metadata,
        },
        "pipelines": pipeline_names,
        "sweeps": sweep_names,
        "records": records,
    }

    output_path = RESULTS_DIR / f"{result_name}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XY localization accuracy.")
    parser.add_argument(
        "--sweep",
        dest="sweeps",
        action="append",
        help="Sweep name to evaluate. Defaults to all available sweeps.",
    )
    parser.add_argument(
        "--pipeline",
        dest="pipelines",
        action="append",
        choices=list(PIPELINES.keys()),
        help="Pipeline name to run. Defaults to all pipelines.",
    )
    parser.add_argument(
        "--result-name",
        default="default",
        help="Filename stem for stored results (without extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline_names = list(args.pipelines) if args.pipelines else list(PIPELINES.keys())
    sweeps = _discover_sweeps(args.sweeps)
    sweep_names = [sweep.sweep_name for sweep in sweeps]

    records = evaluate_sweeps(sweeps, pipeline_names)
    output_path = _write_results(
        records,
        result_name=args.result_name,
        pipeline_names=pipeline_names,
        sweep_names=sweep_names,
    )
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
