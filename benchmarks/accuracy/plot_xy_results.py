"""Plot XY evaluation results as bar charts per parameter set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import ceil, sqrt
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.accuracy import xy_evaluation


def _load_results(result_name: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    results_path = xy_evaluation.RESULTS_DIR / f"{result_name}.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Result file '{result_name}' not found at {results_path}. Generate it with "
            "`python -m benchmarks.accuracy.xy_evaluation`."
        )

    payload = json.loads(results_path.read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Results file {results_path} is missing 'records' list.")

    return payload, records


def _group_by_parameters(records: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        image_key = record.get("image_key")
        sweep_name = record.get("sweep_name", "")
        if not isinstance(image_key, str):
            continue
        label_parts = [image_key]
        size_px = record.get("size_px")
        if size_px is not None:
            label_parts.append(f"size_px={size_px}")
        radius_nm = record.get("radius_nm")
        if radius_nm is not None:
            label_parts.append(f"radius_nm={radius_nm}")
        label = f"{sweep_name}: " + ", ".join(label_parts)
        grouped[label].append(record)
    return grouped


def _ensure_pipeline_records(
    pipeline_names: Sequence[str],
    records: Sequence[Mapping[str, object]],
    group_label: str,
) -> list[Mapping[str, object]]:
    indexed = {record.get("pipeline"): record for record in records if isinstance(record, Mapping)}
    missing = [name for name in pipeline_names if name not in indexed]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            f"Group '{group_label}' is missing records for pipeline(s): {missing_names}."
        )
    return [indexed[name] for name in pipeline_names]


def _euclidean_error(record: Mapping[str, object]) -> float:
    predicted_x = float(record.get("predicted_x", 0.0))
    predicted_y = float(record.get("predicted_y", 0.0))
    truth_x = float(record.get("truth_x", 0.0))
    truth_y = float(record.get("truth_y", 0.0))
    return float(np.hypot(predicted_x - truth_x, predicted_y - truth_y))


def _compute_grid(n_plots: int) -> tuple[int, int]:
    cols = max(1, ceil(sqrt(n_plots)))
    rows = max(1, ceil(n_plots / cols))
    return rows, cols


def _plot_groups(
    grouped_records: Mapping[str, list[Mapping[str, object]]],
    pipeline_names: Sequence[str],
    *,
    result_name: str,
) -> None:
    group_items = list(grouped_records.items())
    if not group_items:
        raise ValueError("No records available to plot.")

    rows, cols = _compute_grid(len(group_items))
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(4 * cols, 3 * rows))
    fig.suptitle(f"XY Evaluation Results: {result_name}")

    for ax, (group_label, records) in zip(axes.flatten(), group_items):
        ordered_records = _ensure_pipeline_records(pipeline_names, records, group_label)
        errors = [_euclidean_error(record) for record in ordered_records]
        ax.bar(pipeline_names, errors, color="tab:blue")
        ax.set_title(group_label, fontsize=10)
        ax.set_ylabel("Euclidean error (px)")
        ax.set_xticklabels(pipeline_names, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for ax in axes.flatten()[len(group_items):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot XY evaluation results from benchmarks.accuracy.xy_evaluation. "
            "Displays bar charts of Euclidean error per pipeline for each parameter set."
        )
    )
    parser.add_argument(
        "--result",
        default="default",
        help="Result name to load from benchmarks/accuracy/results/xy/<name>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload, records = _load_results(args.result)
    pipeline_names = payload.get("pipelines")

    if not isinstance(pipeline_names, list) or not pipeline_names:
        raise ValueError("Results file must include a non-empty 'pipelines' list.")

    grouped_records = _group_by_parameters(records)
    _plot_groups(grouped_records, pipeline_names, result_name=args.result)


if __name__ == "__main__":
    main()
