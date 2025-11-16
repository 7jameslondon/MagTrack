"""Programmatic entry points for running XY-accuracy benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from benchmarks.speed import log_utils as speed_log_utils

from benchmarks.accuracy.xy_accuracy import (
    AccuracySweepConfig,
    AccuracySweepResults,
    DEFAULT_SWEEP_CONFIG,
    format_summary,
    run_accuracy_sweep,
    summarize_accuracy,
)

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


def _ensure_log_dir(log_dir: Path | str | None) -> Path:
    path = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_run_context(log_root: Path) -> dict[str, Any]:
    system_id, timestamp, metadata = speed_log_utils.collect_system_metadata()
    run_dir = log_root / system_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "system_id": system_id,
        "timestamp": timestamp,
        "metadata": metadata,
        "run_dir": run_dir,
    }


def _write_log(
    results: AccuracySweepResults,
    config: AccuracySweepConfig,
    run_context: dict[str, Any],
) -> Dict[str, Path | str | list[dict[str, Any]]]:
    slug = run_context["timestamp"]
    run_dir: Path = run_context["run_dir"]
    csv_path = run_dir / f"xy_accuracy_{slug}.csv"
    log_path = run_dir / f"xy_accuracy_{slug}.json"

    results.to_csv(csv_path)
    summary_rows = summarize_accuracy(results)
    payload = {
        "system_id": run_context["system_id"],
        "run_directory": str(run_dir.resolve()),
        "run_timestamp": run_context["metadata"].get("collected_at"),
        "config": asdict(config),
        "csv_path": str(csv_path.resolve()),
        "summary": summary_rows,
        "metadata": run_context["metadata"],
    }
    log_path.write_text(json.dumps(payload, indent=2))

    return {"csv_path": csv_path, "log_path": log_path, "summary": summary_rows}


def get_latest_csv(log_dir: Path | str | None = None) -> Path:
    """Return the newest CSV file in the accuracy log directory."""

    log_dir_path = _ensure_log_dir(log_dir)
    csv_files = sorted(
        log_dir_path.rglob("xy_accuracy_*.csv"),
        key=lambda path: path.stem.split("xy_accuracy_")[-1],
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No accuracy logs were found in '{log_dir_path}'. Run run_full_accuracy_benchmark first."
        )
    return csv_files[-1]


def run_full_accuracy_benchmark(
    config: AccuracySweepConfig | None = None,
    *,
    log_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Run the full XY-accuracy benchmark pipeline.

    The pipeline executes :func:`run_accuracy_sweep`, writes a timestamped CSV and JSON
    log, and returns the key artifacts from the run.
    """

    active_config = config or DEFAULT_SWEEP_CONFIG
    log_dir_path = _ensure_log_dir(log_dir)
    run_context = _prepare_run_context(log_dir_path)

    results = run_accuracy_sweep(**active_config.to_kwargs())
    log_info = _write_log(results, active_config, run_context)

    return {
        "csv_path": log_info["csv_path"],
        "log_path": log_info["log_path"],
        "summary": format_summary(log_info["summary"]),
    }


if __name__ == "__main__":
    run_full_accuracy_benchmark()


__all__ = [
    "get_latest_csv",
    "run_full_accuracy_benchmark",
]
