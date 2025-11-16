"""Programmatic entry points for running XY-accuracy benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from benchmarks.accuracy.plot_xy_accuracy import make_default_plots
from benchmarks.accuracy.xy_accuracy import (
    AccuracySweepConfig,
    AccuracySweepResults,
    DEFAULT_SWEEP_CONFIG,
    format_summary,
    run_accuracy_sweep,
    summarize_accuracy,
)

_DEFAULT_LOG_DIR = Path("benchmarks/logs/accuracy")


def _ensure_log_dir(log_dir: Path | str | None) -> Path:
    path = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_strings() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S"), now.isoformat()


def _write_log(
    results: AccuracySweepResults,
    config: AccuracySweepConfig,
    log_dir: Path,
) -> Dict[str, Path | str | list[dict[str, Any]]]:
    slug, iso_timestamp = _timestamp_strings()
    csv_path = log_dir / f"xy_accuracy_{slug}.csv"
    log_path = log_dir / f"xy_accuracy_{slug}.json"

    results.to_csv(csv_path)
    summary_rows = summarize_accuracy(results)
    payload = {
        "run_timestamp": iso_timestamp,
        "config": asdict(config),
        "csv_path": str(csv_path.resolve()),
        "summary": summary_rows,
    }
    log_path.write_text(json.dumps(payload, indent=2))

    return {"csv_path": csv_path, "log_path": log_path, "summary": summary_rows}


def get_latest_csv(log_dir: Path | str | None = None) -> Path:
    """Return the newest CSV file in the accuracy log directory."""

    log_dir_path = _ensure_log_dir(log_dir)
    csv_files = sorted(log_dir_path.glob("xy_accuracy_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No accuracy logs were found in '{log_dir_path}'. Run run_full_accuracy_benchmark first."
        )
    return csv_files[-1]


def plot_accuracy_results(
    csv_path: Path | str | None = None,
    *,
    log_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    show: bool = False,
) -> Path:
    """Generate the default plots for a CSV file (latest log if not provided)."""

    if csv_path is None:
        csv_path = get_latest_csv(log_dir)
    csv_path = Path(csv_path)
    results = AccuracySweepResults.from_csv(csv_path)

    if out_dir is None:
        out_dir = csv_path.parent / "plots" / csv_path.stem
    out_dir_path = Path(out_dir)
    make_default_plots(results, out_dir=out_dir_path, show=show)
    return out_dir_path


def run_full_accuracy_benchmark(
    config: AccuracySweepConfig | None = None,
    *,
    log_dir: Path | str | None = None,
    auto_plot: bool = True,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """Run the full XY-accuracy benchmark pipeline.

    The pipeline executes :func:`run_accuracy_sweep`, writes a timestamped CSV and JSON
    log, and (optionally) generates the default plots for the new results.
    """

    active_config = config or DEFAULT_SWEEP_CONFIG
    log_dir_path = _ensure_log_dir(log_dir)

    results = run_accuracy_sweep(**active_config.to_kwargs())
    log_info = _write_log(results, active_config, log_dir_path)

    plot_dir = None
    if auto_plot:
        plot_dir = plot_accuracy_results(
            csv_path=log_info["csv_path"],
            log_dir=log_dir_path,
            show=show_plots,
        )

    return {
        "csv_path": log_info["csv_path"],
        "log_path": log_info["log_path"],
        "summary": format_summary(log_info["summary"]),
        "plot_dir": plot_dir,
    }


if __name__ == "__main__":
    run_full_accuracy_benchmark()


__all__ = [
    "get_latest_csv",
    "plot_accuracy_results",
    "run_full_accuracy_benchmark",
]
