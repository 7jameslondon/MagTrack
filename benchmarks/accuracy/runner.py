"""Programmatic entry points for running XY-accuracy benchmarks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from matplotlib import image as mpimg

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


def _normalize_background(color: tuple[float, float, float]) -> np.ndarray:
    if len(color) != 3:
        raise ValueError("background_color must contain exactly three components")
    background = np.asarray(color, dtype=np.float32)
    if np.any(background > 1.0):
        background = background / 255.0
    background = np.clip(background, 0.0, 1.0)
    return background


def _load_png_image(path: Path, background: np.ndarray) -> np.ndarray:
    image = mpimg.imread(path)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        rgb = array[..., :3]
        alpha = array[..., 3:4]
        array = rgb * alpha + background * (1.0 - alpha)
    elif array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for montage: {array.shape}")

    if array.shape[2] > 3:
        array = array[..., :3]
    array = np.clip(array, 0.0, 1.0)
    return array


def create_accuracy_montage(
    plot_dir: Path | str | None = None,
    *,
    csv_path: Path | str | None = None,
    log_dir: Path | str | None = None,
    out_path: Path | str | None = None,
    columns: int = 2,
    padding_px: int = 10,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Create a montage image from the PNG plots generated during accuracy sweeps.

    Parameters
    ----------
    plot_dir : Path or None, optional
        Directory containing PNGs from :func:`plot_accuracy_results`. If omitted the
        latest accuracy CSV is plotted and that output directory is used.
    csv_path, log_dir : Path or None, optional
        Passed through to :func:`plot_accuracy_results` when ``plot_dir`` is not
        provided.
    out_path : Path or None, optional
        Location of the generated montage PNG. Defaults to ``plot_dir`` with the
        file name ``xy_accuracy_montage.png``.
    columns : int, optional
        Number of columns in the montage grid.
    padding_px : int, optional
        Pixel padding between plots.
    background_color : tuple, optional
        RGB color for padding/background. Values >1 are interpreted as 0-255.
    """

    if columns <= 0:
        raise ValueError("columns must be a positive integer")
    if padding_px < 0:
        raise ValueError("padding_px cannot be negative")

    if plot_dir is None:
        plot_dir_path = plot_accuracy_results(csv_path=csv_path, log_dir=log_dir)
    else:
        plot_dir_path = Path(plot_dir)

    if not plot_dir_path.exists():
        raise FileNotFoundError(f"Plot directory '{plot_dir_path}' does not exist")

    png_files = sorted(plot_dir_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(
            f"No PNG files were found in '{plot_dir_path}'. Run plot_accuracy_results first."
        )

    background = _normalize_background(background_color)
    images = [_load_png_image(path, background) for path in png_files]

    cell_height = max(image.shape[0] for image in images)
    cell_width = max(image.shape[1] for image in images)
    rows = math.ceil(len(images) / columns)

    canvas_height = rows * cell_height + padding_px * (rows + 1)
    canvas_width = columns * cell_width + padding_px * (columns + 1)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.float32)
    canvas[...] = background

    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        top = padding_px + row * (cell_height + padding_px)
        left = padding_px + col * (cell_width + padding_px)
        height, width = image.shape[:2]
        canvas[top : top + height, left : left + width, :] = image

    if out_path is None:
        out_path = plot_dir_path / "xy_accuracy_montage.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(out_path, canvas)
    return out_path


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
    "create_accuracy_montage",
    "run_full_accuracy_benchmark",
]
