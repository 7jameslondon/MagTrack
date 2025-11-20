"""Utilities for visualizing sweep outputs stored as NPZ + metadata JSON."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class SweepImage:
    """Container for a single sweep image and its metadata."""

    key: str
    image: np.ndarray
    metadata: dict[str, Any]


def _resolve_image_key(image_path: str, available_keys: Iterable[str]) -> str:
    """Return the NPZ key corresponding to the provided image path."""

    candidates = [
        Path(image_path).stem,
        Path(image_path).name,
        image_path,
    ]
    key_set = set(available_keys)
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    raise KeyError(
        f"Could not locate NPZ entry for image_path='{image_path}'. Tried: {candidates}"
    )


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metadata_path.read_text())
    if isinstance(payload, dict) and "combinations" in payload:
        combinations = payload["combinations"]
    elif isinstance(payload, list):
        combinations = payload
    else:
        raise ValueError("metadata.json must contain a 'combinations' list")

    if not isinstance(combinations, list):
        raise ValueError("metadata combinations must be a list")
    return combinations


def load_sweep_images(
    sweep_path: Path | str,
    *,
    max_images: int | None = None,
) -> list[SweepImage]:
    """Load images and metadata entries from a sweep directory."""

    sweep_dir = Path(sweep_path)
    metadata_path = sweep_dir / "metadata.json"
    images_path = sweep_dir / "images.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Missing image archive: {images_path}")

    combinations = _load_metadata(metadata_path)
    archive = np.load(images_path)
    available_keys = archive.files

    images: list[SweepImage] = []
    for entry in combinations:
        image_path = entry.get("image_path")
        if image_path is None:
            raise KeyError("Metadata entry missing 'image_path'")
        key = _resolve_image_key(str(image_path), available_keys)
        image = archive[key]
        images.append(SweepImage(key=key, image=image, metadata=dict(entry)))
        if max_images is not None and len(images) >= max_images:
            break
    return images


def _format_title(metadata: Mapping[str, Any]) -> str:
    prioritized_keys = (
        "method",
        "radius_nm",
        "nm_per_px",
        "z_true_nm",
        "background_level",
        "contrast_scale",
    )
    parts: list[str] = []
    for key in prioritized_keys:
        if key in metadata:
            parts.append(f"{key}={metadata[key]}")
    if parts:
        return ", ".join(parts)
    image_path = metadata.get("image_path")
    return str(image_path) if image_path is not None else "image"


def plot_sweep_images(
    sweep_images: Sequence[SweepImage],
    *,
    output: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create a tiled matplotlib plot for the provided sweep images."""

    if not sweep_images:
        raise ValueError("No images were provided for plotting")

    total = len(sweep_images)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_array = np.atleast_2d(axes).reshape(rows, cols)

    for idx, sweep_image in enumerate(sweep_images):
        ax = axes_array.flat[idx]
        image = sweep_image.image
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        ax.imshow(image, cmap="gray")
        ax.set_title(_format_title(sweep_image.metadata), fontsize=9)
        ax.axis("off")

    for ax in axes_array.flat[total:]:
        ax.axis("off")

    if output is not None:
        fig.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_sweep_directory(
    sweep_path: Path | str,
    *,
    max_images: int | None = None,
    output: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Load a sweep directory and plot its contents."""

    images = load_sweep_images(sweep_path, max_images=max_images)
    return plot_sweep_images(images, output=output, show=show)


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot images from an accuracy sweep")
    parser.add_argument(
        "--sweep-path",
        type=Path,
        required=True,
        help="Path to the sweep directory containing images.npz and metadata.json",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to include in the plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the tiled figure instead of (or in addition to) showing it",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display (useful in headless environments)",
    )
    return parser.parse_args(args=args)


def main(cli_args: Sequence[str] | None = None) -> None:
    args = _parse_args(cli_args)
    plot_sweep_directory(
        args.sweep_path,
        max_images=args.max_images,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
