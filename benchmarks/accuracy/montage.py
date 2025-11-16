"""Visualization helpers for XY-accuracy bead images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np

from benchmarks.accuracy.xy_accuracy import AccuracySweepConfig, DEFAULT_SWEEP_CONFIG
from magtrack.simulation import simulate_beads


@dataclass(frozen=True)
class BeadImage:
    """Container that stores a simulated bead image and sweep metadata."""

    data: np.ndarray
    metadata: dict[str, Any]


def _choice_tuple(values: Sequence[float] | None, fallback: Iterable[float]) -> tuple[float, ...]:
    if values is None:
        return tuple(fallback)
    resolved = tuple(values)
    if not resolved:
        return tuple(fallback)
    return resolved


def generate_accuracy_benchmark_images(
    config: AccuracySweepConfig | None = None,
    *,
    max_images: int | None = None,
) -> list[BeadImage]:
    """Return simulated bead images used by the accuracy sweep."""

    cfg = config or DEFAULT_SWEEP_CONFIG
    rng = np.random.default_rng(cfg.rng_seed)

    magnifications = _choice_tuple(cfg.magnification_choices, (100.0,))
    size_nm_values = _choice_tuple(cfg.size_nm_choices, (6400.0,))
    radius_values = _choice_tuple(cfg.radius_nm_choices, (1500.0,))
    background_values = _choice_tuple(cfg.background_levels, (0.3,))
    contrast_values = _choice_tuple(cfg.contrast_scales, (0.5,))
    photons_values = _choice_tuple(cfg.photons_per_unit_choices, (5000.0,))
    z_values = _choice_tuple(cfg.z_nm_choices, (0.0,))
    x_fraction_values = _choice_tuple(cfg.x_fraction_choices, (0.0,))
    y_fraction_values = _choice_tuple(cfg.y_fraction_choices, (0.0,))

    images: list[BeadImage] = []
    image_index_offset = 0

    for (
        magnification_val,
        size_nm_val,
        radius_val,
        background_val,
        contrast_val,
        photons_val,
        z_choice,
        x_fraction,
        y_fraction,
    ) in product(
        magnifications,
        size_nm_values,
        radius_values,
        background_values,
        contrast_values,
        photons_values,
        z_values,
        x_fraction_values,
        y_fraction_values,
    ):
        nm_per_px_val = cfg.camera_pixel_size_nm / magnification_val
        size_px_val = max(int(round(size_nm_val / nm_per_px_val)), 1)

        xyz_nm = np.array([[x_fraction * size_nm_val, y_fraction * size_nm_val, z_choice]], dtype=np.float64)
        clean_stack = simulate_beads(
            xyz_nm,
            nm_per_px=nm_per_px_val,
            size_px=size_px_val,
            radius_nm=radius_val,
            background_level=background_val,
            contrast_scale=contrast_val,
        ).astype(np.float32)
        lam = np.clip(clean_stack * photons_val, 0, None)

        for local_idx in range(cfg.n_images):
            noisy = rng.poisson(lam).astype(np.float32) / photons_val
            metadata = {
                "image_index": image_index_offset + local_idx,
                "radius_nm": radius_val,
                "background_level": background_val,
                "contrast_scale": contrast_val,
                "z_true_nm": z_choice,
                "nm_per_px": nm_per_px_val,
                "size_px": size_px_val,
                "size_nm": size_nm_val,
                "magnification": magnification_val,
                "photons_per_unit": photons_val,
                "x_fraction_of_size": x_fraction,
                "y_fraction_of_size": y_fraction,
            }
            images.append(BeadImage(data=noisy[:, :, 0], metadata=metadata))
            if max_images is not None and len(images) >= max_images:
                return images

        image_index_offset += cfg.n_images

    return images


def show_accuracy_montage(
    config: AccuracySweepConfig | None = None,
    *,
    max_images: int | None = 64,
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Render a matplotlib montage of bead images using the Tk backend."""

    images = generate_accuracy_benchmark_images(config=config, max_images=max_images)
    if not images:
        return

    matplotlib.use("TkAgg", force=False)
    import matplotlib.pyplot as plt  # Imported lazily to honor backend selection.

    total = len(images)
    cols = max(math.ceil(math.sqrt(total)), 1)
    rows = math.ceil(total / cols)
    if figsize is None:
        figsize = (cols * 1.5, rows * 1.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_iter = np.atleast_1d(axes).ravel()

    for ax, bead in zip(axes_iter, images):
        ax.imshow(bead.data, cmap=cmap, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"idx {bead.metadata.get('image_index', '?')}\nrad {bead.metadata.get('radius_nm', 0.0):.0f}nm",
            fontsize=6,
        )

    for ax in axes_iter[total:]:
        ax.axis("off")

    fig.suptitle("Accuracy sweep bead images", fontsize=12)
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Convenience CLI entry point that shows the default montage."""

    show_accuracy_montage()


__all__ = [
    "BeadImage",
    "generate_accuracy_benchmark_images",
    "show_accuracy_montage",
]
