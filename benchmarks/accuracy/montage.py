"""Visualization helpers for XY-accuracy bead images."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
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


def _as_tuple(value: float | int | Sequence[float | int]) -> tuple[float | int, ...]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _resolve_choices(
    values: float | Sequence[float] | None,
    *,
    fallback: Iterable[float] | None,
    name: str,
) -> tuple[float, ...]:
    if values is None:
        if fallback is None:
            raise ValueError(f"{name} must contain at least one value")
        values = tuple(float(v) for v in fallback)
    resolved = tuple(float(v) for v in _as_tuple(values))
    if not resolved:
        raise ValueError(f"{name} must contain at least one value")
    return resolved


def generate_accuracy_benchmark_images(
    config: AccuracySweepConfig | None = None,
    *,
    max_images: int | None = None,
) -> list[BeadImage]:
    """Return simulated bead images used by the accuracy sweep."""

    cfg = config or DEFAULT_SWEEP_CONFIG
    if cfg.n_images <= 0:
        raise ValueError("AccuracySweepConfig.n_images must be positive")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")

    rng = np.random.default_rng(cfg.rng_seed)

    magnifications = _resolve_choices(cfg.magnification_choices, fallback=None, name="magnification_choices")
    size_nm_values = _resolve_choices(cfg.size_nm_choices, fallback=None, name="size_nm_choices")
    radius_values = _resolve_choices(cfg.radius_nm_choices, fallback=None, name="radius_nm_choices")
    background_values = _resolve_choices(cfg.background_levels, fallback=None, name="background_levels")
    contrast_values = _resolve_choices(cfg.contrast_scales, fallback=None, name="contrast_scales")
    photons_values = _resolve_choices(cfg.photons_per_unit_choices, fallback=None, name="photons_per_unit_choices")
    z_values = _resolve_choices(cfg.z_nm_choices, fallback=(0.0, 1000.0), name="z_nm_choices")
    x_fraction_values = _resolve_choices(cfg.x_fraction_choices, fallback=(0.0, 0.01), name="x_fraction_choices")
    y_fraction_values = _resolve_choices(cfg.y_fraction_choices, fallback=(0.0, 0.01), name="y_fraction_choices")

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
    ) in _cartesian_product(
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
        if magnification_val <= 0:
            raise ValueError("magnification choices must be positive")
        nm_per_px_val = cfg.camera_pixel_size_nm / magnification_val
        if nm_per_px_val <= 0:
            raise ValueError("camera_pixel_size_nm must be positive")

        size_px_exact = size_nm_val / nm_per_px_val
        size_px_val = int(round(size_px_exact))
        if size_px_val <= 0:
            raise ValueError(
                "Derived size_px must be positive. Adjust magnification or size_nm choices."
            )
        if abs(size_px_exact - size_px_val) > 1e-6:
            raise ValueError(
                "size_nm value {size_nm} is incompatible with magnification {mag}, resulting in a non-integer pixel width".format(
                    size_nm=size_nm_val,
                    mag=magnification_val,
                )
            )

        x_nm = x_fraction * size_nm_val
        y_nm = y_fraction * size_nm_val

        xyz_nm = np.array([[x_nm, y_nm, z_choice]], dtype=np.float64)
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
        raise RuntimeError("No bead images were generated")

    if matplotlib.get_backend().lower() != "tkagg":
        matplotlib.use("TkAgg", force=True)

    import matplotlib.pyplot as plt  # Imported lazily to honor backend selection.

    total = len(images)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    if figsize is None:
        figsize = (cols * 1.5, rows * 1.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_iter = np.atleast_1d(axes).ravel()

    for ax, bead in zip(axes_iter, images):
        ax.imshow(bead.data, cmap=cmap, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            "idx {idx}\nrad {radius:.0f}nm bg {bg:.2f}".format(
                idx=bead.metadata.get("image_index", "?"),
                radius=bead.metadata.get("radius_nm", 0.0),
                bg=bead.metadata.get("background_level", 0.0),
            ),
            fontsize=6,
        )

    for ax in axes_iter[total:]:
        ax.axis("off")

    fig.suptitle("Accuracy sweep bead images", fontsize=12)
    fig.tight_layout()
    plt.show()


def _cartesian_product(*arrays: Iterable[float]) -> Iterable[tuple[float, ...]]:
    if not arrays:
        return []
    from itertools import product

    return product(*arrays)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize accuracy benchmark bead images.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=64,
        help="Limit the number of images shown in the montage (default: 64).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Override the RNG seed used for bead noise generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = DEFAULT_SWEEP_CONFIG
    if args.rng_seed is not None:
        config = replace(config, rng_seed=args.rng_seed)
    show_accuracy_montage(config=config, max_images=args.max_images)


if __name__ == "__main__":
    main()


__all__ = [
    "BeadImage",
    "generate_accuracy_benchmark_images",
    "show_accuracy_montage",
]
