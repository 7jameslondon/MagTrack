"""Visualize simulated bead images from the accuracy sweep as a montage."""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib


def _configure_tk_backend() -> None:
    """Force Matplotlib to use TkAgg and fail fast if Tk is unavailable."""

    backend = "TkAgg"
    try:
        import tkinter  # type: ignore # pylint: disable=unused-import
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise SystemExit(
            "TkAgg requires Tkinter but it is not installed. Install Tk/Tcl "
            "support (e.g., `python -m pip install tk` on Windows) and re-run."
        ) from exc

    try:
        matplotlib.use(backend, force=True)
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - backend specific
        raise SystemExit(
            f"Unable to activate Matplotlib backend '{backend}'. "
            "Verify that Tkinter is available and working."
        ) from exc


_configure_tk_backend()

import matplotlib.pyplot as plt

from benchmarks.accuracy.xy_accuracy import (
    DEFAULT_SWEEP_CONFIG,
    create_accuracy_sweep_montage,
)


_DEF_RNG_SEED = 0 if DEFAULT_SWEEP_CONFIG.rng_seed is None else DEFAULT_SWEEP_CONFIG.rng_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the noisy bead images synthesized during an accuracy sweep "
            "as a Matplotlib montage."
        )
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=DEFAULT_SWEEP_CONFIG.n_images,
        help=(
            "Number of noisy frames simulated per sweep configuration "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=_DEF_RNG_SEED,
        help="Seed for the NumPy random generator (default: %(default)s).",
    )
    parser.add_argument(
        "--tile-columns",
        type=int,
        default=10,
        help="Number of tiles to place in each row of the montage (default: %(default)s).",
    )
    parser.add_argument(
        "--tile-spacing",
        type=int,
        default=2,
        help="Pixel spacing between tiles (default: %(default)s).",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help=(
            "Pixel value used for padding and spacing regions within the montage "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=10.0,
        help="Width of the Matplotlib figure in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=8.0,
        help="Height of the Matplotlib figure in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Accuracy sweep bead montage",
        help="Title to display above the montage (default: %(default)s).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap used for rendering the montage (default: %(default)s).",
    )
    return parser.parse_args()


def _montage_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    sweep_kwargs = DEFAULT_SWEEP_CONFIG.to_kwargs()
    sweep_kwargs.pop("methods", None)
    sweep_kwargs["n_images"] = args.n_images
    sweep_kwargs["rng_seed"] = args.rng_seed
    sweep_kwargs.update(
        {
            "tile_columns": args.tile_columns,
            "tile_spacing_px": args.tile_spacing,
            "fill_value": args.fill_value,
        }
    )
    return sweep_kwargs


def main() -> None:
    args = _parse_args()
    montage = create_accuracy_sweep_montage(**_montage_kwargs(args))

    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_height))
    ax.imshow(montage, cmap=args.cmap, interpolation="nearest")
    ax.set_title(args.title)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
