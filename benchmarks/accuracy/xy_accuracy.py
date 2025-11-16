"""XY localization accuracy benchmark utilities."""

from __future__ import annotations

import argparse
from collections.abc import Iterable as IterableABC
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from magtrack.core import auto_conv_sub_pixel, center_of_mass
from magtrack.simulation import simulate_beads

MethodFunc = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


METHOD_REGISTRY: Dict[str, MethodFunc] = {}
DEFAULT_CAMERA_PIXEL_SIZE_NM = 10000.0


def register_method(name: str) -> Callable[[MethodFunc], MethodFunc]:
    """Decorator to register localization methods."""

    def decorator(func: MethodFunc) -> MethodFunc:
        METHOD_REGISTRY[name] = func
        return func

    return decorator


@register_method("com_xy")
def _center_of_mass_xy(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate coordinates via center-of-mass with mean background subtraction."""

    x_px, y_px = center_of_mass(stack, background="mean")
    return np.asarray(x_px), np.asarray(y_px)


@register_method("com_autoconv_xy")
def _com_autoconv_xy(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-of-mass seed followed by auto-convolution sub-pixel refinement."""

    x0_px, y0_px = center_of_mass(stack, background="mean")
    x_px, y_px = auto_conv_sub_pixel(stack, x0_px, y0_px)
    return np.asarray(x_px), np.asarray(y_px)


def _ensure_methods(methods: Iterable[str] | None) -> List[str]:
    if methods is None:
        return list(METHOD_REGISTRY.keys())
    unknown = [name for name in methods if name not in METHOD_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown methods requested: {', '.join(unknown)}")
    return list(methods)


def _true_xy_pixels(x_nm: float, y_nm: float, nm_per_px: float, size_px: int) -> Tuple[float, float]:
    center = size_px // 2
    x_px = center + x_nm / nm_per_px
    y_px = center + y_nm / nm_per_px
    return x_px, y_px


def _summarize(df: pd.DataFrame) -> str:
    summary = df.groupby("method").apply(
        lambda g: pd.Series(
            {
                "rmse_dx_nm": float(np.sqrt(np.mean(g["dx_nm"] ** 2))),
                "rmse_dy_nm": float(np.sqrt(np.mean(g["dy_nm"] ** 2))),
            }
        ),
        include_groups=False,
    )
    return summary.to_string(float_format="{:.3f}".format)


def _as_tuple(value: float | int | Sequence[float | int]) -> Tuple[float | int, ...]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, IterableABC):
        return tuple(value)
    return (value,)


def run_accuracy_sweep(
    n_images: int = 100,
    magnification_choices: float | Sequence[float] = 100.0,
    size_nm_choices: float | Sequence[float] = 6400.0,
    radius_nm_choices: Tuple[float, ...] = (1500.0, 2500.0),
    background_levels: Tuple[float, ...] = (0.3, 0.8),
    contrast_scales: Tuple[float, ...] = (0.5, 1.0),
    z_nm_choices: Sequence[float] | None = None,
    x_fraction_choices: Sequence[float] | None = None,
    y_fraction_choices: Sequence[float] | None = None,
    photons_per_unit_choices: float | Sequence[float] = 5000.0,
    camera_pixel_size_nm: float = DEFAULT_CAMERA_PIXEL_SIZE_NM,
    rng_seed: int | None = 0,
    methods: List[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    method_names = _ensure_methods(methods)

    rows: List[pd.DataFrame] = []
    if camera_pixel_size_nm <= 0:
        raise ValueError("camera_pixel_size_nm must be positive")
    magnification_values = tuple(float(v) for v in _as_tuple(magnification_choices))
    if not magnification_values:
        raise ValueError("magnification_choices must contain at least one value")
    size_nm_values = tuple(float(v) for v in _as_tuple(size_nm_choices))
    if not size_nm_values:
        raise ValueError("size_nm_choices must contain at least one value")
    radius_values = tuple(float(v) for v in radius_nm_choices)
    background_values = tuple(float(v) for v in background_levels)
    contrast_values = tuple(float(v) for v in contrast_scales)
    photons_values = tuple(float(v) for v in _as_tuple(photons_per_unit_choices))
    if z_nm_choices is None:
        raise ValueError("z_nm_choices must be specified")
    z_values = tuple(float(v) for v in _as_tuple(z_nm_choices))
    if not z_values:
        raise ValueError("z_nm_choices must contain at least one value")
    if x_fraction_choices is None:
        raise ValueError("x_fraction_choices must be specified")
    x_fraction_values = tuple(float(v) for v in _as_tuple(x_fraction_choices))
    if not x_fraction_values:
        raise ValueError("x_fraction_choices must contain at least one value")
    if y_fraction_choices is None:
        raise ValueError("y_fraction_choices must be specified")
    y_fraction_values = tuple(float(v) for v in _as_tuple(y_fraction_choices))
    if not y_fraction_values:
        raise ValueError("y_fraction_choices must contain at least one value")

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
        magnification_values,
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
        nm_per_px_val = camera_pixel_size_nm / magnification_val
        size_px_exact = size_nm_val / nm_per_px_val
        size_px_val = int(round(size_px_exact))
        if size_px_val <= 0:
            raise ValueError("Derived size_px must be positive. Adjust magnification or size_nm choices.")
        if abs(size_px_exact - size_px_val) > 1e-6:
            raise ValueError(
                "size_nm value {size_nm} is incompatible with magnification {mag}, "
                "resulting in a non-integer pixel width".format(
                    size_nm=size_nm_val, mag=magnification_val
                )
            )

        x_true_px = np.empty(n_images, dtype=np.float64)
        y_true_px = np.empty(n_images, dtype=np.float64)
        z_true_nm = np.empty(n_images, dtype=np.float64)
        radius_nm_arr = np.full(n_images, radius_val, dtype=np.float64)
        background_arr = np.full(n_images, background_val, dtype=np.float64)
        contrast_arr = np.full(n_images, contrast_val, dtype=np.float64)
        x_fraction_arr = np.full(n_images, x_fraction, dtype=np.float64)
        y_fraction_arr = np.full(n_images, y_fraction, dtype=np.float64)

        x_nm = x_fraction * size_nm_val
        y_nm = y_fraction * size_nm_val
        x_px_val, y_px_val = _true_xy_pixels(x_nm, y_nm, nm_per_px_val, size_px_val)
        x_true_px.fill(x_px_val)
        y_true_px.fill(y_px_val)
        z_true_nm.fill(z_choice)

        xyz_nm = np.array([[x_nm, y_nm, z_choice]], dtype=np.float64)
        clean_stack = simulate_beads(
            xyz_nm,
            nm_per_px=nm_per_px_val,
            size_px=size_px_val,
            radius_nm=radius_val,
            background_level=background_val,
            contrast_scale=contrast_val,
        ).astype(np.float32)

        stack_list = []
        lam = np.clip(clean_stack * photons_val, 0, None)
        for _ in range(n_images):
            noisy = rng.poisson(lam).astype(np.float32) / photons_val
            stack_list.append(noisy)

        stack = np.concatenate(stack_list, axis=2)

        for method in method_names:
            estimator = METHOD_REGISTRY[method]
            x_est_px, y_est_px = estimator(stack)
            x_est_px = np.asarray(x_est_px, dtype=np.float64)
            y_est_px = np.asarray(y_est_px, dtype=np.float64)

            dx_px = x_est_px - x_true_px
            dy_px = y_est_px - y_true_px
            dx_nm = dx_px * nm_per_px_val
            dy_nm = dy_px * nm_per_px_val

            rows.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "image_index": np.arange(n_images, dtype=int) + image_index_offset,
                        "x_true_px": x_true_px,
                        "y_true_px": y_true_px,
                        "x_est_px": x_est_px,
                        "y_est_px": y_est_px,
                        "dx_px": dx_px,
                        "dy_px": dy_px,
                        "dx_nm": dx_nm,
                        "dy_nm": dy_nm,
                        "radius_nm": radius_nm_arr,
                        "background_level": background_arr,
                        "contrast_scale": contrast_arr,
                        "z_true_nm": z_true_nm,
                        "nm_per_px": np.full(n_images, nm_per_px_val, dtype=np.float64),
                        "size_px": np.full(n_images, size_px_val, dtype=np.int64),
                        "size_nm": np.full(n_images, size_nm_val, dtype=np.float64),
                        "magnification": np.full(n_images, magnification_val, dtype=np.float64),
                        "photons_per_unit": np.full(n_images, photons_val, dtype=np.float64),
                        "x_fraction_of_size": x_fraction_arr,
                        "y_fraction_of_size": y_fraction_arr,
                    }
                )
            )

        image_index_offset += n_images

    return pd.concat(rows, ignore_index=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run XY localization accuracy benchmark.")
    parser.add_argument("--n-images", type=int, default=100, help="Number of images to simulate.")
    parser.add_argument("--out", type=str, default=None, help="Optional path to CSV output.")
    parser.add_argument(
        "--magnification-choices",
        type=str,
        default="100.0",
        help="Comma-separated magnifications to test (default: 100).",
    )
    parser.add_argument(
        "--size-nm-choices",
        type=str,
        default="6400.0",
        help="Comma-separated physical ROI sizes in nanometers (default: 6400).",
    )
    parser.add_argument(
        "--camera-pixel-size-nm",
        type=float,
        default=DEFAULT_CAMERA_PIXEL_SIZE_NM,
        help="Physical camera pixel size in nanometers (default: 10000).",
    )
    parser.add_argument(
        "--z-nm-choices",
        type=str,
        required=True,
        help="Comma-separated list of z positions in nanometers (required).",
    )
    parser.add_argument(
        "--x-fraction-choices",
        type=str,
        required=True,
        help="Comma-separated fractions of size_px specifying x offsets from the center.",
    )
    parser.add_argument(
        "--y-fraction-choices",
        type=str,
        required=True,
        help="Comma-separated fractions of size_px specifying y offsets from the center.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of method names to run (default: all registered).",
    )
    args = parser.parse_args(argv)

    method_list = args.methods.split(",") if args.methods else None
    magnifications = tuple(
        float(part) for part in args.magnification_choices.split(",") if part.strip()
    )
    if not magnifications:
        raise SystemExit("--magnification-choices must include at least one numeric value")
    size_nm_choices = tuple(
        float(part) for part in args.size_nm_choices.split(",") if part.strip()
    )
    if not size_nm_choices:
        raise SystemExit("--size-nm-choices must include at least one numeric value")
    z_choices = tuple(float(part) for part in args.z_nm_choices.split(",") if part.strip())
    if not z_choices:
        raise SystemExit("--z-nm-choices must include at least one numeric value")
    x_fractions = tuple(float(part) for part in args.x_fraction_choices.split(",") if part.strip())
    if not x_fractions:
        raise SystemExit("--x-fraction-choices must include at least one numeric value")
    y_fractions = tuple(float(part) for part in args.y_fraction_choices.split(",") if part.strip())
    if not y_fractions:
        raise SystemExit("--y-fraction-choices must include at least one numeric value")
    df = run_accuracy_sweep(
        n_images=args.n_images,
        methods=method_list,
        magnification_choices=magnifications,
        size_nm_choices=size_nm_choices,
        camera_pixel_size_nm=args.camera_pixel_size_nm,
        z_nm_choices=z_choices,
        x_fraction_choices=x_fractions,
        y_fraction_choices=y_fractions,
    )

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    else:
        print(_summarize(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
