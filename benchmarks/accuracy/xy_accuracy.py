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
    nm_per_px: float | Sequence[float] = 100.0,
    size_px: int | Sequence[int] = 64,
    radius_nm_choices: Tuple[float, ...] = (1500.0, 2500.0),
    background_levels: Tuple[float, ...] = (0.3, 0.8),
    contrast_scales: Tuple[float, ...] = (0.5, 1.0),
    rng_seed: int | None = 0,
    methods: List[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    method_names = _ensure_methods(methods)

    rows: List[pd.DataFrame] = []
    photons_per_unit = 5000.0
    nm_per_px_values = tuple(float(v) for v in _as_tuple(nm_per_px))
    size_px_values = tuple(int(v) for v in _as_tuple(size_px))
    radius_values = tuple(float(v) for v in radius_nm_choices)
    background_values = tuple(float(v) for v in background_levels)
    contrast_values = tuple(float(v) for v in contrast_scales)

    image_index_offset = 0

    for nm_per_px_val, size_px_val, radius_val, background_val, contrast_val in product(
        nm_per_px_values, size_px_values, radius_values, background_values, contrast_values
    ):
        x_true_px = np.empty(n_images, dtype=np.float64)
        y_true_px = np.empty(n_images, dtype=np.float64)
        z_true_nm = np.empty(n_images, dtype=np.float64)
        radius_nm_arr = np.full(n_images, radius_val, dtype=np.float64)
        background_arr = np.full(n_images, background_val, dtype=np.float64)
        contrast_arr = np.full(n_images, contrast_val, dtype=np.float64)

        xy_range_nm = size_px_val * nm_per_px_val * 0.3

        stack_list = []
        for i in range(n_images):
            x_nm = rng.uniform(-xy_range_nm, xy_range_nm)
            y_nm = rng.uniform(-xy_range_nm, xy_range_nm)
            z_nm = rng.uniform(-500.0, 500.0)
            z_true_nm[i] = z_nm
            x_true_px[i], y_true_px[i] = _true_xy_pixels(x_nm, y_nm, nm_per_px_val, size_px_val)

            xyz_nm = np.array([[x_nm, y_nm, z_nm]], dtype=np.float64)
            clean_stack = simulate_beads(
                xyz_nm,
                nm_per_px=nm_per_px_val,
                size_px=size_px_val,
                radius_nm=radius_val,
                background_level=background_val,
                contrast_scale=contrast_val,
            ).astype(np.float32)

            lam = np.clip(clean_stack * photons_per_unit, 0, None)
            noisy = rng.poisson(lam).astype(np.float32) / photons_per_unit
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
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of method names to run (default: all registered).",
    )
    args = parser.parse_args(argv)

    method_list = args.methods.split(",") if args.methods else None
    df = run_accuracy_sweep(n_images=args.n_images, methods=method_list)

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    else:
        print(_summarize(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
