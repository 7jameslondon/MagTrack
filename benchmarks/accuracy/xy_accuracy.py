"""XY localization accuracy benchmark utilities."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from magtrack.core import auto_conv_sub_pixel, center_of_mass
from magtrack.simulation import simulate_beads

MethodFunc = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


METHOD_REGISTRY: Dict[str, MethodFunc] = {}
DEFAULT_CAMERA_PIXEL_SIZE_NM = 10000.0


@dataclass(frozen=True)
class AccuracySweepConfig:
    """Default configuration for :func:`run_accuracy_sweep`."""

    n_images: int = 100
    magnification_choices: Tuple[float, ...] = (100.0,)
    size_nm_choices: Tuple[float, ...] = (6400.0,)
    radius_nm_choices: Tuple[float, ...] = (1500.0, 2500.0)
    background_levels: Tuple[float, ...] = (0.3, 0.8)
    contrast_scales: Tuple[float, ...] = (0.5, 1.0)
    z_nm_choices: Tuple[float, ...] = (0.0, 1000.0)
    x_fraction_choices: Tuple[float, ...] = (0.0, 0.01)
    y_fraction_choices: Tuple[float, ...] = (0.0, 0.01)
    photons_per_unit_choices: Tuple[float, ...] = (5000.0,)
    camera_pixel_size_nm: float = DEFAULT_CAMERA_PIXEL_SIZE_NM
    rng_seed: int | None = 0
    methods: Tuple[str, ...] | None = None

    def to_kwargs(self) -> Dict[str, object]:
        """Return a dictionary of keyword arguments for :func:`run_accuracy_sweep`."""

        return {
            "n_images": self.n_images,
            "magnification_choices": self.magnification_choices,
            "size_nm_choices": self.size_nm_choices,
            "radius_nm_choices": self.radius_nm_choices,
            "background_levels": self.background_levels,
            "contrast_scales": self.contrast_scales,
            "z_nm_choices": self.z_nm_choices,
            "x_fraction_choices": self.x_fraction_choices,
            "y_fraction_choices": self.y_fraction_choices,
            "photons_per_unit_choices": self.photons_per_unit_choices,
            "camera_pixel_size_nm": self.camera_pixel_size_nm,
            "rng_seed": self.rng_seed,
            "methods": list(self.methods) if self.methods is not None else None,
        }


DEFAULT_SWEEP_CONFIG = AccuracySweepConfig()


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


def summarize_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE summaries for a sweep DataFrame."""

    summary = df.groupby("method").apply(
        lambda g: pd.Series(
            {
                "rmse_dx_nm": float(np.sqrt(np.mean(g["dx_nm"] ** 2))),
                "rmse_dy_nm": float(np.sqrt(np.mean(g["dy_nm"] ** 2))),
            }
        ),
        include_groups=False,
    )
    summary = summary.reset_index()
    return summary


def format_summary(summary_df: pd.DataFrame) -> str:
    """Return a pretty string representation of :func:`summarize_accuracy`."""

    return summary_df.to_string(index=False, float_format="{:.3f}".format)


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
    z_nm_choices: Sequence[float] | None = (0.0, 1000.0),
    x_fraction_choices: Sequence[float] | None = (0.0, 0.01),
    y_fraction_choices: Sequence[float] | None = (0.0, 0.01),
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
        z_nm_choices = (0.0, 1000.0)
    z_values = tuple(float(v) for v in _as_tuple(z_nm_choices))
    if not z_values:
        raise ValueError("z_nm_choices must contain at least one value")
    if x_fraction_choices is None:
        x_fraction_choices = (0.0, 0.01)
    x_fraction_values = tuple(float(v) for v in _as_tuple(x_fraction_choices))
    if not x_fraction_values:
        raise ValueError("x_fraction_choices must contain at least one value")
    if y_fraction_choices is None:
        y_fraction_choices = (0.0, 0.01)
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


__all__ = [
    "AccuracySweepConfig",
    "DEFAULT_CAMERA_PIXEL_SIZE_NM",
    "DEFAULT_SWEEP_CONFIG",
    "MethodFunc",
    "format_summary",
    "register_method",
    "run_accuracy_sweep",
    "summarize_accuracy",
]
