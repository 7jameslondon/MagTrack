"""XY localization accuracy benchmark utilities."""

from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from magtrack.core import auto_conv_sub_pixel, center_of_mass
from magtrack.simulation import simulate_beads

MethodFunc = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


METHOD_REGISTRY: Dict[str, MethodFunc] = {}
DEFAULT_CAMERA_PIXEL_SIZE_NM = 10000.0

SWEEP_COLUMN_ORDER: Tuple[str, ...] = (
    "method",
    "image_index",
    "x_true_px",
    "y_true_px",
    "x_est_px",
    "y_est_px",
    "dx_px",
    "dy_px",
    "dx_nm",
    "dy_nm",
    "radius_nm",
    "background_level",
    "contrast_scale",
    "z_true_nm",
    "nm_per_px",
    "size_px",
    "size_nm",
    "magnification",
    "photons_per_unit",
    "x_fraction_of_size",
    "y_fraction_of_size",
)

SWEEP_COLUMN_DTYPES: Dict[str, Any] = {
    "method": object,
    "image_index": np.int64,
    "x_true_px": np.float64,
    "y_true_px": np.float64,
    "x_est_px": np.float64,
    "y_est_px": np.float64,
    "dx_px": np.float64,
    "dy_px": np.float64,
    "dx_nm": np.float64,
    "dy_nm": np.float64,
    "radius_nm": np.float64,
    "background_level": np.float64,
    "contrast_scale": np.float64,
    "z_true_nm": np.float64,
    "nm_per_px": np.float64,
    "size_px": np.int64,
    "size_nm": np.float64,
    "magnification": np.float64,
    "photons_per_unit": np.float64,
    "x_fraction_of_size": np.float64,
    "y_fraction_of_size": np.float64,
}

SWEEP_COLUMN_TYPES: Dict[str, type] = {
    "method": str,
    "image_index": int,
    "x_true_px": float,
    "y_true_px": float,
    "x_est_px": float,
    "y_est_px": float,
    "dx_px": float,
    "dy_px": float,
    "dx_nm": float,
    "dy_nm": float,
    "radius_nm": float,
    "background_level": float,
    "contrast_scale": float,
    "z_true_nm": float,
    "nm_per_px": float,
    "size_px": int,
    "size_nm": float,
    "magnification": float,
    "photons_per_unit": float,
    "x_fraction_of_size": float,
    "y_fraction_of_size": float,
}


def _to_numpy_column(values: Sequence[Any], dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


class AccuracySweepResults:
    """Column-oriented container for accuracy sweep outputs."""

    __slots__ = ("_data", "_row_count", "_column_order")

    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
        column_order: Sequence[str] | None = None,
    ) -> None:
        if not data:
            raise ValueError("AccuracySweepResults requires at least one column")
        if column_order is None:
            column_order = tuple(data.keys())

        ordered: "OrderedDict[str, np.ndarray]" = OrderedDict()
        row_count: int | None = None
        seen: set[str] = set()
        for name in column_order:
            if name not in data:
                raise KeyError(f"Column '{name}' not found in data")
            column = _to_numpy_column(data[name])
            if row_count is None:
                row_count = column.shape[0]
            elif column.shape[0] != row_count:
                raise ValueError("All columns must have the same number of rows")
            ordered[name] = column
            seen.add(name)

        for name, values in data.items():
            if name in seen:
                continue
            column = _to_numpy_column(values)
            if row_count is None:
                row_count = column.shape[0]
            elif column.shape[0] != row_count:
                raise ValueError("All columns must have the same number of rows")
            ordered[name] = column
            seen.add(name)
            column_order = tuple(list(column_order) + [name])

        if row_count is None:
            row_count = 0

        self._data = ordered
        self._row_count = row_count
        self._column_order = tuple(column_order)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __len__(self) -> int:
        return self._row_count

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self) -> Iterable[tuple[str, np.ndarray]]:
        return self._data.items()

    def values(self) -> Iterable[np.ndarray]:
        return self._data.values()

    @property
    def columns(self) -> Tuple[str, ...]:
        return self._column_order

    @property
    def row_count(self) -> int:
        return self._row_count

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        for row_values in zip(*self._data.values()):
            yield {
                name: _python_scalar(value)
                for name, value in zip(self._data.keys(), row_values)
            }

    def to_csv(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self._data.keys()))
            writer.writeheader()
            for row in self.iter_rows():
                writer.writerow(row)

    def as_dict(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        if copy:
            return {key: value.copy() for key, value in self._data.items()}
        return dict(self._data)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        column_order: Sequence[str] | None = None,
    ) -> "AccuracySweepResults":
        if not rows:
            raise ValueError("rows must contain at least one entry")
        if column_order is None:
            ordered: list[str] = []
            for row in rows:
                for key in row:
                    if key not in ordered:
                        ordered.append(key)
            column_order = tuple(ordered)
        data: MutableMapping[str, list[Any]] = OrderedDict((name, []) for name in column_order)
        for row in rows:
            for name in column_order:
                if name not in row:
                    raise KeyError(f"Row is missing required column '{name}'")
                data[name].append(row[name])
        arrays = {name: _to_numpy_column(values) for name, values in data.items()}
        return cls(arrays, column_order=tuple(column_order))

    @classmethod
    def from_csv(cls, path: Path | str) -> "AccuracySweepResults":
        path = Path(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("CSV file is missing a header row")
            columns: MutableMapping[str, list[Any]] = OrderedDict((name, []) for name in fieldnames)
            for row in reader:
                for name in fieldnames:
                    raw_value = row.get(name, "")
                    converter = SWEEP_COLUMN_TYPES.get(name, float)
                    columns[name].append(_convert_csv_value(raw_value, converter))
        arrays = {name: _to_numpy_column(values, SWEEP_COLUMN_DTYPES.get(name)) for name, values in columns.items()}
        return cls(arrays, column_order=tuple(fieldnames))


def _convert_csv_value(value: str, converter: type) -> Any:
    if converter is str:
        return value
    if converter is int:
        return int(value)
    if converter is float:
        return float(value)
    return converter(value)


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


def _get_column(
    table: AccuracySweepResults | Mapping[str, Sequence[Any]],
    column: str,
) -> np.ndarray:
    if isinstance(table, AccuracySweepResults):
        if column not in table:
            raise KeyError(f"Column '{column}' is missing from AccuracySweepResults")
        return table[column]
    if column not in table:
        raise KeyError(f"Column '{column}' is missing from the provided mapping")
    return np.asarray(table[column])


def summarize_accuracy(
    results: AccuracySweepResults | Mapping[str, Sequence[Any]]
) -> list[dict[str, Any]]:
    """Compute RMSE summaries for sweep results grouped by method."""

    if isinstance(results, AccuracySweepResults) and results.row_count == 0:
        return []

    methods = _get_column(results, "method")
    dx_nm = _get_column(results, "dx_nm").astype(np.float64, copy=False)
    dy_nm = _get_column(results, "dy_nm").astype(np.float64, copy=False)

    if dx_nm.size == 0:
        return []

    unique_methods = sorted({str(_python_scalar(m)) for m in methods})
    summary_rows: list[dict[str, Any]] = []
    for method in unique_methods:
        mask = methods == method
        dx_vals = dx_nm[mask]
        dy_vals = dy_nm[mask]
        if dx_vals.size == 0:
            continue
        summary_rows.append(
            {
                "method": method,
                "rmse_dx_nm": float(np.sqrt(np.mean(dx_vals**2))),
                "rmse_dy_nm": float(np.sqrt(np.mean(dy_vals**2))),
            }
        )
    return summary_rows


def format_summary(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a table-style string for the provided summary rows."""

    if not summary_rows:
        return "No methods were summarized."

    header = f"{'method':<20} {'rmse_dx_nm':>12} {'rmse_dy_nm':>12}"
    lines = [header, "-" * len(header)]
    for row in summary_rows:
        method = str(row.get("method", ""))
        rmse_dx = float(row.get("rmse_dx_nm", float("nan")))
        rmse_dy = float(row.get("rmse_dy_nm", float("nan")))
        lines.append(f"{method:<20} {rmse_dx:>12.3f} {rmse_dy:>12.3f}")
    return "\n".join(lines)


def _as_tuple(value: float | int | Sequence[float | int]) -> Tuple[float | int, ...]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, IterableABC):
        return tuple(value)
    return (value,)


def _finalize_column_chunks(chunks: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=dtype)
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=dtype)
    return np.concatenate([np.asarray(chunk, dtype=dtype) for chunk in chunks])


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
) -> AccuracySweepResults:
    rng = np.random.default_rng(rng_seed)
    method_names = _ensure_methods(methods)

    column_chunks: Dict[str, List[np.ndarray]] = {name: [] for name in SWEEP_COLUMN_ORDER}
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

            column_chunks["method"].append(np.full(n_images, method, dtype=object))
            column_chunks["image_index"].append(
                np.arange(n_images, dtype=np.int64) + image_index_offset
            )
            column_chunks["x_true_px"].append(x_true_px)
            column_chunks["y_true_px"].append(y_true_px)
            column_chunks["x_est_px"].append(x_est_px)
            column_chunks["y_est_px"].append(y_est_px)
            column_chunks["dx_px"].append(dx_px)
            column_chunks["dy_px"].append(dy_px)
            column_chunks["dx_nm"].append(dx_nm)
            column_chunks["dy_nm"].append(dy_nm)
            column_chunks["radius_nm"].append(radius_nm_arr)
            column_chunks["background_level"].append(background_arr)
            column_chunks["contrast_scale"].append(contrast_arr)
            column_chunks["z_true_nm"].append(z_true_nm)
            column_chunks["nm_per_px"].append(np.full(n_images, nm_per_px_val, dtype=np.float64))
            column_chunks["size_px"].append(np.full(n_images, size_px_val, dtype=np.int64))
            column_chunks["size_nm"].append(np.full(n_images, size_nm_val, dtype=np.float64))
            column_chunks["magnification"].append(
                np.full(n_images, magnification_val, dtype=np.float64)
            )
            column_chunks["photons_per_unit"].append(
                np.full(n_images, photons_val, dtype=np.float64)
            )
            column_chunks["x_fraction_of_size"].append(x_fraction_arr)
            column_chunks["y_fraction_of_size"].append(y_fraction_arr)

        image_index_offset += n_images

    finalized_data = {
        name: _finalize_column_chunks(column_chunks[name], SWEEP_COLUMN_DTYPES[name])
        for name in SWEEP_COLUMN_ORDER
    }
    return AccuracySweepResults(finalized_data, column_order=SWEEP_COLUMN_ORDER)


__all__ = [
    "AccuracySweepConfig",
    "AccuracySweepResults",
    "DEFAULT_CAMERA_PIXEL_SIZE_NM",
    "DEFAULT_SWEEP_CONFIG",
    "MethodFunc",
    "format_summary",
    "register_method",
    "run_accuracy_sweep",
    "summarize_accuracy",
]
