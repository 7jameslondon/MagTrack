"""Backend-agnostic array utilities for MagTrack."""
from __future__ import annotations

import os
from types import ModuleType
from typing import Any

import numpy as _np
from scipy import signal as _sp_signal

__all__ = [
    "xp",
    "asarray",
    "asnumpy",
    "get_array_module",
    "fftconvolve",
    "is_cupy_backend",
    "refresh_backend",
]

_cp: ModuleType | None = None
_cupyx_signal: ModuleType | None = None
_USING_CUPY = False


def _initialize_backend() -> None:
    """Initialise the computation backend (CuPy if available, else NumPy)."""
    global _cp, _cupyx_signal, _USING_CUPY

    force_numpy = os.environ.get("MAGTRACK_FORCE_NUMPY", "").strip()
    if force_numpy:
        _cp = None
        _cupyx_signal = None
        _USING_CUPY = False
        return

    try:  # pragma: no cover - exercised conditionally in CI depending on CuPy
        import cupy as cp  # type: ignore
        from cupyx.scipy import signal as cupyx_signal  # type: ignore
    except Exception:  # pragma: no cover - import failure means fallback to NumPy
        _cp = None
        _cupyx_signal = None
        _USING_CUPY = False
    else:
        _cp = cp
        _cupyx_signal = cupyx_signal
        _USING_CUPY = True


def refresh_backend() -> None:
    """Re-evaluate backend availability.

    This is useful in tests where the environment is temporarily adjusted to
    force the NumPy fallback.
    """

    _initialize_backend()


_initialize_backend()


class _ArrayModuleProxy:
    """Proxy object exposing the active array module."""

    def __getattr__(self, name: str) -> Any:
        module = _cp if _USING_CUPY and _cp is not None else _np
        return getattr(module, name)

    def __dir__(self) -> list[str]:  # pragma: no cover - trivial forwarding
        module = _cp if _USING_CUPY and _cp is not None else _np
        return sorted(set(dir(module)))

    def __repr__(self) -> str:  # pragma: no cover - representational helper
        backend = "cupy" if _USING_CUPY and _cp is not None else "numpy"
        return f"<ArrayModuleProxy backend={backend}>"


xp = _ArrayModuleProxy()


def is_cupy_backend() -> bool:
    """Return ``True`` when the active backend is CuPy."""

    return _USING_CUPY and _cp is not None


def asarray(array: Any, dtype: Any | None = None) -> Any:
    """Convert *array* to the active backend."""

    if is_cupy_backend():  # pragma: no branch - simple backend dispatch
        return _cp.asarray(array, dtype=dtype)  # type: ignore[union-attr]
    return _np.asarray(array, dtype=dtype)


def asnumpy(array: Any) -> Any:
    """Convert *array* to a NumPy ``ndarray`` if using CuPy."""

    if is_cupy_backend():  # pragma: no branch - simple backend dispatch
        return _cp.asnumpy(array)  # type: ignore[union-attr]
    return array


def get_array_module(array: Any) -> ModuleType:
    """Return the array module corresponding to *array*."""

    if is_cupy_backend():  # pragma: no branch - simple backend dispatch
        return _cp.get_array_module(array)  # type: ignore[union-attr]
    return _np


def fftconvolve(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic ``fftconvolve`` implementation."""

    if is_cupy_backend():  # pragma: no branch - simple backend dispatch
        return _cupyx_signal.fftconvolve(*args, **kwargs)  # type: ignore[union-attr]
    return _sp_signal.fftconvolve(*args, **kwargs)
