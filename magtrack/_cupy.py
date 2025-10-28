"""Compatibility helpers for optional CuPy support.

This module exposes ``cp`` and ``cupyx`` objects that either proxy to the real
CuPy modules (when available) or provide lightweight NumPy/SciPy fallbacks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as _np

_cupy_available = False

try:  # pragma: no cover - exercised implicitly when CuPy is installed
    import cupy as _cupy  # type: ignore
    import cupyx as _cupyx  # type: ignore
except ImportError:  # pragma: no cover - exercised when CuPy is absent
    _cupy = None
    _cupyx = None
else:
    _cupy_available = True

if _cupy_available:
    cp = _cupy  # type: ignore[assignment]
    cupyx = _cupyx  # type: ignore[assignment]
else:
    try:
        import scipy as _scipy  # type: ignore
    except ImportError as exc:  # pragma: no cover - SciPy is an optional dep
        raise ImportError(
            "SciPy is required for the CPU fallback when CuPy is unavailable."
        ) from exc

    class _NumPyCupyCompat:
        """Subset of the CuPy API implemented with NumPy."""

        def __init__(self) -> None:
            self._np = _np
            self.random = _np.random
            self.fft = _np.fft
            self.linalg = _np.linalg
            self.testing = _np.testing
            self.cuda = SimpleNamespace(is_available=lambda: False)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._np, name)

        def asarray(self, obj, dtype=None):
            return self._np.asarray(obj, dtype=dtype)

        def array(self, obj, dtype=None):
            return self._np.array(obj, dtype=dtype)

        def asnumpy(self, obj):
            return obj

        def get_array_module(self, _):
            return self._np

    class _CupyxScipyCompat:
        """Subset of :mod:`cupyx.scipy` backed by SciPy."""

        def __init__(self) -> None:
            self._scipy = _scipy
            self.signal = _scipy.signal
            self.ndimage = _scipy.ndimage
            self.special = _scipy.special

        def get_array_module(self, _):
            return self

        def __getattr__(self, name: str) -> Any:
            return getattr(self._scipy, name)

    cp = _NumPyCupyCompat()
    cupyx = SimpleNamespace(scipy=_CupyxScipyCompat())


def is_cupy_available() -> bool:
    """Return ``True`` when the real CuPy package is importable."""

    return _cupy_available
