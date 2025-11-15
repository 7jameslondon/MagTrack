"""Utility package for MagTrack benchmark scripts."""

from __future__ import annotations

import sys

try:  # pragma: no cover - defensive aliasing
    from . import confbenchmarks as _confbenchmarks
except ImportError:  # pragma: no cover - when file missing, leave untouched
    _confbenchmarks = None  # type: ignore[assignment]
else:  # pragma: no cover - import carries side effects
    sys.modules.setdefault("confbenchmarks", _confbenchmarks)

# The benchmark scripts rely on relative imports when executed via
# ``python -m benchmarks.run_all``. This module ensures the directory behaves as
# a package without imposing additional side effects.
