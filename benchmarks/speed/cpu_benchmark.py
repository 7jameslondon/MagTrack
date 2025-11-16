"""Utilities for benchmarking CPU-only functions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass
class CPUTimingResult:
    """Simple container mirroring :func:`cupyx.profiler.benchmark` results."""

    cpu_times: np.ndarray


def cpu_benchmark(
    func: Callable[..., Any],
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    *,
    n_repeat: int = 10,
    n_warmup: int = 3,
    max_duration: float = 5.0,
) -> CPUTimingResult:
    """Benchmark a CPU-only function similarly to ``cupy.profiler.benchmark``.

    Parameters
    ----------
    func:
        Callable to be measured.
    args:
        Positional arguments supplied to ``func`` for each invocation.
    kwargs:
        Keyword arguments supplied to ``func`` for each invocation.
    n_repeat:
        Maximum number of timed runs to perform.
    n_warmup:
        Number of warm-up runs that are executed without timing.
    max_duration:
        Upper bound on the cumulative duration of all timed runs in seconds.

    Returns
    -------
    CPUTimingResult
        Object containing ``cpu_times`` with the elapsed seconds for each run.
    """

    if n_repeat <= 0:
        raise ValueError("n_repeat must be a positive integer")
    if n_warmup < 0:
        raise ValueError("n_warmup must be non-negative")
    if max_duration <= 0:
        raise ValueError("max_duration must be positive")

    call_args = tuple(args) if args is not None else tuple()
    call_kwargs = dict(kwargs or {})

    # Warm-up phase to mitigate one-time setup costs.
    for _ in range(n_warmup):
        func(*call_args, **call_kwargs)

    times: list[float] = []
    elapsed_total = 0.0
    for _ in range(n_repeat):
        start = perf_counter()
        func(*call_args, **call_kwargs)
        elapsed = perf_counter() - start
        times.append(elapsed)
        elapsed_total += elapsed
        if elapsed_total >= max_duration:
            break

    return CPUTimingResult(cpu_times=np.asarray(times, dtype=float))
