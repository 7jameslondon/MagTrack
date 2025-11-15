"""Benchmark orchestrator that executes all available benchmark suites."""

from __future__ import annotations

import argparse
import importlib
import importlib.resources as resources
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from benchmarks import log_utils


BenchmarkFunc = Callable[[], Any]


class BenchmarkRecorder:
    """Collect timing information emitted by benchmark helpers."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._current_entries: list[dict[str, Any]] = []
        self._sequence: int = 0

    @contextmanager
    def activate(self, module: Any) -> Iterable[None]:
        """Context manager that patches helper utilities to capture timings."""

        from benchmarks import cpu_benchmark as cpu_module

        original_cpu_benchmark = cpu_module.cpu_benchmark

        def recording_cpu_benchmark(*args: Any, **kwargs: Any):  # type: ignore[override]
            result = original_cpu_benchmark(*args, **kwargs)
            self.record_cpu(result)
            return result

        cpu_module.cpu_benchmark = recording_cpu_benchmark  # type: ignore[assignment]

        original_module_cpu = getattr(module, "cpu_benchmark", None)
        module_has_cpu = hasattr(module, "cpu_benchmark")
        module_cpu_matches = original_module_cpu is original_cpu_benchmark
        if module_has_cpu and module_cpu_matches:
            setattr(module, "cpu_benchmark", recording_cpu_benchmark)

        cupy_profiler = None
        original_cupy_benchmark = None
        try:
            import cupyx.profiler as cupy_profiler  # type: ignore

            original_cupy_benchmark = cupy_profiler.benchmark

            def recording_cupy_benchmark(*args: Any, **kwargs: Any):  # type: ignore[override]
                result = original_cupy_benchmark(*args, **kwargs)
                self.record_gpu(result)
                return result

            cupy_profiler.benchmark = recording_cupy_benchmark  # type: ignore[assignment]
        except Exception:  # noqa: BLE001 - CuPy is optional
            cupy_profiler = None

        try:
            yield
        finally:
            cpu_module.cpu_benchmark = original_cpu_benchmark  # type: ignore[assignment]
            if module_has_cpu and module_cpu_matches:
                setattr(module, "cpu_benchmark", original_module_cpu)
            if cupy_profiler is not None and original_cupy_benchmark is not None:
                cupy_profiler.benchmark = original_cupy_benchmark  # type: ignore[assignment]

    def begin(self, module: str, benchmark: str) -> None:
        """Mark the start of a benchmark function invocation."""

        self._sequence += 1
        self._current = {
            "module": module,
            "benchmark": benchmark,
            "sequence": self._sequence,
        }
        self._current_entries = []

    def record_cpu(self, result: Any) -> None:
        """Record CPU-only timing results."""

        if self._current is None:
            return
        times = self._to_float_list(getattr(result, "cpu_times", []))
        entry = {
            "backend": "cpu",
            "times": times,
            "statistics": log_utils.summarise_times(times),
        }
        self._current_entries.append(entry)

    def record_gpu(self, result: Any) -> None:
        """Record GPU timing results emitted by ``cupyx.profiler``."""

        if self._current is None:
            return
        cpu_times = self._to_float_list(getattr(result, "cpu_times", []))
        gpu_times = self._to_float_list(getattr(result, "gpu_times", []))
        if gpu_times:
            length = max(len(cpu_times), len(gpu_times))
            if len(cpu_times) < length:
                cpu_times = cpu_times + [0.0] * (length - len(cpu_times))
            if len(gpu_times) < length:
                gpu_times = gpu_times + [0.0] * (length - len(gpu_times))
            total_times = [c + g for c, g in zip(cpu_times, gpu_times)]
        else:
            total_times = cpu_times
        entry = {
            "backend": "gpu",
            "times": total_times,
            "statistics": log_utils.summarise_times(total_times),
            "details": {
                "cpu_times": cpu_times,
                "gpu_times": gpu_times,
            },
        }
        self._current_entries.append(entry)

    def finalize(self, *, status: str = "success", error: str | None = None) -> None:
        """Persist entries captured during :meth:`begin`."""

        if self._current is None:
            return
        if not self._current_entries:
            self._current_entries.append(
                {
                    "backend": "cpu",
                    "times": [],
                    "statistics": log_utils.summarise_times([]),
                }
            )
        for entry in self._current_entries:
            payload = {
                **self._current,
                **entry,
                "status": status,
            }
            if error is not None:
                payload["error"] = error
            self._records.append(payload)
        self._current = None
        self._current_entries = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @staticmethod
    def _to_float_list(values: Any) -> list[float]:
        try:
            return [float(v) for v in np.asarray(values, dtype=float).ravel()]
        except Exception:  # noqa: BLE001
            try:
                import cupy as cp  # type: ignore

                return [float(v) for v in cp.asnumpy(values).ravel()]
            except Exception:  # noqa: BLE001
                return []


def discover_benchmark_modules(root: Path | None = None) -> list[str]:
    """Return fully-qualified module names containing benchmarks."""

    modules: list[str] = []

    traversable = None
    if root is not None:
        traversable = root
    else:
        try:
            traversable = resources.files("benchmarks")
        except Exception:  # noqa: BLE001 - fall back to filesystem path lookup
            traversable = Path(__file__).resolve().parent

    for entry in sorted(traversable.iterdir(), key=lambda item: item.name):
        if not entry.name.startswith("benchmark_"):
            continue
        if entry.name in {"benchmark_utils.py", "benchmark_common.py"}:
            continue
        if entry.is_file() and entry.name.endswith(".py"):
            modules.append(f"benchmarks.{Path(entry.name).stem}")
    return modules


def discover_benchmark_functions(module: Any) -> list[tuple[str, BenchmarkFunc]]:
    """Return ``benchmark_*`` callables defined in *module*."""

    candidates: list[tuple[str, BenchmarkFunc]] = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("benchmark_"):
            candidates.append((name, obj))
    return candidates


def run_benchmarks(selected_modules: Iterable[str] | None = None) -> dict[str, Any]:
    """Execute benchmark modules and return structured results."""

    modules = (
        list(selected_modules)
        if selected_modules is not None
        else discover_benchmark_modules()
    )

    recorder = BenchmarkRecorder()
    results: list[dict[str, Any]] = []

    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - keep orchestrator running
            results.append(
                {
                    "module": module_name,
                    "benchmark": None,
                    "backend": "error",
                    "status": "error",
                    "error": repr(exc),
                    "times": [],
                    "statistics": log_utils.summarise_times([]),
                }
            )
            continue

        functions = discover_benchmark_functions(module)
        for func_name, func in functions:
            print(f"\nRunning {module_name}.{func_name}()")
            recorder.begin(module_name, func_name)
            try:
                with recorder.activate(module):
                    func()
            except Exception as exc:  # noqa: BLE001 - keep processing other benchmarks
                recorder.finalize(status="error", error=repr(exc))
                print(f"  -> Failed: {exc}")
            else:
                recorder.finalize(status="success")

    results.extend(recorder.records)
    return {
        "results": results,
        "modules": modules,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modules",
        nargs="*",
        help="Optional explicit benchmark modules to run (e.g. benchmarks.benchmark_gaussian)",
    )
    args = parser.parse_args(argv)

    system_id, timestamp, metadata_dict = log_utils.collect_system_metadata()
    module_names = args.modules or None
    if module_names is not None:
        print("Using explicitly provided modules:", ", ".join(module_names))

    run_data = run_benchmarks(module_names)
    metadata_dict = {**metadata_dict, "benchmarks": run_data.get("modules", [])}
    run_data["metadata"] = metadata_dict
    run_data["system_id"] = system_id
    run_data["timestamp"] = timestamp

    run_dir = log_utils.write_run_log(system_id, timestamp, metadata_dict, run_data["results"])
    print(f"\n\n\nBenchmark results written to {run_dir}")

    aggregated_rows = log_utils.aggregate_logs()
    aggregate_path = log_utils.LOG_ROOT / "combined_results.csv"
    log_utils.write_aggregate_csv(aggregated_rows, aggregate_path)
    print(f"Aggregated history written to {aggregate_path}")

    try:
        from benchmarks.plot_benchmarks import plot_benchmark_history
        import matplotlib.pyplot as plt

        fig = plot_benchmark_history(log_utils.LOG_ROOT, run_dir)
        if fig is not None:
            plt.show()
    except Exception as exc:  # noqa: BLE001 - plotting should not abort the run
        print(f"Plotting failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

