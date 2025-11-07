"""Utilities for managing benchmark logs and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
import subprocess
from importlib import metadata
from pathlib import Path
import platform
import socket
from typing import Any, Iterable, Sequence

LOG_ROOT = Path(__file__).resolve().parent / "logs"


def _safe_float(value: Any) -> float | None:
    """Convert *value* to ``float`` if possible."""

    try:
        return float(value)
    except Exception:  # noqa: BLE001 - tolerate any conversion issue
        return None


def _sanitize_component(component: str) -> str:
    """Return a filesystem-friendly identifier component."""

    safe = [c if c.isalnum() or c in {"-", "_"} else "-" for c in component]
    return "".join(safe).strip("-") or "unknown"


def _raw_cpu_string() -> str | None:
    """Return a descriptive CPU brand string when available."""

    system = platform.system()

    if system == "Windows":
        try:  # pragma: no cover - exercised via unit tests through fallbacks
            import winreg  # type: ignore

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if value:
                    return str(value).strip()
        except Exception:  # noqa: BLE001
            pass

        try:
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            for line in lines:
                if line.lower() != "name":
                    return line
        except Exception:  # noqa: BLE001
            pass

    elif system == "Darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if output:
                return output.strip()
        except Exception:  # noqa: BLE001
            pass

    elif system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text(errors="ignore").splitlines():
                    key, _, value = line.partition(":")
                    if key.strip().lower() == "model name":
                        value = value.strip()
                        if value:
                            return value
        except Exception:  # noqa: BLE001
            pass

        try:
            output = subprocess.check_output(
                ["lscpu"], text=True, stderr=subprocess.DEVNULL
            )
            for line in output.splitlines():
                if line.lower().startswith("model name"):
                    _, _, value = line.partition(":")
                    value = value.strip()
                    if value:
                        return value
        except Exception:  # noqa: BLE001
            pass

    uname = platform.uname()
    for candidate in (
        getattr(uname, "processor", None),
        platform.processor(),
        getattr(uname, "machine", None),
    ):
        if candidate:
            text = str(candidate).strip()
            if text and text.lower() not in {"unknown", "generic"}:
                return text
    return None


def _normalize_cpu_brand(raw: str | None) -> str:
    """Return a short, sanitised CPU identifier slug."""

    if raw is None:
        return "unknown"

    brand = raw.strip()
    if not brand:
        return "unknown"

    for mark in ("(R)", "(r)", "(TM)", "(tm)", "(SM)", "(sm)", "®", "™", "℠"):
        brand = brand.replace(mark, "")

    brand = re.sub(r"@.*", "", brand)
    brand = re.sub(r"\b(?:CPU|Processor)\b", "", brand, flags=re.IGNORECASE)
    brand = re.sub(r"\b\d+(?:\.\d+)?\s*[GM]Hz\b", "", brand, flags=re.IGNORECASE)
    brand = re.sub(r"\s+", " ", brand).strip()

    if not brand:
        return "unknown"

    tokens = re.split(r"[\s/]+", brand)
    drop_tokens = {
        "core",
        "processor",
        "cpu",
        "with",
        "dual",
        "quad",
        "six",
        "eight",
        "twelve",
        "sixteen",
        "twenty",
        "thirty-two",
        "desktop",
        "mobile",
    }

    filtered: list[str] = []
    for token in tokens:
        token = token.strip("-_")
        if not token:
            continue
        lowered = token.lower()
        if lowered in drop_tokens:
            continue
        if "core" in lowered:
            continue
        filtered.append(token)

    if not filtered:
        filtered = [token for token in tokens if token]

    slug = "-".join(filtered)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return _sanitize_component(slug) or "unknown"


def make_system_id(hostname: str, system: str, machine: str, python_version: str) -> str:
    """Construct a deterministic identifier for the current system."""

    components = [hostname, system, machine, f"py{python_version}".replace(".", "_")]
    sanitized = [_sanitize_component(part.lower()) for part in components if part]
    return "-".join(sanitized)


def collect_system_metadata() -> tuple[str, str, dict[str, Any]]:
    """Gather runtime metadata about the host system and Python environment."""

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")

    uname = platform.uname()
    hostname = socket.gethostname() or uname.node
    python_version = platform.python_version()

    raw_cpu = _raw_cpu_string()
    cpu_slug = _normalize_cpu_brand(raw_cpu)

    metadata_dict: dict[str, Any] = {
        "collected_at": now.isoformat(),
        "hostname": hostname,
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": raw_cpu or uname.processor or platform.processor(),
        },
        "python": {
            "version": python_version,
            "implementation": platform.python_implementation(),
        },
        "dependencies": {},
    }

    # Dependency versions via importlib.metadata when available.
    dependencies = metadata_dict["dependencies"]

    for package in ("magtrack", "numpy", "scipy"):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue

    # Capture any installed CuPy distributions, regardless of suffix.
    try:
        cupy_packages: set[str] = set()
        for dist in metadata.distributions():
            name = dist.metadata.get("Name")
            if name and "cupy" in name.lower():
                cupy_packages.add(name)
    except Exception:  # noqa: BLE001 - fallback gracefully if metadata scan fails
        cupy_packages = set()

    for package in sorted(cupy_packages):
        if package in dependencies:
            continue
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue

    # psutil provides detailed CPU and memory metrics when installed.
    try:
        import psutil  # type: ignore

        cpu_freq = psutil.cpu_freq()
        metadata_dict["cpu"] = {
            "count_logical": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "frequency_mhz": _safe_float(cpu_freq.max if cpu_freq else None),
        }
        virtual_memory = psutil.virtual_memory()
        metadata_dict["memory"] = {
            "total_bytes": int(virtual_memory.total),
            "available_bytes": int(virtual_memory.available),
        }
    except Exception:  # noqa: BLE001 - psutil is optional
        metadata_dict["cpu"] = None
        metadata_dict["memory"] = None

    gpu_info: list[dict[str, Any]] = []
    try:
        import cupy as cp  # type: ignore

        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except Exception:  # noqa: BLE001 - GPU may be unavailable
            device_count = 0
        for device_id in range(device_count):
            try:
                props = cp.cuda.runtime.getDeviceProperties(device_id)
            except Exception:  # noqa: BLE001 - skip on error
                continue
            gpu_info.append(
                {
                    "id": device_id,
                    "name": props.get("name", b"?").decode(errors="ignore")
                    if isinstance(props.get("name"), (bytes, bytearray))
                    else props.get("name", "unknown"),
                    "total_memory": int(props.get("totalGlobalMem", 0)),
                    "multiprocessor_count": int(props.get("multiProcessorCount", 0)),
                }
            )
    except Exception:  # noqa: BLE001 - CuPy is optional
        gpu_info = []

    metadata_dict["gpus"] = gpu_info

    system_id = make_system_id(
        hostname,
        uname.system,
        cpu_slug if cpu_slug != "unknown" else uname.machine,
        python_version,
    )

    return system_id, timestamp, metadata_dict


def _ensure_serializable(value: Any) -> Any:
    """Convert *value* into a JSON-serialisable representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    try:
        return json.loads(json.dumps(value))
    except Exception:  # noqa: BLE001
        return repr(value)


def write_run_log(
    system_id: str,
    timestamp: str,
    metadata_dict: dict[str, Any],
    results: Sequence[dict[str, Any]],
    *,
    log_root: Path | None = None,
) -> Path:
    """Persist benchmark *results* and metadata to disk and return the run directory."""

    root = log_root or LOG_ROOT
    run_directory = root / system_id / timestamp
    run_directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "system_id": system_id,
        "timestamp": timestamp,
        "metadata": _ensure_serializable(metadata_dict),
        "results": [_ensure_serializable(entry) for entry in results],
    }

    log_path = run_directory / "results.json"
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return run_directory


def iter_run_logs(log_root: Path | None = None) -> Iterable[Path]:
    """Yield all ``results.json`` files stored under *log_root*."""

    root = (log_root or LOG_ROOT).resolve()
    if not root.exists():
        return []
    for system_dir in sorted(root.iterdir()):
        if not system_dir.is_dir():
            continue
        for run_dir in sorted(system_dir.iterdir()):
            candidate = run_dir / "results.json"
            if candidate.is_file():
                yield candidate


def aggregate_logs(log_root: Path | None = None) -> list[dict[str, Any]]:
    """Return flattened rows describing all recorded benchmark runs."""

    rows: list[dict[str, Any]] = []
    for log_path in iter_run_logs(log_root):
        try:
            data = json.loads(log_path.read_text())
        except Exception:  # noqa: BLE001 - ignore malformed logs
            continue

        system_id = data.get("system_id", "unknown")
        timestamp = data.get("timestamp", "")
        run_id = f"{system_id}/{timestamp}" if timestamp else system_id

        for entry in data.get("results", []):
            if entry.get("status") == "error":
                continue
            backend = entry.get("backend")
            if backend not in {"cpu", "gpu"}:
                continue
            stats = entry.get("statistics", {})
            rows.append(
                {
                    "run_id": run_id,
                    "system_id": system_id,
                    "timestamp": timestamp,
                    "module": entry.get("module"),
                    "benchmark": entry.get("benchmark"),
                    "backend": backend,
                    "mean_time": _safe_float(stats.get("mean")),
                    "std_time": _safe_float(stats.get("std")),
                    "min_time": _safe_float(stats.get("min")),
                    "max_time": _safe_float(stats.get("max")),
                    "repeat": int(stats.get("repeat", 0)),
                }
            )

    rows.sort(key=lambda r: (r["timestamp"], r["benchmark"], r["backend"]))
    return rows


def write_aggregate_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Write aggregated *rows* into ``output_path`` as CSV."""

    import csv

    fieldnames = [
        "timestamp",
        "run_id",
        "system_id",
        "module",
        "benchmark",
        "backend",
        "mean_time",
        "std_time",
        "min_time",
        "max_time",
        "repeat",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class BenchmarkStatistics:
    """Container describing benchmark statistics."""

    times: Sequence[float]

    @property
    def mean(self) -> float:
        return float(sum(self.times) / len(self.times)) if self.times else math.nan

    @property
    def std(self) -> float:
        if not self.times:
            return math.nan
        mu = self.mean
        variance = sum((x - mu) ** 2 for x in self.times) / len(self.times)
        return float(math.sqrt(variance))

    @property
    def minimum(self) -> float:
        return float(min(self.times)) if self.times else math.nan

    @property
    def maximum(self) -> float:
        return float(max(self.times)) if self.times else math.nan

    @property
    def repeat(self) -> int:
        return len(self.times)


def summarise_times(times: Sequence[float]) -> dict[str, Any]:
    """Return a statistics dictionary for *times*."""

    stats = BenchmarkStatistics(list(times))
    return {
        "mean": stats.mean,
        "std": stats.std,
        "min": stats.minimum,
        "max": stats.maximum,
        "repeat": stats.repeat,
    }

