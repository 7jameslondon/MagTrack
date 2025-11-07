"""Tests for the benchmark logging utilities."""

from __future__ import annotations

from collections import namedtuple
import sys
import types

import pytest

from benchmarks import log_utils


def test_normalize_cpu_brand_truncates_trademarks() -> None:
    raw = "Intel(R) Core(TM) i7-13700 CPU @ 2.10GHz"
    assert log_utils._normalize_cpu_brand(raw) == "Intel-i7-13700"


def test_collect_system_metadata_uses_cpu_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_cpu = "Intel(R) Core(TM) i7-13700K CPU @ 3.40GHz"

    monkeypatch.setattr(log_utils, "_raw_cpu_string", lambda: raw_cpu)

    FakeUname = namedtuple(
        "FakeUname", "system node release version machine processor"
    )
    fake_uname = FakeUname(
        system="Linux",
        node="node.local",
        release="6.5.0",
        version="#1 SMP",
        machine="x86_64",
        processor="",
    )
    monkeypatch.setattr(log_utils.platform, "uname", lambda: fake_uname)
    monkeypatch.setattr(log_utils.platform, "python_version", lambda: "3.11.4")
    monkeypatch.setattr(log_utils.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(log_utils.socket, "gethostname", lambda: "bench-host")

    def fake_version(package: str) -> str:
        if package == "magtrack":
            return "0.1.0"
        raise log_utils.metadata.PackageNotFoundError

    monkeypatch.setattr(log_utils.metadata, "version", fake_version)
    monkeypatch.setattr(log_utils.metadata, "distributions", lambda: [])

    dummy_psutil = types.SimpleNamespace(
        cpu_freq=lambda: types.SimpleNamespace(max=3600.0),
        cpu_count=lambda logical=True: 16 if logical else 8,
        virtual_memory=lambda: types.SimpleNamespace(total=32, available=16),
    )
    monkeypatch.setitem(sys.modules, "psutil", dummy_psutil)

    system_id, timestamp, metadata = log_utils.collect_system_metadata()

    assert timestamp
    assert metadata["platform"]["processor"] == raw_cpu

    expected_cpu_slug = log_utils._normalize_cpu_brand(raw_cpu)
    expected_id = log_utils.make_system_id(
        "bench-host",
        fake_uname.system,
        expected_cpu_slug,
        "3.11.4",
    )
    assert system_id == expected_id

    # Ensure the identifier keeps the expected OS-CPU-GPU-style structure.
    assert "-linux-" in system_id
    assert f"-{expected_cpu_slug.lower()}-" in system_id
