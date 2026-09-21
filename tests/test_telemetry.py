"""Telemetry collector: shape, JSON-safety, and the no-secrets guarantee."""

import io
import json
import os
from contextlib import redirect_stdout

from thunder_vllm.utils import telemetry


def test_system_info_shape():
    info = telemetry.system_info()
    assert set(info) == {
        "platform", "cpu_count", "gpu", "versions", "git_revision",
        "env", "flags",
    }
    assert set(info["gpu"]) == {
        "cuda_available", "device_count", "devices",
    }
    for key in ("python", "torch", "triton", "vllm", "thunder_vllm"):
        assert key in info["versions"]
    # Must survive JSON round-trip (this is what profiling logs store).
    json.dumps(info, sort_keys=True, default=str)


def test_env_snapshot_excludes_secrets(monkeypatch):
    monkeypatch.setenv("THUNDER_SPLITS", "4")
    monkeypatch.setenv("TOTALLY_SECRET_TOKEN", "hunter2")
    monkeypatch.setenv("HF_TOKEN", "hunter2")
    snap = telemetry.env_snapshot()
    assert snap.get("THUNDER_SPLITS") == "4"
    assert "TOTALLY_SECRET_TOKEN" not in snap
    assert "HF_TOKEN" not in snap
    assert all(
        k.startswith("THUNDER_") or k in telemetry._ENV_ALLOW_EXACT
        for k in snap
    )


def test_emit_is_machine_readable():
    buf = io.StringIO()
    with redirect_stdout(buf):
        telemetry.emit("unit-test", {"counters": {"a": 1},
                                     "nested": {"b": [1.0, 2.0]}})
    line = buf.getvalue().strip()
    assert line.startswith("[TQ-TELEMETRY] ")
    record = json.loads(line[len("[TQ-TELEMETRY] "):])
    assert record["tag"] == "unit-test"
    assert record["payload"]["counters"] == {"a": 1}
    assert "system" in record and "gpu" in record["system"]


def test_system_line_prefix():
    assert telemetry.system_line().startswith("[TQ-SYS] ")
    # The fingerprint line itself must parse.
    json.loads(telemetry.system_line()[len("[TQ-SYS] "):])
