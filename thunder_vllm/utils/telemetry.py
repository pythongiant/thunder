"""Machine-readable telemetry for profiling runs and tests.

Every profiling/test entry point starts with one ``[TQ-SYS]`` line carrying a
full system fingerprint, and every diagnostic dump goes through
:func:`emit` so counters travel with the machine, versions, config, and git
revision they were measured with.

No secrets are ever collected: only an allowlist of ``THUNDER_*``/CUDA/HF
environment variables is read. Anything else in the environment is invisible
to this module by construction.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

# Env vars that are safe to record. Everything else is excluded, including any
# token/key/secret variables.
_ENV_ALLOW_PREFIXES = ("THUNDER_",)
_ENV_ALLOW_EXACT = frozenset({
    "CUDA_VISIBLE_DEVICES",
    "CUTE_DSL_ARCH",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_HOME",
    "HF_HOME",
    "FLASH_ATTENTION_NUM_SMS",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
})

# Runtime flags worth resolving to booleans alongside their raw values.
_KNOWN_FLAGS = (
    "THUNDER_ONEPASS",
    "THUNDER_REG_RESCALE",
    "THUNDER_CAUSAL_BOUND",
    "THUNDER_8B_INDIRECT",
    "THUNDER_STORE3",
    "THUNDER_FASTLAUNCH",
    "THUNDER_SPLITS",
    "THUNDER_ALLOW_ARCH",
    "THUNDER_SCHEDULE",
    "THUNDER_STAGE_TIMING",
    "THUNDER_TIME_LAUNCH",
    "THUNDER_DIAG",
)


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _git_revision() -> str | None:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        rev = out.stdout.strip()
        return rev or None
    except Exception:  # noqa: BLE001
        return None


def _gpu_info() -> dict:
    info: dict = {"cuda_available": False, "device_count": 0, "devices": []}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if not info["cuda_available"]:
            return info
        info["device_count"] = int(torch.cuda.device_count())
        for i in range(info["device_count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                })
            except Exception:  # noqa: BLE001
                info["devices"].append({"index": i, "error": "unreadable"})
    except Exception:  # noqa: BLE001
        pass
    return info


def env_snapshot() -> dict[str, str]:
    """Allowlisted environment variables only; never secrets."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k.startswith(_ENV_ALLOW_PREFIXES) or k in _ENV_ALLOW_EXACT:
            out[k] = v
    return out


def system_info() -> dict:
    """Full system fingerprint. All values are JSON-serializable."""
    from thunder_vllm.utils.logging import env_flag

    versions = {
        "python": platform.python_version(),
        "torch": _pkg_version("torch"),
        "triton": _pkg_version("triton"),
        "cutlass_dsl": _pkg_version("nvidia-cutlass-dsl"),
        "vllm": _pkg_version("vllm"),
        "thunder_vllm": _pkg_version("thunder-vllm"),
        "transformers": _pkg_version("transformers"),
    }
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_info(),
        "versions": versions,
        "git_revision": _git_revision(),
        "env": env_snapshot(),
        "flags": {name: env_flag(name) for name in _KNOWN_FLAGS},
    }


def system_line() -> str:
    """One-line ``[TQ-SYS]`` fingerprint for the top of every profiling run."""
    return "[TQ-SYS] " + json.dumps(system_info(), sort_keys=True)


def emit(tag: str, payload: dict | None = None) -> None:
    """Print one machine-readable telemetry record.

    ``tag`` names the payload (``stage``, ``diag``, ``launch``, ``csr``,
    ``count``); the system fingerprint is attached automatically.
    """
    record = {"tag": tag, "system": system_info(), "payload": payload or {}}
    print("[TQ-TELEMETRY] " + json.dumps(record, sort_keys=True, default=str),
          flush=True)
