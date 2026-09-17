"""vLLM plugin registration.

The entry point is declared in ``pyproject.toml``::

    [project.entry-points."vllm.general_plugins"]
    turboquant_cute = "turboquant_vllm.model.registry:register"

vLLM calls ``register()`` once at engine startup.

Why ``AttentionBackendEnum.CUSTOM``
-----------------------------------
vLLM ``main`` already registers a backend under ``AttentionBackendEnum.TURBOQUANT``
(the reference Triton implementation). Overriding it would delete the baseline
this project exists to beat, so the CuTe backend registers into the third-party
``CUSTOM`` slot and is selected with ``--attention-backend CUSTOM``.

If a future vLLM exposes a dynamic backend-registration hook (or the plugin is
deployed alongside a patched enum), :func:`register` also installs a
``TURBOQUANT_CUTE`` enum alias when that is possible without touching the
upstream ``TURBOQUANT`` member.
"""

from __future__ import annotations

import os
from typing import Any

from turboquant_vllm.attention.backend import (
    BACKEND_NAME,
    TurboQuantAttentionBackend,
    TurboQuantCuteConfig,
)
from turboquant_vllm.utils.logging import get_logger

logger = get_logger("model.registry")

_BACKEND_PATH = (
    "turboquant_vllm.attention.backend.TurboQuantAttentionBackend"
)


def configure(
    *,
    k_bits: int | None = None,
    v_bits: int | None = None,
    num_stages: int | None = None,
    num_threads: int | None = None,
) -> None:
    """Set plugin knobs via the environment before the engine starts.

    Call this from the model/deploy entry point. The backend reads the
    environment once at impl construction.
    """
    mapping = {
        "TURBOQUANT_K_BITS": k_bits,
        "TURBOQUANT_V_BITS": v_bits,
        "TURBOQUANT_NUM_STAGES": num_stages,
        "TURBOQUANT_NUM_THREADS": num_threads,
    }
    for key, val in mapping.items():
        if val is not None:
            os.environ[key] = str(val)


def _install_enum_alias() -> None:
    """Best-effort install of a ``TURBOQUANT_CUTE`` enum member.

    Python ``Enum`` forbids adding members after class creation; we only inject
    into the lookup tables, and only when the member is absent. Failure is
    non-fatal: selection still works through ``CUSTOM``.
    """
    try:  # pragma: no cover - depends on vLLM internals
        from vllm.v1.attention.backends.registry import (  # type: ignore
            AttentionBackendEnum,
            register_backend,
        )
    except Exception:  # noqa: BLE001
        return
    try:
        if BACKEND_NAME in AttentionBackendEnum.__members__:
            return
        # Reuse the enum machinery to build a member from the CUSTOM template.
        member = AttentionBackendEnum.__new__(AttentionBackendEnum, _BACKEND_PATH)
        member._name_ = BACKEND_NAME
        member._value_ = _BACKEND_PATH
        AttentionBackendEnum._member_map_[BACKEND_NAME] = member
        AttentionBackendEnum._value2member_map_[_BACKEND_PATH] = member
        register_backend(member, _BACKEND_PATH)  # type: ignore[arg-type]
        logger.info("registered %s as a first-class backend enum", BACKEND_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not install %s enum alias: %s", BACKEND_NAME, exc)


def register() -> None:
    """Entry point invoked by vLLM's general-plugin loader."""
    try:  # pragma: no cover - depends on vLLM internals
        from vllm.v1.attention.backends.registry import (  # type: ignore
            AttentionBackendEnum,
            register_backend,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "vLLM attention backend registry not found (%s); "
            "TurboQuant-CuTe backend will not be selectable",
            exc,
        )
        return

    register_backend(AttentionBackendEnum.CUSTOM, _BACKEND_PATH)
    _install_enum_alias()

    cfg = TurboQuantCuteConfig.from_env()
    logger.info(
        "registered attention backend %s (K_BITS=%d V_BITS=%d, select with "
        "--attention-backend %s)",
        BACKEND_NAME,
        cfg.k_bits,
        cfg.v_bits,
        BACKEND_NAME,
    )


def backend_cls() -> type[TurboQuantAttentionBackend]:
    return TurboQuantAttentionBackend


__all__ = ["backend_cls", "configure", "register"]


# Keep static analysers from flagging the re-export-only import.
_ = Any
