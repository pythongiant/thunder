"""Logging helpers.

Uses ``vllm.logger`` when running inside a vLLM process so log lines land in
the same stream as the rest of the engine, and falls back to the stdlib logger
otherwise (tests, benchmarks, CPU-only tooling).
"""

from __future__ import annotations

import logging
import os

_PREFIX = "turboquant_vllm"
_configured: dict[str, logging.Logger] = {}

try:  # pragma: no cover - exercised only inside a real vLLM install
    from vllm.logger import init_logger as _vllm_init_logger  # type: ignore
except Exception:  # noqa: BLE001
    _vllm_init_logger = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger for ``name`` (defaults to the package root)."""
    full = _PREFIX if not name else f"{_PREFIX}.{name}"
    if full in _configured:
        return _configured[full]

    if _vllm_init_logger is not None:
        logger = _vllm_init_logger(full)
    else:
        logger = logging.getLogger(full)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(os.environ.get("TURBOQUANT_LOG_LEVEL", "INFO").upper())
        logger.propagate = False

    _configured[full] = logger
    return logger


def log_once(logger: logging.Logger, message: str, *args: object) -> None:
    """Emit ``message`` at most once per process for this logger."""
    key = f"_tq_logged_{id(logger)}"
    if getattr(logger, key, False):
        return
    logger.info(message, *args)
    setattr(logger, key, True)
