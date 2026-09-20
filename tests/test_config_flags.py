"""Regression tests for the production config surface (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thunder_vllm.attention.backend import ThunderCuteConfig  # noqa: E402


FLAGS = ("onepass", "reg_rescale", "causal_bound")


def test_fast_paths_off_by_default(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(f"THUNDER_{name.upper()}", raising=False)
    cfg = ThunderCuteConfig.from_env()
    assert cfg.onepass is False
    assert cfg.reg_rescale is False
    assert cfg.causal_bound is False


@pytest.mark.parametrize("name", FLAGS)
def test_flag_parsing(monkeypatch, name):
    env = f"THUNDER_{name.upper()}"
    for val, expected in (("1", True), ("true", True), ("on", True),
                          ("0", False), ("false", False), ("off", False)):
        monkeypatch.setenv(env, val)
        assert getattr(ThunderCuteConfig.from_env(), name) is expected


def test_kernel_key_tracks_flags(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(f"THUNDER_{name.upper()}", raising=False)
    base = ThunderCuteConfig.from_env().kernel_key(128, 8, True)
    for name in FLAGS:
        monkeypatch.setenv(f"THUNDER_{name.upper()}", "1")
        key = ThunderCuteConfig.from_env().kernel_key(128, 8, True)
        assert key != base, f"{name} must be part of the compiled-kernel key"
        monkeypatch.delenv(f"THUNDER_{name.upper()}", raising=False)


def test_allow_arch_sandbox_override(monkeypatch):
    """THUNDER_ALLOW_ARCH lets a non-Blackwell sandbox run the backend."""
    from thunder_vllm.attention.backend import ThunderAttentionBackend

    monkeypatch.delenv("THUNDER_ALLOW_ARCH", raising=False)
    assert ThunderAttentionBackend.supports_compute_capability((10, 0))
    assert not ThunderAttentionBackend.supports_compute_capability((8, 0))

    monkeypatch.setenv("THUNDER_ALLOW_ARCH", "80")
    assert ThunderAttentionBackend.supports_compute_capability((8, 0))
    assert not ThunderAttentionBackend.supports_compute_capability((7, 0))

    monkeypatch.setenv("THUNDER_ALLOW_ARCH", "all")
    assert ThunderAttentionBackend.supports_compute_capability((8, 6))
