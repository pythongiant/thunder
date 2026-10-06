"""Regression tests for the production config surface (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thunder_vllm.attention.backend import ThunderCuteConfig  # noqa: E402


FLAGS = ("onepass", "reg_rescale", "causal_bound")


def test_fast_paths_on_by_default(monkeypatch):
    """Verified kernel wins are on by default; THUNDER_*=0 opts out."""
    for name in FLAGS:
        monkeypatch.delenv(f"THUNDER_{name.upper()}", raising=False)
    cfg = ThunderCuteConfig.from_env()
    assert cfg.onepass is True
    assert cfg.reg_rescale is True
    assert cfg.causal_bound is True
    for name in FLAGS:
        monkeypatch.setenv(f"THUNDER_{name.upper()}", "0")
        assert getattr(ThunderCuteConfig.from_env(), name) is False
        monkeypatch.delenv(f"THUNDER_{name.upper()}", raising=False)


@pytest.mark.parametrize("name", FLAGS)
def test_flag_parsing(monkeypatch, name):
    env = f"THUNDER_{name.upper()}"
    for val, expected in (("1", True), ("true", True), ("on", True),
                          ("0", False), ("false", False), ("off", False)):
        monkeypatch.setenv(env, val)
        assert getattr(ThunderCuteConfig.from_env(), name) is expected


def test_gqa_pack_on_by_default_with_opt_out(monkeypatch):
    """GQA-packed decode is the shipped decode schedule; THUNDER_GQA_PACK=0 opts out.

    Measured at the current decode tile: 2.72 vs 11.62 ms at batch 16/32k (4.3x)
    and 0.536 vs 0.778 at batch 1, because it removes the 4x redundant KV load
    and dequant (one CTA per KV head serves the whole query group). It measured
    neutral at the old 64-row tile, which is why it used to be off.
    """
    monkeypatch.delenv("THUNDER_GQA_PACK", raising=False)
    assert ThunderCuteConfig.from_env().gqa_pack is True
    monkeypatch.setenv("THUNDER_GQA_PACK", "0")
    assert ThunderCuteConfig.from_env().gqa_pack is False


def test_kernel_key_tracks_flags(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(f"THUNDER_{name.upper()}", raising=False)
    base = ThunderCuteConfig.from_env().kernel_key(128, 8, True)
    for name in FLAGS:
        # Defaults are True, so flip to 0 to change the key.
        monkeypatch.setenv(f"THUNDER_{name.upper()}", "0")
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
