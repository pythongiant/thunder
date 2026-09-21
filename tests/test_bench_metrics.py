"""CPU tests for the P3 batched-benchmark metric helpers."""
import pytest

torch = pytest.importorskip("torch")

from benchmarks.bench_common import (  # noqa: E402
    effective_bw_gbps,
    fp16_per_token_bytes,
    kv_bytes_per_step,
    kv_bytes_per_token_from_layout,
)
from thunder_vllm.attention.cache_layout import ThunderCacheLayout  # noqa: E402


def test_fp16_per_token_bytes_qwen3_8b():
    # 8 KV heads, head_dim 128, K+V fp16 -> 4096 B/token
    assert fp16_per_token_bytes(8, 128) == 4096


def test_packed_per_token_bytes_3bit_k_4bit_v():
    layout = ThunderCacheLayout(num_kv_heads=8, head_dim=128, k_bits=3, v_bits=4,
                                block_size=16)
    # K 48 B + V 64 B per head, plus 2 fp16 norms per head
    assert kv_bytes_per_token_from_layout(layout) == 8 * (48 + 64) + 8 * 2 * 2


def test_kv_bytes_per_step_scales_with_batch_and_ctx():
    assert kv_bytes_per_step(928, 1, 4096) == 928 * 4096
    assert kv_bytes_per_step(928, 64, 4096) == 64 * 928 * 4096


def test_effective_bw_gbps():
    assert effective_bw_gbps(1e9, 1.0) == pytest.approx(1.0)
    assert effective_bw_gbps(2e9, 0.5) == pytest.approx(4.0)
    assert effective_bw_gbps(1e9, 0.0) == 0.0
    assert effective_bw_gbps(1e9, None) == 0.0


def test_packed_vs_fp16_ratio():
    layout = ThunderCacheLayout(num_kv_heads=8, head_dim=128, k_bits=3, v_bits=4,
                                block_size=16)
    ratio = fp16_per_token_bytes(8, 128) / kv_bytes_per_token_from_layout(layout)
    assert ratio == pytest.approx(4.41, abs=0.02)
