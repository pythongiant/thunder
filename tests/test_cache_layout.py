"""Cache layout, packing, codebook and quantizer round-trip tests (CPU)."""

from __future__ import annotations

import pytest
import torch

from thunder_vllm.attention.cache_layout import (
    ThunderCacheLayout,
    allocate_kv_cache,
    reshape_and_cache_ref,
)
from thunder_vllm.quant.hadamard import build_hadamard
from thunder_vllm.quant.lloyd_max import build_lloyd_max_codebook, build_lut
from thunder_vllm.quant.packing import pack_indices, unpack_indices
from thunder_vllm.quant.quantizer import ThunderQuantizer

BITS_SWEEP = [(2, 2), (3, 3), (4, 4), (3, 4), (4, 4)]
HEAD_DIMS = [64, 128, 256]


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_hadamard_is_orthonormal_and_symmetric(head_dim):
    h = build_hadamard(head_dim)
    assert h.shape == (head_dim, head_dim)
    eye = torch.eye(head_dim)
    assert torch.allclose(h @ h.t(), eye, atol=1e-5)
    assert torch.allclose(h, h.t(), atol=1e-6)


def test_hadamard_non_power_of_two_uses_random_orthogonal():
    h = build_hadamard(12)
    assert h.shape == (12, 12)
    assert torch.allclose(h @ h.t(), torch.eye(12), atol=1e-5)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_packing_roundtrip(bits, head_dim):
    torch.manual_seed(0)
    idx = torch.randint(0, 1 << bits, (9, head_dim))
    packed = pack_indices(idx, bits, head_dim)
    assert packed.shape[-1] == (head_dim * bits + 7) // 8
    assert packed.dtype == torch.uint8
    assert torch.equal(unpack_indices(packed, bits, head_dim), idx)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_codebook_is_sorted_and_symmetric(bits):
    cb = build_lloyd_max_codebook(
        bits, num_iterations=50, num_samples=100_000, head_dim=128
    )
    assert cb.centroids.numel() == 1 << bits
    assert torch.all(cb.centroids[1:] >= cb.centroids[:-1])
    assert cb.boundaries.numel() == (1 << bits) - 1
    # Symmetric distribution -> centroids are exactly antisymmetric after the
    # projection in build_lloyd_max_codebook.
    assert torch.allclose(cb.centroids, -cb.centroids.flip(0), atol=1e-5)


def test_lut_shape_and_broadcast():
    cb = build_lloyd_max_codebook(4, num_iterations=20, num_samples=50_000, head_dim=64)
    lut = build_lut(cb, head_dim_padded=64)
    assert lut.shape == (16, 64)
    assert lut.dtype == torch.float16
    # Every column identical.
    assert torch.equal(lut[:, 0:1].expand(-1, 64), lut)


@pytest.mark.parametrize("k_bits,v_bits", BITS_SWEEP)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_quantizer_roundtrip_error(k_bits, v_bits, head_dim):
    torch.manual_seed(1)
    q = ThunderQuantizer(head_dim, k_bits, v_bits)
    k = torch.randn(8, 4, head_dim, dtype=torch.float16)
    v = torch.randn(8, 4, head_dim, dtype=torch.float16)
    kv = q.quantize(k, v)
    kd, vd = q.dequantize(kv)
    k_rel = (kd - k.float()).norm() / k.float().norm()
    v_rel = (vd - v.float()).norm() / v.float().norm()
    # Per-side budget: a 2-bit side is coarse even when the other side is 4-bit.
    budget = {2: 0.35, 3: 0.22, 4: 0.15}
    assert k_rel.item() < budget[k_bits], f"K rel err {k_rel.item():.4f}"
    assert v_rel.item() < budget[v_bits], f"V rel err {v_rel.item():.4f}"


@pytest.mark.parametrize("k_bits,v_bits", BITS_SWEEP)
def test_cache_layout_shapes_and_views(k_bits, v_bits):
    layout = ThunderCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=k_bits, v_bits=v_bits, block_size=16
    )
    nb = 4
    kv, scales = allocate_kv_cache(nb, 16, 8, 128, k_bits, v_bits, device="cpu")
    # vLLM canonical layout: (num_blocks, num_kv_heads, block_size, slot), combined
    # K+V, no leading K/V dim. The kernels' (nb, bs, Hk, ...) order is produced by
    # the single transpose boundary in k_codes/v_codes/k_norm/v_norm.
    assert kv.shape == layout.get_kv_cache_shape(nb)
    assert kv.shape == (nb, 8, 16, layout.head_slot_bytes)
    assert scales.shape == layout.get_scales_shape(nb)
    assert scales.shape == (nb, 8, 16, 2)
    views = layout.views(kv, scales)
    assert views.k_codes.shape == (nb, 16, 8, layout.k_packed_bytes)
    assert views.v_codes.shape == (nb, 16, 8, layout.v_packed_bytes)
    assert views.k_norm.shape == (nb, 16, 8)


def test_cache_write_scatter_matches_reference():
    layout = ThunderCacheLayout(
        num_kv_heads=4, head_dim=128, k_bits=4, v_bits=4, block_size=16
    )
    q = ThunderQuantizer(128, 4, 4)
    kv, scales = allocate_kv_cache(6, 16, 4, 128, 4, 4, device="cpu")

    torch.manual_seed(3)
    n = 20
    key = torch.randn(n, 4, 128, dtype=torch.float16)
    value = torch.randn(n, 4, 128, dtype=torch.float16)
    slot = torch.randperm(6 * 16)[:n].to(torch.long)

    reshape_and_cache_ref(key, value, slot, kv, scales, q, layout)

    expected = q.quantize(key, value)
    k_view = layout.k_codes(kv)
    for i in range(n):
        blk, off = layout.split_slot(int(slot[i]))
        assert torch.equal(k_view[blk, off], expected.k_packed[i])
        assert torch.allclose(
            scales[blk, :, off, 0].float(), expected.k_norm[i].float()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton scatter needs CUDA")
def test_triton_scatter_matches_reference_at_the_engine_geometry():
    """The Triton scatter at the contract the ENGINE actually uses.

    `test_cache_write_scatter_matches_reference` covers the torch reference at
    4-bit K, which takes `_reshape_and_cache_kernel`; the engine's default is
    3-bit K, an unsupported width that routes to `_scatter_codes` -- torch
    quantizer plus the Triton data movement -- which therefore had NO parity
    coverage at all. The shapes here are the engine's measured ones
    (`k_packed=(n, 8, 48)`, `k_norm=(n, 8)`, cache `(nb, 8, 16, 112)`), and the
    slot mapping carries PAD_SLOT_ID, since the engine's padded batches do.
    """
    from thunder_vllm.attention.cache_layout import reshape_and_cache

    layout = ThunderCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=3, v_bits=4, block_size=16
    )
    q = ThunderQuantizer(128, 3, 4)
    nb, n = 6, 20
    torch.manual_seed(11)
    key = torch.randn(n, 8, 128, dtype=torch.float16, device="cuda")
    value = torch.randn(n, 8, 128, dtype=torch.float16, device="cuda")
    slot = torch.randperm(nb * 16, device="cuda")[:n].to(torch.long)
    slot[3] = -1  # PAD_SLOT_ID, as a padded engine batch has

    kv_ref, sc_ref = allocate_kv_cache(nb, 16, 8, 128, 3, 4, device="cuda")
    kv_tri, sc_tri = allocate_kv_cache(nb, 16, 8, 128, 3, 4, device="cuda")
    reshape_and_cache_ref(key, value, slot, kv_ref, sc_ref, q, layout)
    reshape_and_cache(key, value, slot, kv_tri, sc_tri, q, layout)
    torch.cuda.synchronize()

    assert torch.equal(layout.k_codes(kv_tri), layout.k_codes(kv_ref))
    assert torch.equal(layout.v_codes(kv_tri), layout.v_codes(kv_ref))
    assert torch.equal(layout.k_norm(sc_tri), layout.k_norm(sc_ref))
    assert torch.equal(layout.v_norm(sc_tri), layout.v_norm(sc_ref))
