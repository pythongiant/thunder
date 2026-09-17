"""Kernel correctness vs. an fp32 reference (SM100/SM110 only).

Correctness gate from the spec: quantize -> dequantize -> SDPA is the
reference; the fused kernel must match it within ``atol=rtol=1e-2``. Speed
numbers are only meaningful if this passes.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from thunder_vllm.attention.cache_layout import (
    ThunderCacheLayout,
    allocate_kv_cache,
    reshape_and_cache_ref,
)
from thunder_vllm.attention.paged_kv import make_paged_kv_manager
from thunder_vllm.quant.quantizer import ThunderQuantizer

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

ATOL = 1e-2
RTOL = 1e-2

SHAPES = [
    # (seqlen_q, seqlen_k, batch, num_heads, num_kv_heads, head_dim, causal)
    (128, 512, 1, 32, 8, 128, True),
    (128, 2048, 1, 32, 8, 128, True),
    (1, 4096, 1, 32, 8, 128, True),
    (1, 8192, 1, 32, 8, 128, True),
    (1, 2048, 4, 32, 4, 128, True),
    (1, 2048, 1, 32, 1, 128, True),
    (512, 512, 1, 32, 32, 64, True),
    (2048, 8192, 1, 32, 8, 128, True),
]


def _reference(q, k_hat, v_hat, causal, scale):
    """fp32 SDPA on the *dequantized* K/V (GQA expanded)."""
    q_t = q.transpose(0, 1)  # (Hq, Nq, D)
    hq, nq, d = q_t.shape
    hk = k_hat.shape[1]
    group = hq // hk
    k_t = k_hat.transpose(0, 1)  # (Hk, Nk, D)
    v_t = v_hat.transpose(0, 1)
    k_t = k_t.repeat_interleave(group, dim=0)
    v_t = v_t.repeat_interleave(group, dim=0)
    out = F.scaled_dot_product_attention(
        q_t.float().unsqueeze(0),
        k_t.float().unsqueeze(0),
        v_t.float().unsqueeze(0),
        is_causal=causal,
        scale=scale,
    )[0]
    return out.transpose(0, 1)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("k_bits,v_bits", [(4, 4), (3, 4), (2, 2)])
def test_kernel_matches_dequant_reference(causal, k_bits, v_bits):
    from thunder_vllm.attention.cute_kernel import (
        KernelNotReadyError,
        launch_thunder_attention,
    )

    torch.manual_seed(0)
    device = "cuda"
    nq, nk, batch, hq, hk, d = 128, 2048, 1, 32, 8, 128
    scale = d**-0.5

    q = torch.randn(nq, hq, d, device=device, dtype=torch.float16)
    k = torch.randn(nk, hk, d, device=device, dtype=torch.float16)
    v = torch.randn(nk, hk, d, device=device, dtype=torch.float16)

    quant = ThunderQuantizer(d, k_bits, v_bits, device=device)
    layout = ThunderCacheLayout(
        num_kv_heads=hk, head_dim=d, k_bits=k_bits, v_bits=v_bits, block_size=16
    )
    nb = (nk + 15) // 16
    kv, scales = allocate_kv_cache(nb, 16, hk, d, k_bits, v_bits, device=device)
    slots = torch.arange(nk, device=device, dtype=torch.long)

    kv_flat = kv.reshape(-1, layout.kv_slot_bytes)
    scales_flat = scales.reshape(-1, hk, 2)
    reshape_and_cache_ref(
        k, v, slots,
        kv_flat.reshape(nb, 16, layout.kv_slot_bytes),
        scales_flat.reshape(nb, 16, hk, 2),
        quant,
        layout,
    )

    k_hat = quant.dequantize_k(
        layout.k_codes(kv).reshape(nb * 16, hk, layout.k_packed_bytes)[:nk],
        scales_flat.reshape(nb * 16, hk, 2)[:nk, :, 0],
    )
    v_hat = quant.dequantize_v(
        layout.v_codes(kv).reshape(nb * 16, hk, layout.v_packed_bytes)[:nk],
        scales_flat.reshape(nb * 16, hk, 2)[:nk, :, 1],
    )

    ref = _reference(q, k_hat, v_hat, causal, scale)
    out = torch.zeros_like(q)

    try:
        out = _run_kernel(
            q, kv, scales, layout, launch_thunder_attention, quant,
            nq, nk, hq, hk, d, scale, causal,
        )
    except KernelNotReadyError:
        pytest.skip("kernel not ready")

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def _run_kernel(q, kv, scales, layout, launcher, quant, nq, nk, hq, hk, d, scale, causal):
    """Mirror the plugin contract: rotate Q in, un-rotate O out.

    The kernel scores in the rotated basis (exact, since the rotation is
    orthonormal) and accumulates ``P @ (R V)``, so the caller must apply ``R^T``
    to the output -- which ``ThunderAttentionImpl.forward`` does as its output
    projection GEMM.
    """
    block_table = torch.arange(
        (nk + 15) // 16, device=q.device, dtype=torch.int32
    ).unsqueeze(0)
    mgr = make_paged_kv_manager(
        layout, max_num_reqs=1, max_model_len=nk, device=q.device
    )
    gathered = mgr.gather_packed_tiles(block_table, kv, scales)
    metadata = type(
        "M",
        (),
        {
            "num_reqs": 1,
            "num_actual_tokens": nq,
            "max_query_len": nq,
            "is_prefill": nq > 1,
            "max_blocks_per_req": block_table.shape[1],
            "block_table": block_table,
            "seq_lens": torch.tensor([nk], device=q.device, dtype=torch.int32),
            "slot_mapping": torch.arange(nq, device=q.device, dtype=torch.long),
        },
    )()
    from thunder_vllm.attention.cute_kernel import ThunderAttentionForward

    kernel = ThunderAttentionForward(
        head_dim=d, K_BITS=layout.k_bits, V_BITS=layout.v_bits,
        qhead_per_kvhead=hq // hk, is_causal=causal,
        m_block_size=64, n_block_size=64, num_threads=128,
    )
    rot = quant.rotation
    q_rot = rot.rotate(q.float()).to(q.dtype)
    out = torch.zeros_like(q)
    launcher(kernel, q_rot, gathered, out, metadata, scale, quantizer=quant)
    return rot.inverse(out.float()).to(q.dtype)
