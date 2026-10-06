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
    """fp32 SDPA on the *dequantized* K/V (GQA expanded).

    The causal mask is bottom-right aligned (query row ``i`` sees every key up to
    ``(S - L) + i``). That is the kernel's ``_valid`` rule -- ``kv <= (kv_len -
    q_len) + q_off + row`` -- and it is what a continuation prefill means. Passing
    ``is_causal=True`` to ``F.scaled_dot_product_attention`` instead applies an
    upper-left mask whenever ``L != S``, which for a 128-query/2048-key prefill
    exposes only the first 128 keys and for a one-row decode exposes a single
    key, i.e. it stops describing the operator under test.
    """
    q_t = q.transpose(0, 1)  # (Hq, Nq, D)
    hq, nq, d = q_t.shape
    hk = k_hat.shape[1]
    group = hq // hk
    k_t = k_hat.transpose(0, 1)  # (Hk, Nk, D)
    v_t = v_hat.transpose(0, 1)
    k_t = k_t.repeat_interleave(group, dim=0)
    v_t = v_t.repeat_interleave(group, dim=0)
    mask = None
    if causal:
        sk = k_t.shape[1]
        q_pos = torch.arange(nq, device=q.device).view(nq, 1) + (sk - nq)
        kv_pos = torch.arange(sk, device=q.device).view(1, sk)
        mask = kv_pos <= q_pos
    out = F.scaled_dot_product_attention(
        q_t.float().unsqueeze(0),
        k_t.float().unsqueeze(0),
        v_t.float().unsqueeze(0),
        attn_mask=mask,
        scale=scale,
    )[0]
    return out.transpose(0, 1)


def _build_case(nq, nk, hq, hk, d, k_bits=4, v_bits=4, causal=True, block_size=16,
                device="cuda"):
    """Random K/V -> quantized paged cache -> ``(q, kv, scales, layout, quant, oracle)``.

    ``block_size`` must stay 16: :func:`_run_kernel` derives its block table from
    that literal. The cache is handed over in the layout's 4-D canonical form
    ``(blocks, Hk, block_size, slot)``; ``k_codes``/``k_norm`` are the only
    transpose boundary, so never reshape it into a "flat" 3-D view by hand.
    """
    torch.manual_seed(0)
    q = torch.randn(nq, hq, d, device=device, dtype=torch.float16)
    k = torch.randn(nk, hk, d, device=device, dtype=torch.float16)
    v = torch.randn(nk, hk, d, device=device, dtype=torch.float16)

    quant = ThunderQuantizer(d, k_bits, v_bits, device=device)
    layout = ThunderCacheLayout(
        num_kv_heads=hk, head_dim=d, k_bits=k_bits, v_bits=v_bits, block_size=block_size
    )
    nb = (nk + block_size - 1) // block_size
    kv, scales = allocate_kv_cache(nb, block_size, hk, d, k_bits, v_bits, device=device)
    slots = torch.arange(nk, device=device, dtype=torch.long)
    reshape_and_cache_ref(k, v, slots, kv, scales, quant, layout)

    rows = nb * block_size
    k_hat = quant.dequantize_k(
        layout.k_codes(kv).reshape(rows, hk, layout.k_packed_bytes)[:nk],
        layout.k_norm(scales).reshape(rows, hk)[:nk],
    )
    v_hat = quant.dequantize_v(
        layout.v_codes(kv).reshape(rows, hk, layout.v_packed_bytes)[:nk],
        layout.v_norm(scales).reshape(rows, hk)[:nk],
    )
    return q, kv, scales, layout, quant, _reference(q, k_hat, v_hat, causal, d**-0.5)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("k_bits,v_bits", [(4, 4), (3, 4), (2, 2)])
def test_kernel_matches_dequant_reference(causal, k_bits, v_bits):
    from thunder_vllm.attention.cute_kernel import (
        KernelNotReadyError,
        launch_thunder_attention,
    )

    nq, nk, hq, hk, d = 128, 2048, 32, 8, 128
    scale = d**-0.5
    q, kv, scales, layout, quant, ref = _build_case(
        nq, nk, hq, hk, d, k_bits, v_bits, causal
    )
    out = torch.zeros_like(q)

    try:
        out = _run_kernel(
            q, kv, scales, layout, launch_thunder_attention, quant,
            nq, nk, hq, hk, d, scale, causal,
        )
    except KernelNotReadyError:
        pytest.skip("kernel not ready")

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def _run_kernel(q, kv, scales, layout, launcher, quant, nq, nk, hq, hk, d, scale, causal,
                num_splits: int = 1):
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
            "query_start_loc": torch.tensor([0, nq], device=q.device, dtype=torch.int32),
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
    launcher(
        kernel, q_rot, gathered, out, metadata, scale,
        quantizer=quant, num_splits=num_splits,
    )
    return rot.inverse(out.float()).to(q.dtype)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("num_splits", [2, 4])
def test_split_k_decode_matches_dequant_reference(causal, num_splits):
    """Split-K decode parity against the dequant oracle.

    Splitting is a distinct reduction -- per-split max/sum rescale in the
    kernel plus a host-side merge and scatter -- and causal decode is exactly
    where the shipped policy declines to split, so no other test covers it.
    """
    from thunder_vllm.attention.cute_kernel import (
        KernelNotReadyError,
        launch_thunder_attention,
    )

    nq, nk, hq, hk, d = 1, 8192, 32, 8, 128
    scale = d**-0.5
    q, kv, scales, layout, quant, ref = _build_case(nq, nk, hq, hk, d, 4, 4, causal)

    try:
        out = _run_kernel(
            q, kv, scales, layout, launch_thunder_attention, quant,
            nq, nk, hq, hk, d, scale, causal, num_splits=num_splits,
        )
    except KernelNotReadyError:
        pytest.skip("kernel not ready")

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
