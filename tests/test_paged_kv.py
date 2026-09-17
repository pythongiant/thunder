"""Paged-KV gather tests (CPU).

Verifies the gather layer against a Python loop that walks the block table
directly, so the gather can be changed (or replaced by an in-kernel TMA walk)
with this test as the invariant.
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.attention.cache_layout import (
    TurboQuantCacheLayout,
    allocate_kv_cache,
    reshape_and_cache_ref,
)
from turboquant_vllm.attention.paged_kv import PagedKVManager, make_paged_kv_manager
from turboquant_vllm.quant.quantizer import TurboQuantQuantizer


def _fill_cache(layout, q, kv, scales, n_tokens, seed=0):
    torch.manual_seed(seed)
    key = torch.randn(n_tokens, layout.num_kv_heads, layout.head_dim, dtype=torch.float16)
    value = torch.randn(n_tokens, layout.num_kv_heads, layout.head_dim, dtype=torch.float16)
    slots = torch.arange(n_tokens, dtype=torch.long) % (kv.shape[0] * layout.block_size)
    reshape_and_cache_ref(key, value, slots, kv, scales, q, layout)
    return key, value, slots


def test_gather_matches_block_table_walk():
    layout = TurboQuantCacheLayout(
        num_kv_heads=4, head_dim=128, k_bits=4, v_bits=4, block_size=16
    )
    q = TurboQuantQuantizer(128, 4, 4)
    nb = 8
    kv, scales = allocate_kv_cache(nb, 16, 4, 128, 4, 4, device="cpu")
    key, _, _ = _fill_cache(layout, q, kv, scales, nb * 16)

    block_table = torch.tensor(
        [[3, 0, 5], [7, 2, 0]], dtype=torch.long  # 2 requests, 3 blocks each
    )
    mgr = PagedKVManager(
        layout, max_num_reqs=3, max_blocks_per_req=3, device="cpu"
    )
    got = mgr.gather_packed_tiles_ref(block_table, kv, scales)

    # Reference: walk the block table and concatenate.
    k_ref, v_ref = [], []
    for req in range(block_table.shape[0]):
        for blk in block_table[req].tolist():
            k_ref.append(layout.k_codes(kv)[blk])
            v_ref.append(layout.v_codes(kv)[blk])
    # Pad request 3 with block 0 in the manager; reference only covers 2 rows.
    k_ref = torch.cat(k_ref, dim=0)
    v_ref = torch.cat(v_ref, dim=0)

    assert got.k_packed.shape[0] == 3 * 3  # max_num_reqs * max_blocks_per_req rows
    got_k = got.k_packed.reshape(-1, layout.num_kv_heads, layout.k_packed_bytes)
    got_v = got.v_packed.reshape(-1, layout.num_kv_heads, layout.v_packed_bytes)
    assert torch.equal(got_k[: k_ref.shape[0]], k_ref)
    assert torch.equal(got_v[: v_ref.shape[0]], v_ref)


def test_gather_row_map_and_reserve_identity():
    layout = TurboQuantCacheLayout(
        num_kv_heads=2, head_dim=64, k_bits=4, v_bits=4, block_size=16
    )
    mgr = PagedKVManager(layout, max_num_reqs=2, max_blocks_per_req=4, device="cpu")
    first = mgr.reserve()
    second = mgr.reserve()
    assert first is second, "reserve must be idempotent (pointer stability)"
    bt = torch.tensor([[1, 2, 0, 0], [5, 6, 7, 8]], dtype=torch.long)
    row_map = mgr.block_table_row_map(bt)
    assert row_map.tolist() == [1, 2, 0, 0, 5, 6, 7, 8]


def test_gather_in_place_reuses_buffer():
    layout = TurboQuantCacheLayout(
        num_kv_heads=2, head_dim=64, k_bits=4, v_bits=4, block_size=16
    )
    q = TurboQuantQuantizer(64, 4, 4)
    nb = 4
    kv, scales = allocate_kv_cache(nb, 16, 2, 64, 4, 4, device="cpu")
    _fill_cache(layout, q, kv, scales, nb * 16, seed=7)
    mgr = PagedKVManager(layout, max_num_reqs=2, max_blocks_per_req=2, device="cpu")
    bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    a = mgr.gather_packed_tiles(bt, kv, scales)
    ptr = a.k_packed.data_ptr()
    b = mgr.gather_packed_tiles(bt, kv, scales)
    assert b.k_packed.data_ptr() == ptr, "captured pointers must be stable"


def test_make_manager_block_count():
    layout = TurboQuantCacheLayout(
        num_kv_heads=1, head_dim=64, k_bits=4, v_bits=4, block_size=32
    )
    mgr = make_paged_kv_manager(layout, max_num_reqs=4, max_model_len=100, device="cpu")
    assert mgr.max_blocks_per_req == 4  # ceil(100 / 32)
    assert mgr.max_page_rows == 16


def test_seq_row_counts():
    layout = TurboQuantCacheLayout(
        num_kv_heads=1, head_dim=64, k_bits=2, v_bits=2, block_size=16
    )
    mgr = PagedKVManager(layout, max_num_reqs=4, max_blocks_per_req=4, device="cpu")
    seq = torch.tensor([1, 16, 17, 64])
    assert mgr.seq_row_counts(seq).tolist() == [1, 1, 2, 4]


def test_gather_at_smaller_table_than_reservation():
    """vLLM's warm-up passes a block table that does not match the reservation.

    Regression: the gather used to reshape to a fixed `max_page_rows`, which
    failed with
      RuntimeError: shape '[32768, 16, 8, 64]' is invalid for input of size ...
    The page-row count must come from the (padded) table actually handed in, and
    the in-place variant must copy only the valid rows.
    """
    layout = TurboQuantCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=4, v_bits=4, block_size=16
    )
    nb = 8
    kv, scales = allocate_kv_cache(nb, 16, 8, 128, 4, 4, device="cpu")

    # Reserve for far more requests/blocks than the table we will pass.
    mgr = PagedKVManager(layout, max_num_reqs=32, max_blocks_per_req=64, device="cpu")
    assert mgr.max_page_rows == 32 * 64

    # A table that is smaller in both dimensions but exceeds nothing.
    table = torch.arange(2 * 4, dtype=torch.int32).reshape(2, 4)
    seq = torch.tensor([64, 64])
    out = mgr.gather_packed_tiles(table, kv, scales, seq_lens=seq)

    # In-place path returns the reservation (pointer stability), but only the
    # live region [r, b_live] is gathered; the rest stays stale and is never read.
    assert out.k_packed.shape[0] == mgr.max_page_rows
    ref = mgr.gather_packed_tiles_ref(table, kv, scales, seq_lens=seq)
    assert ref.k_packed.shape[0] == mgr.max_page_rows  # ref still pads

    # seq 64 / block_size 16 -> 4 live blocks per request, 2 requests.
    shape = (mgr.max_num_reqs, mgr.max_blocks_per_req, layout.block_size,
             layout.num_kv_heads, layout.k_packed_bytes)
    vshape = (*shape[:-1], layout.v_packed_bytes)
    assert torch.equal(
        out.k_packed.view(shape)[:2, :4], ref.k_packed.view(shape)[:2, :4]
    )
    assert torch.equal(
        out.v_packed.view(vshape)[:2, :4], ref.v_packed.view(vshape)[:2, :4]
    )
