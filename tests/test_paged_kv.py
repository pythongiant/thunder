"""Paged-KV gather tests (CPU).

Verifies the gather layer against a Python loop that walks the block table
directly, so the gather can be changed (or replaced by an in-kernel TMA walk)
with this test as the invariant.
"""

from __future__ import annotations

import pytest
import torch

from thunder_vllm.attention.cache_layout import (
    ThunderCacheLayout,
    allocate_kv_cache,
    reshape_and_cache_ref,
)
from thunder_vllm.attention.paged_kv import PagedKVManager, make_paged_kv_manager
from thunder_vllm.quant.quantizer import ThunderQuantizer


def _fill_cache(layout, q, kv, scales, n_tokens, seed=0):
    torch.manual_seed(seed)
    key = torch.randn(n_tokens, layout.num_kv_heads, layout.head_dim, dtype=torch.float16)
    value = torch.randn(n_tokens, layout.num_kv_heads, layout.head_dim, dtype=torch.float16)
    slots = torch.arange(n_tokens, dtype=torch.long) % (kv.shape[0] * layout.block_size)
    reshape_and_cache_ref(key, value, slots, kv, scales, q, layout)
    return key, value, slots


def test_gather_matches_block_table_walk():
    layout = ThunderCacheLayout(
        num_kv_heads=4, head_dim=128, k_bits=4, v_bits=4, block_size=16
    )
    q = ThunderQuantizer(128, 4, 4)
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
    layout = ThunderCacheLayout(
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
    layout = ThunderCacheLayout(
        num_kv_heads=2, head_dim=64, k_bits=4, v_bits=4, block_size=16
    )
    q = ThunderQuantizer(64, 4, 4)
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
    layout = ThunderCacheLayout(
        num_kv_heads=1, head_dim=64, k_bits=4, v_bits=4, block_size=32
    )
    mgr = make_paged_kv_manager(layout, max_num_reqs=4, max_model_len=100, device="cpu")
    assert mgr.max_blocks_per_req == 4  # ceil(100 / 32)
    assert mgr.max_page_rows == 16


def test_seq_row_counts():
    layout = ThunderCacheLayout(
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
    layout = ThunderCacheLayout(
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


def test_gather_csr_packs_live_blocks_and_bounds_capacity():
    """CSR gather: packed per-request tokens, capacity <= physical blocks."""
    layout = ThunderCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=4, v_bits=4, block_size=16
    )
    nb = 40
    kv, scales = allocate_kv_cache(nb, 16, 8, 128, 4, 4, device="cpu")
    mgr = PagedKVManager(layout, max_num_reqs=64, max_blocks_per_req=256, device="cpu")

    # Shuffled physical ids so a logical==physical assumption would fail.
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(nb, generator=gen)[:16].to(torch.int32)
    table = perm.reshape(4, 4)  # 4 requests, 4 logical blocks each
    blocks_per_req = [1, 4, 2, 3]
    seq = torch.tensor([b * 16 for b in blocks_per_req])

    out = mgr.gather_csr(table, kv, scales, blocks_per_req)
    nrows = sum(blocks_per_req) * 16
    assert nrows <= nb * 16
    assert mgr.page_rows == min(mgr.max_page_rows, nb) == nb
    assert out.k_packed.shape[0] == nb

    indptr = mgr.indptr.tolist()
    assert indptr == [0] + list(torch.tensor(blocks_per_req).cumsum(0).mul(16).tolist())

    # Packed rows equal the physical cache content in CSR order.
    k_codes = layout.k_codes(kv)
    off = 0
    for r, bn in enumerate(blocks_per_req):
        for lb in range(bn):
            phys = int(table[r, lb])
            assert torch.equal(
                out.k_packed.view(-1, 16, 8, layout.k_packed_bytes)[off],
                k_codes[phys],
            )
            off += 1
    assert off == nrows // 16


def test_build_csr_device_matches_host_build():
    """Phase B: device-side CSR build must equal the host build (indptr + sel)."""
    layout = ThunderCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=4, v_bits=4, block_size=16
    )
    nb = 40
    kv, scales = allocate_kv_cache(nb, 16, 8, 128, 4, 4, device="cpu")
    mgr = PagedKVManager(layout, max_num_reqs=8, max_blocks_per_req=4, device="cpu")
    gen = torch.Generator().manual_seed(3)
    table = torch.randperm(nb, generator=gen)[:8 * 4].reshape(8, 4).to(torch.int32)

    class _MD:
        pass
    md = _MD()
    md.block_table = table
    md.seq_lens = torch.tensor([1, 16, 17, 64, 33, 128, 5, 80])
    md.seq_lens_cpu = md.seq_lens
    md.num_actual_tokens = 8
    md.max_query_len = 1
    md.is_prefill = False

    host = mgr._build_csr_index(table, mgr._blocks_per_req(md), nb)
    dev = mgr.build_csr_device(md, nb)
    assert torch.equal(host.indptr, dev.indptr)
    assert host.nrows == dev.nrows
    assert torch.equal(host.sel[:host.nrows], dev.sel[:host.nrows])


def test_pack3_word_formula_matches_pack_indices():
    """The Triton 3-bit packer builds word = sum(idx<<3c) over 8 columns and
    slices bytes; verify that equals quant.packing.pack_indices."""
    from thunder_vllm.quant.packing import pack_indices

    torch.manual_seed(0)
    head_dim = 128
    ng = head_dim // 8
    idx = torch.randint(0, 8, (5, head_dim), dtype=torch.int32)

    # word-based packing (same math as the Triton kernel)
    s = idx.reshape(5, ng, 8).to(torch.int32)
    w = (1 << (3 * torch.arange(8, dtype=torch.int32)))
    word = (s * w[None, None, :]).sum(dim=2)                    # (5, ng)
    sub = torch.arange(3, dtype=torch.int32)
    byte = ((word[:, :, None] >> (8 * sub)[None, None, :]) & 0xFF)
    got = byte.reshape(5, ng * 3).to(torch.uint8)

    want = pack_indices(idx, 3, head_dim)
    assert torch.equal(got, want), (got[:1, :12], want[:1, :12])


def test_direct_paged_tile_rows_match_csr_sel():
    """Option A: block-table addressing must reproduce the CSR path's rows.

    CSR `sel` holds BLOCK ids; the physical token row for tile position ``i`` is
    ``sel[block_index]*bs + (token % bs)``. The direct-paged kernel computes
    ``block_table[req, token//bs]*bs + token%bs`` and must land on the same rows.
    Validating this on CPU de-risks the addressing change before the kernel is
    rewritten.
    """
    from thunder_vllm.attention.paged_kv import direct_paged_tile_rows

    layout = ThunderCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=3, v_bits=4, block_size=16
    )
    nb, bs = 64, 16
    kv, scales = allocate_kv_cache(nb, bs, 8, 128, 3, 4, device="cpu")
    mgr = PagedKVManager(layout, max_num_reqs=4, max_blocks_per_req=16, device="cpu")

    gen = torch.Generator().manual_seed(5)
    table = torch.randperm(nb, generator=gen)[:4 * 16].reshape(4, 16).to(torch.int32)
    seq_lens = torch.tensor([256, 200, 96, 48])
    blocks_per_req = [int((s + bs - 1) // bs) for s in seq_lens]

    idx = mgr._build_csr_index(table, blocks_per_req, nb)
    sel = idx.sel                      # block ids, req-major, block-major

    tile_n = 64
    for req, nblocks in enumerate(blocks_per_req):
        base_block = sum(blocks_per_req[:req])
        for nt in range(nblocks * bs // tile_n):
            tokens = nt * tile_n + torch.arange(tile_n)
            csr_rows = sel[base_block + tokens // bs].to(torch.int64) * bs + (tokens % bs)
            want = direct_paged_tile_rows(table, req, nt, tile_n, bs)
            assert torch.equal(csr_rows, want), (req, nt, csr_rows[:4], want[:4])


def test_direct_paged_tile_bytes_match_csr_gather():
    """End-to-end: reading the raw cache via the block table yields the same
    packed bytes the CSR gather produced (the P0-b invariant)."""
    from thunder_vllm.attention.paged_kv import direct_paged_tile_rows

    layout = ThunderCacheLayout(
        num_kv_heads=8, head_dim=128, k_bits=3, v_bits=4, block_size=16
    )
    nb, bs = 64, 16
    kv, scales = allocate_kv_cache(nb, bs, 8, 128, 3, 4, device="cpu")
    mgr = PagedKVManager(layout, max_num_reqs=4, max_blocks_per_req=16, device="cpu")
    gen = torch.Generator().manual_seed(9)
    table = torch.randperm(nb, generator=gen)[:4 * 16].reshape(4, 16).to(torch.int32)
    blocks_per_req = [16, 13, 6, 3]

    out = mgr.gather_csr(table, kv, scales, blocks_per_req)
    k_codes_flat = layout.k_codes(kv).reshape(-1, 8, layout.k_packed_bytes)
    gathered_flat = out.k_packed.reshape(-1, 8, layout.k_packed_bytes)

    tile_n = 64
    for req, nblocks in enumerate(blocks_per_req):
        base_token = sum(blocks_per_req[:req]) * bs
        for nt in range(nblocks * bs // tile_n):
            rows = direct_paged_tile_rows(table, req, nt, tile_n, bs)
            got = k_codes_flat[rows]
            want = gathered_flat[base_token + nt * tile_n: base_token + nt * tile_n + tile_n]
            assert torch.equal(got, want), (req, nt)
