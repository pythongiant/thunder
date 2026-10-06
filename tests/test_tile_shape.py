"""Regression tests for the tile-shape policy (CPU, cutlass-free)."""

from __future__ import annotations

from thunder_vllm.attention.tile_shape import (
    BATCHED_DECODE_FROM,
    DECODE_TILE,
    DECODE_TILE_BATCHED,
    PREFILL_TILE,
    PREFILL_TILE_CHUNKED,
    tile_shape,
)


def test_decode_and_prefill_use_different_tiles():
    # Measured at 32k with split-K 8: the decode tile (32 rows, two warps, 32-wide
    # KV tile) is 12% faster at batch 1 and 42% at batch 16, while a small tile
    # costs prefill 64%. Pin the split so a refactor cannot quietly collapse them.
    assert tile_shape(is_prefill=False) == DECODE_TILE
    assert tile_shape(is_prefill=True) == PREFILL_TILE
    assert len({str(DECODE_TILE), str(PREFILL_TILE), str(DECODE_TILE_BATCHED)}) == 3


def test_batched_decode_switches_tile_at_the_measured_batch():
    # Measured: at batch 16 the 16-row tile is 4.3% faster, at batch 1 it is 34%
    # slower, so the switch waits for the batch that was actually measured.
    assert tile_shape(is_prefill=False, num_reqs=BATCHED_DECODE_FROM - 1) == DECODE_TILE
    assert tile_shape(is_prefill=False, num_reqs=BATCHED_DECODE_FROM) == DECODE_TILE_BATCHED
    # Prefill ignores the batch entirely.
    assert tile_shape(is_prefill=True, num_reqs=1024) == PREFILL_TILE


def test_tile_satisfies_the_kernel_layout_rule():
    # The kernel asserts m_block == num_warps * 16; a tile that violates it fails
    # at compile time with an unhelpful message, so catch it here.
    for tile in (DECODE_TILE, PREFILL_TILE, DECODE_TILE_BATCHED):
        assert tile["m_block"] == (tile["num_threads"] // 32) * 16, tile

def test_chunked_prefill_takes_the_smaller_tile():
    """vLLM chunks long prompts, so an engine prefill usually has only a few query
    rows per request and 16 live rows in a 64-row tile waste three quarters of the
    QK/PV and staging work. At the engine's 4k geometry (256 requests x 16 rows) the
    full-height tile measures 34.07 ms per layer against 23.86 ms for the 32-row one.
    """
    assert tile_shape(True, 256, 16) == PREFILL_TILE_CHUNKED
    assert tile_shape(True, 256, 32) == PREFILL_TILE_CHUNKED
    assert tile_shape(True, 256, 33) == PREFILL_TILE
    assert tile_shape(True, 1, 4096) == PREFILL_TILE
    assert tile_shape(True, 1, None) == PREFILL_TILE  # unknown keeps full height
