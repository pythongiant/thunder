"""Regression tests for the tile-shape policy (CPU, cutlass-free)."""

from __future__ import annotations

from thunder_vllm.attention.tile_shape import DECODE_TILE, PREFILL_TILE, tile_shape


def test_decode_and_prefill_use_different_tiles():
    # Measured at 32k with split-K 8: the decode tile (32 rows, two warps, 32-wide
    # KV tile) is 12% faster at batch 1 and 42% at batch 16, while a small tile
    # costs prefill 64%. Pin the split so a refactor cannot quietly collapse them.
    assert tile_shape(is_prefill=False) == DECODE_TILE
    assert tile_shape(is_prefill=True) == PREFILL_TILE
    assert DECODE_TILE != PREFILL_TILE


def test_tile_satisfies_the_kernel_layout_rule():
    # The kernel asserts m_block == num_warps * 16; a tile that violates it fails
    # at compile time with an unhelpful message, so catch it here.
    for tile in (DECODE_TILE, PREFILL_TILE):
        assert tile["m_block"] == (tile["num_threads"] // 32) * 16, tile
