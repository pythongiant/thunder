"""Regression tests for the decode split-K heuristic (CPU, cutlass-free)."""

from __future__ import annotations

from thunder_vllm.attention.splits import (
    choose_split_count,
    decode_split_count,
    splits_allowed,
)


def test_prefill_and_mha_never_split():
    # The engine refuses split-K for prefill-style steps and for MHA: prefill
    # keeps the baseline schedule and MHA has no occupancy to buy. Pin it.
    assert decode_split_count(32768, is_prefill=True, num_kv_groups=4) == 1
    assert decode_split_count(32768, is_prefill=False, num_kv_groups=1) == 1
    assert splits_allowed(True, 4) is False
    assert splits_allowed(False, 1) is False
    assert splits_allowed(False, 4) is True


def test_decode_splits_as_fine_as_the_tile_budget_allows():
    # The knee tracks the tile shape: 16 at the current decode tile (32 rows,
    # 32-wide KV tiles), where S=16 measures 0.776 ms against S=8's 1.225 at
    # batch 1/32k, while S=32/64 are flat at 32k and worse at 4k.
    for seq in (4096, 8192, 16384, 32768):
        assert decode_split_count(seq, is_prefill=False, num_kv_groups=4) == 16
        assert choose_split_count(seq) == 16


def test_short_contexts_are_capped_by_the_tile_budget():
    # Never finer than eight tiles per split: past that the merge cost dominates
    # (measured: 4k prefers 16 splits over 32, and a 512-token context has no
    # room to split at all).
    assert choose_split_count(64) == 1          # 2 tiles
    assert choose_split_count(256) == 1         # 8 tiles -> 1 split of 8
    assert choose_split_count(1024) == 4        # 32 tiles
    assert choose_split_count(0) == 1
    assert choose_split_count(-5) == 1


def test_counts_are_powers_of_two_and_bounded():
    for seq in (64, 128, 256, 1024, 4096, 20000, 32768):
        s = choose_split_count(seq)
        assert s in (1, 2, 4, 8, 16), s
