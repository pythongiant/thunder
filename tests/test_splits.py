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
    # S=8 is best or tied at every batch measured on B200 at 32k: batch 1
    # (-25% against unsplit), 16 (-13%) and 64 (-1.8%). The rule this replaced
    # targeted a CTA count and returned 1 at batch 16, where splitting still
    # measured 13% faster.
    for seq in (1024, 4096, 8192, 16384, 32768):
        assert decode_split_count(seq, is_prefill=False, num_kv_groups=4) == 8
        assert choose_split_count(seq) == 8


def test_short_contexts_are_capped_by_the_tile_budget():
    # Never finer than two tiles per split: an empty split is pure merge cost.
    assert choose_split_count(64) == 1          # 1 tile
    assert choose_split_count(256) == 2         # 4 tiles
    assert choose_split_count(1024) == 8        # 16 tiles
    assert choose_split_count(0) == 1
    assert choose_split_count(-5) == 1


def test_counts_are_powers_of_two_and_bounded():
    for seq in (64, 128, 256, 1024, 4096, 20000, 32768):
        s = choose_split_count(seq)
        assert s in (1, 2, 4, 8), s
