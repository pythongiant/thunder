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
    # The knee tracks the grid: GQA packing puts KV heads on the head axis, so
    # the CTA count needs 4x the splits -- 64 at 32k, where S=64 measures
    # 0.215 ms against S=16's 0.537 at batch 1, and 128 is flat.
    assert choose_split_count(4096) == 16        # 128 tiles / 8
    assert choose_split_count(8192) == 32        # 256 tiles / 8
    assert choose_split_count(16384) == 64       # 512 tiles / 8
    assert choose_split_count(32768) == 64       # capped at max_splits
    for seq in (4096, 8192, 16384, 32768):
        assert decode_split_count(seq, is_prefill=False, num_kv_groups=4) == \
            choose_split_count(seq)


def test_batched_decode_caps_the_split_count():
    # At batch 16 the grid already carries one CTA per (request, head, split), so
    # extra splits stop buying parallelism and start costing merge work: at 4k,
    # 16 splits measure 0.350 ms against 32 splits' 0.366, and at 32k the curve
    # is flat from 8 to 64 so 16 is free there.
    assert choose_split_count(32768, num_reqs=1) == 64
    assert choose_split_count(32768, num_reqs=16) == 16
    assert choose_split_count(4096, num_reqs=16) == 16


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
        assert s in (1, 2, 4, 8, 16, 32, 64), s
