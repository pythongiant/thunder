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
    assert decode_split_count(32768, 1, 32, is_prefill=True, num_kv_groups=4) == 1
    assert decode_split_count(32768, 1, 32, is_prefill=False, num_kv_groups=1) == 1
    assert splits_allowed(True, 4) is False
    assert splits_allowed(False, 1) is False
    assert splits_allowed(False, 4) is True


def test_decode_gqa_splits_like_choose_split_count():
    assert decode_split_count(32768, 1, 32, is_prefill=False, num_kv_groups=4) == 4
    assert decode_split_count(4096, 1, 32, is_prefill=False, num_kv_groups=4) == 4
    assert decode_split_count(64, 1, 32, is_prefill=False, num_kv_groups=4) == 1


def test_defaults_are_conservative():
    # No information -> never split.
    assert choose_split_count(0, 1, 32) == 1
    assert choose_split_count(-5, 1, 32) == 1
    assert choose_split_count(4096, 0, 32) == 1


def test_qwen3_8b_batch1_knee_is_four():
    # 32 q-heads, batch 1: the measured knee is 128 CTAs == S=4, and S is
    # frozen at 4 regardless of how long the context gets.
    for seq in (4096, 8192, 16384, 32768):
        assert choose_split_count(seq, 1, 32) == 4


def test_more_heads_needs_less_splitting():
    # Larger batches already supply the CTAs, so no split is needed.
    assert choose_split_count(32768, 1024, 32) == 1
    assert choose_split_count(32768, 256, 32) == 1


def test_never_finer_than_two_tiles():
    # 1 tile total -> cannot split (min_tiles_per_split=2).
    assert choose_split_count(64, 1, 32) == 1
    # 4 tiles with a huge target: capped by tiles//2 == 2.
    assert choose_split_count(256, 1, 32) == 2


def test_powers_of_two_and_bounded():
    for seq in (1024, 4096, 20000, 32768):
        for batch in (1, 2, 8):
            for heads in (8, 16, 32):
                s = choose_split_count(seq, batch, heads)
                assert s in (1, 2, 4), s
