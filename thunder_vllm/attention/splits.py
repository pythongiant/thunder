"""Host-side launch-shape heuristics (cutlass-free, CPU-testable)."""

from __future__ import annotations

import math


def splits_allowed(is_prefill: bool, num_kv_groups: int) -> bool:
    """Whether split-K decode can pay for its merge at all.

    Prefill (a multi-token query block) and MHA (``num_kv_groups <= 1``) never
    split in the engine: prefill keeps the baseline schedule, and with one CTA
    per head there is no occupancy to buy. Kept separate from
    :func:`choose_split_count` because the caller needs this answer *before*
    resolving ``seq_len`` (which can require a device sync, illegal under CUDA
    graph capture).
    """
    return not is_prefill and num_kv_groups > 1


def decode_split_count(
    seq_len: int,
    num_reqs: int,
    num_q_heads: int,
    *,
    is_prefill: bool,
    num_kv_groups: int,
    tile_n: int = 64,
) -> int:
    """Split-K count for a decode step under the engine policy.

    Single source of truth for the engine and for the benchmark harness, so a
    measured run is the shipped configuration. Note the engine only reaches
    this with ``is_prefill`` False: a one-token decode step is not causal in the
    kernel's sense (``backend.py`` derives ``is_causal = is_prefill or
    max_query_len > 1``), so a causal *attention* shape still splits when it is
    a one-row decode.
    """
    if not splits_allowed(is_prefill, num_kv_groups):
        return 1
    return choose_split_count(seq_len, num_reqs, num_q_heads, tile_n=tile_n)


def choose_split_count(
    seq_len: int,
    batch: int,
    num_q_heads: int,
    *,
    tile_n: int = 64,
    target_ctas: int = 256,
    max_splits: int = 8,
    min_tiles_per_split: int = 2,
) -> int:
    """Smallest split-K count that reaches roughly ``target_ctas`` CTAs.

    Measured on B200/Qwen3-8B, batch 1: the knee is ~256 CTAs (S=8 at 32
    q-heads), not the ~128 it used to be. The kernel's shared-memory footprint
    now fits two CTAs per SM (89 KB, after the output staging buffer went fp16),
    so the extra CTAs actually get resident instead of queueing; before that
    change S=8 measured identical to S=4. S=16 is flat again (3+ CTAs/SM does
    not fit), and S stays capped at 8.

    Counts stay powers of two so the number of distinct captured launches stays
    small, and a split is never finer than ``min_tiles_per_split`` tiles (an
    empty split is pure merge cost).
    """
    if seq_len <= 0 or batch <= 0 or num_q_heads <= 0:
        return 1
    want = math.ceil(target_ctas / (batch * num_q_heads))
    n_tiles = math.ceil(seq_len / tile_n)
    cap = min(max_splits, max(1, n_tiles // max(min_tiles_per_split, 1)))
    splits = 1
    while splits * 2 <= min(want, cap):
        splits *= 2
    return max(1, min(splits, cap, max_splits))
