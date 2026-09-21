"""Host-side launch-shape heuristics (cutlass-free, CPU-testable)."""

from __future__ import annotations

import math


def choose_split_count(
    seq_len: int,
    batch: int,
    num_q_heads: int,
    *,
    tile_n: int = 64,
    target_ctas: int = 128,
    max_splits: int = 4,
    min_tiles_per_split: int = 2,
) -> int:
    """Smallest split-K count that reaches roughly ``target_ctas`` CTAs.

    Measured on B200/Qwen3-8B: the split-K knee is ~128 CTAs (S=4 at 32 q-heads,
    batch 1); S=8/16 add nothing and S=32 only ~10-15% at the longest contexts,
    so decode is FROZEN at ``max_splits=4``. Counts stay powers of two so the
    number of distinct captured launches stays small, and a split is never
    finer than ``min_tiles_per_split`` tiles (an empty split is pure merge cost).
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
