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


def choose_split_count(
    seq_len: int,
    *,
    tile_n: int = 64,
    max_splits: int = 8,
    min_tiles_per_split: int = 2,
) -> int:
    """Split-K count for a decode step: as fine as the tile budget allows.

    Measured on B200/Qwen3-8B at 32k context, ``max_splits=8`` is best or tied at
    every batch tested -- 1 (-25% against unsplit), 16 (-13%) and 64 (-1.8%) --
    so the count no longer targets a CTA number. The rule it replaces
    (``target_ctas=256``) returned 1 at batch 16 because the grid already had 512
    CTAs, yet splitting there still measured 13% faster: CTAs from the batch do
    not substitute for a shorter per-CTA KV walk.

    A split is never finer than ``min_tiles_per_split`` tiles (an empty split is
    pure merge cost), which is what caps the count at short contexts.
    """
    if seq_len <= 0:
        return 1
    n_tiles = math.ceil(seq_len / tile_n)
    return max(1, min(max_splits, n_tiles // max(min_tiles_per_split, 1)))


def decode_split_count(
    seq_len: int,
    *,
    is_prefill: bool,
    num_kv_groups: int,
    tile_n: int = 64,
) -> int:
    """Split-K count for a decode step under the engine policy.

    Single source of truth for the engine and for the benchmark harness, so a
    measured run is the shipped configuration. Note the engine only reaches this
    with ``is_prefill`` False: a one-token decode step is not causal in the
    kernel's sense (``backend.py`` derives ``is_causal = is_prefill or
    max_query_len > 1``), so a causal *attention* shape still splits when it is a
    one-row decode.
    """
    if not splits_allowed(is_prefill, num_kv_groups):
        return 1
    return choose_split_count(seq_len, tile_n=tile_n)
