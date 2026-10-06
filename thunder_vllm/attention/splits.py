"""Host-side launch-shape heuristics (cutlass-free, CPU-testable)."""

from __future__ import annotations

import math

from thunder_vllm.attention.tile_shape import tile_shape


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
    tile_n: int | None = None,
    max_splits: int = 64,
    min_tiles_per_split: int = 8,
) -> int:
    """Split-K count for a decode step: as fine as the tile budget allows.

    The knee tracks the *grid*, and the grid changed twice. It was 8 at the old
    64-row/64-wide tile, 16 at the current decode tile (32 rows, 32-wide KV
    tiles), and 64 once GQA packing took over the head axis -- that schedule puts
    KV heads (8) on the grid instead of query heads (32), so the same CTA count
    needs 4x the splits. Measured at batch 1/32k: S=16 0.537 ms, S=32 0.318,
    S=64 0.215, S=128 0.218 (flat). At batch 16 the curve is flat from 8 to 32, so
    64 costs nothing there.

    ``min_tiles_per_split=8`` is what keeps short contexts from over-splitting:
    at 4k (128 tiles) it yields 16, which measures 0.077 ms against S=8's 0.142,
    and a 512-token context gets 2.
    """
    if seq_len <= 0:
        return 1
    if tile_n is None:
        # The decode tile's width, from the same policy the kernel is built from
        # (it is 32, not the 64 this defaulted to before the tile change).
        tile_n = tile_shape(is_prefill=False)["n_block"]
    n_tiles = math.ceil(seq_len / tile_n)
    return max(1, min(max_splits, n_tiles // max(min_tiles_per_split, 1)))


def decode_split_count(
    seq_len: int,
    *,
    is_prefill: bool,
    num_kv_groups: int,
    tile_n: int | None = None,
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
