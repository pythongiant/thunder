"""Tile-shape policy by schedule (measured on B200, not guessed).

Decode runs one query row per request, so an M tile of 64 rows is almost
entirely padding: the QK and PV MMAs, the score/probability staging and the row
loops all scale with it. Prefill has thousands of live query rows and pays for
the opposite: a small M tile means more q-blocks, and a wide KV tile means more
masked-out work per tile.

| tile (m, threads, n) | decode b=1, 32k | decode b=16, 32k | prefill 4k |
|---|---|---|---|
| 64, 128, 16 | — | — | **4.53 ms** |
| 64, 128, 32 | — | — | 5.08 ms |
| 64, 128, 64 | — | — | 5.69 ms |
| 64, 128, 128 | — | — | 8.85 ms |
| 128, 256, 64 | — | — | 5.60 ms |
| 32, 64, 32 | **0.215 ms** | — | — |
| 16, 32, 16 | 0.924 ms | **2.72 ms** | — |

So the schedules want different tiles in every dimension: decode takes the
smallest M tile that holds its live rows, prefill takes a full-height M tile with
the narrowest KV tile that builds (8 does not; the MMA floor is 16). ``t`` is not
free: the kernel requires ``m_block == num_threads // 32 * 16``, so 64 rows go
with 128 threads and 128 rows with 256. The two are separate schedules in the
engine already (``is_causal``), so the shape is chosen from that rather than from
a global config knob.

The decode column is with GQA packing and split-K at the knee; the same decode
tiles measured 1.229 ms (batch 1) and 12.153 ms (batch 16) before those two
changes, which is why the batch-1 choice reversed -- see ``splits.py``.

The prefill column is also with packing on (``GQA`` prefill packs the group's
heads over ``tile_m // qhead_per_kvhead`` tokens, see ``cute_kernel.py``), and
the prefill tile did not have to move for it: 4k measures 4.45 ms at 64x128x16
against 5.14 ms at 64x128x32 and 7.17 ms at 32x64x16. A narrower M tile shrinks
the packed q-block, and the causal bound then trims less per CTA, which costs
more than the extra KV reuse buys.
"""

from __future__ import annotations

# (m_block, n_block, num_threads). The kernel requires m_block == num_warps * 16,
# so 32 rows go with 64 threads and 64 rows with 128.
DECODE_TILE = {"m_block": 32, "n_block": 32, "num_threads": 64}
PREFILL_TILE = {"m_block": 64, "n_block": 16, "num_threads": 128}
# Batched decode fills the M tile with one live row per request: at batch 16 the
# 32-row tile is half live and the 16-row tile is exactly live, which measures
# 4.3% faster (11.63 vs 12.15 ms at 32k). At batch 1 the same tile is 34% slower,
# so the switch is gated on the batch rather than applied globally.
DECODE_TILE_BATCHED = {"m_block": 16, "n_block": 16, "num_threads": 32}
BATCHED_DECODE_FROM = 16
# A prefill with at most this many query rows per request is a chunk of a longer
# prompt (vLLM's chunked prefill), not a whole prompt.
CHUNKED_PREFILL_MAX_Q = 32
PREFILL_TILE_CHUNKED = {"m_block": 32, "n_block": 16, "num_threads": 64}


def tile_shape(is_prefill: bool, num_reqs: int = 1,
               max_query_len: int | None = None) -> dict[str, int]:
    """Tile shape for a step: prefill, batched decode, or plain decode.

    ``is_prefill`` is the engine's "this step is prefill-like" flag
    (``is_causal = is_prefill or max_query_len > 1``); a one-token decode step
    takes a decode shape even when the attention itself is causal.
    ``num_reqs`` only matters for decode, and only above the measured threshold
    (batch 1 and 16 were measured; nothing in between was, so the switch waits
    for the batch that was actually measured).
    """
    if is_prefill:
        # vLLM chunks long prompts, so an engine prefill is usually a
        # many-request step with only a few query rows per request. At the
        # engine's 4k geometry (256 requests x 16 rows) the full-height tile
        # measures 34.07 ms per layer against 23.86 ms for the 32-row one --
        # -30% -- because 16 live rows inside a 64-row tile waste three quarters
        # of the QK/PV and staging work. The grid shape is faithful: the kernel
        # anchors each request's query rows at the END of its context (see
        # _valid in cute_kernel.py), so those 16 rows attend the whole 4096-token
        # prefix exactly as an engine chunk does. A genuinely long prefill keeps the
        # full-height tile (it has thousands of live rows and pays for the extra
        # q-blocks otherwise).
        if max_query_len is not None and max_query_len <= CHUNKED_PREFILL_MAX_Q:
            return PREFILL_TILE_CHUNKED
        return PREFILL_TILE
    if num_reqs >= BATCHED_DECODE_FROM:
        return DECODE_TILE_BATCHED
    return DECODE_TILE
