"""Tile-shape policy by schedule (measured on B200, not guessed).

Decode runs one query row per request, so an M tile of 64 rows is almost
entirely padding: the QK and PV MMAs, the score/probability staging and the row
loops all scale with it. Measured at 32k context with split-K at 8:

| tile (m, threads, n) | decode batch 1 | decode batch 16 | prefill 4k |
|---|---|---|---|
| 64, 128, 64 | 1.392 ms | 20.977 ms | 5.674 ms |
| 64, 128, 32 | — | 19.061 ms | — |
| 32, 64, 64 | 1.263 ms | 18.719 ms | 9.275 ms |
| **32, 64, 32** | **1.229 ms** | **12.153 ms** | — |
| 16, 32, 32 | — | 14.624 ms | — |
| 64, 128, 128 | — | 32.215 ms | — |

So decode wants a quarter-sized tile and prefill wants the full one: a small
tile costs prefill 64% (it has thousands of live query rows and pays for the
extra q-blocks), while it saves decode 12% at batch 1 and 42% at batch 16. The
two are separate schedules in the engine already (``is_causal``), so the shape
is chosen from that rather than from a global config knob.

Note this reverses the earlier finding that a small tile loses at batch 1: that
was measured before split-K went to 8 and before the load/staging work, when
fewer CTAs meant fewer warps to hide the smaller tile's shorter thread count.
"""

from __future__ import annotations

# (m_block, n_block, num_threads). The kernel requires m_block == num_warps * 16,
# so 32 rows go with 64 threads and 64 rows with 128.
DECODE_TILE = {"m_block": 32, "n_block": 32, "num_threads": 64}
PREFILL_TILE = {"m_block": 64, "n_block": 64, "num_threads": 128}


def tile_shape(is_prefill: bool) -> dict[str, int]:
    """Tile shape for a step: the decode schedule or the prefill schedule.

    ``is_prefill`` is the engine's "this step is prefill-like" flag
    (``is_causal = is_prefill or max_query_len > 1``); a one-token decode step
    takes the decode shape even when the attention itself is causal.
    """
    return PREFILL_TILE if is_prefill else DECODE_TILE
