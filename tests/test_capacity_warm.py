"""The capture warm-up's launch metadata (CPU).

`_warm_meta` builds the metadata for the eager launches that pre-compile the
CUDA-graph capture's geometry (``docs/FAILURE_MODES.md`` 14). Every field MUST stay
zero: the kernel is launched with `num_reqs` requests over the step's own gathered
buffers, and any live entry there would make it read real rows (or write the step's
output). Zero means `kv_len == 0` and `q_len == 0`, so the tile loop is
`ceil(0/tile_n) == 0`, the epilogue is guarded by `q_off + tok < q_len`, and the
launch only does what it exists for: compile that (config, shape, grid) and
allocate its split partials.
"""

from __future__ import annotations

import torch

from thunder_vllm.attention.backend import _warm_meta


def _bt():
    return torch.zeros((2, 4), dtype=torch.int32)


def test_warm_meta_is_all_zero():
    md, indptr = _warm_meta(4, 6, _bt(), 4, torch.device("cpu"))
    assert md.num_reqs == 4
    assert md.max_query_len == 1
    assert md.seq_lens.tolist() == [0, 0, 0, 0]
    assert md.query_start_loc.tolist() == [0, 0, 0, 0, 0]
    assert indptr.tolist() == [0] * 6
    # A live request count would make the kernel read the gathered rows.
    assert md.num_actual_tokens == 0


def test_warm_meta_stays_zero_across_calls():
    md1, ind1 = _warm_meta(2, 3, _bt(), 4, torch.device("cpu"))
    md1.seq_lens[0] = 7  # a caller mutating it must not leak into the next warm
    md2, ind2 = _warm_meta(2, 3, _bt(), 4, torch.device("cpu"))
    assert md2.seq_lens.tolist() == [0, 0]
    assert ind2.tolist() == [0, 0, 0]
    assert md2.seq_lens is md1.seq_lens  # cached buffers, reused


def test_warm_meta_carries_the_capacity_indptr_length():
    """The indirect path indexes `mIndptr[req]` for `req < num_reqs`; the buffer is
    sized by the ENGINE capacity, which is what the capture's own indptr is."""
    md, indptr = _warm_meta(1, 1025, _bt(), 4, torch.device("cpu"))
    assert indptr.numel() == 1025
    assert int(md.num_reqs) == 1
