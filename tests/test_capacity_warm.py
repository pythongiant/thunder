"""The capacity warm-up's padded metadata (CPU).

`_capacity_warm_meta` builds the metadata for the eager launch that pre-compiles
the CUDA-graph capture's geometry (``docs/FAILURE_MODES.md`` 14). Its padding MUST
stay zero: the launch runs the real kernel with ``num_reqs = capacity``, and a
stale ``seq_lens``/``query_start_loc`` entry in the tail would make the kernel
read past the gathered buffers -- the same class of failure as FAILURE_MODES 15.
Zeroed entries mean those requests have ``kv_len == 0`` and ``q_len == 0``, so
their CTAs exit without touching anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from thunder_vllm.attention.backend import _capacity_warm_meta


def _meta(num_reqs: int, cap: int) -> SimpleNamespace:
    return SimpleNamespace(
        seq_lens=torch.arange(1, num_reqs + 1, dtype=torch.int32),
        query_start_loc=torch.arange(0, num_reqs + 1, dtype=torch.int32),
        block_table=torch.zeros((num_reqs, 4), dtype=torch.int32),
        max_num_reqs_capacity=cap,
        max_blocks_per_req=4,
        max_query_len=1,
    )


def test_capacity_meta_pads_the_tail_with_zeros():
    md = _capacity_warm_meta(_meta(3, 8), torch.device("cpu"))
    assert md.num_reqs == 8
    assert md.seq_lens.tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert md.query_start_loc.tolist() == [0, 1, 2, 3, 0, 0, 0, 0, 0]


def test_capacity_meta_refreshes_the_live_prefix():
    obs = _meta(3, 8)
    md = _capacity_warm_meta(obs, torch.device("cpu"))
    obs.seq_lens[:] = torch.tensor([5, 6, 7], dtype=torch.int32)
    md2 = _capacity_warm_meta(obs, torch.device("cpu"))
    assert md2.seq_lens is md.seq_lens  # cached buffers, refreshed in place
    assert md2.seq_lens.tolist() == [5, 6, 7, 0, 0, 0, 0, 0]
    assert md2.query_start_loc.tolist() == [0, 1, 2, 3, 0, 0, 0, 0, 0]


def test_no_capacity_warm_without_headroom():
    # Nothing to warm when the step already has the capacity's geometry.
    assert _capacity_warm_meta(_meta(8, 8), torch.device("cpu")) is None
    assert _capacity_warm_meta(_meta(8, 0), torch.device("cpu")) is None


def test_capacity_meta_clears_the_previous_live_tail():
    """The buffer is reused across steps, so a smaller step must ZERO what the
    previous step left live: the launch runs with `num_reqs = capacity`, and a
    stale entry in that tail is an out-of-bounds read of the gathered buffers.
    """
    big = _capacity_warm_meta(_meta(8, 16), torch.device("cpu"))
    assert big.seq_lens.tolist()[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    small = _capacity_warm_meta(_meta(3, 16), torch.device("cpu"))
    assert small.seq_lens.tolist() == [1, 2, 3] + [0] * 13
    assert small.query_start_loc.tolist() == [0, 1, 2, 3] + [0] * 13
