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


def test_capacity_warm_reaches_the_launcher(monkeypatch):
    """The warm must actually launch, per capture size and at the capacity.

    `launch_thunder_attention` is imported *inside* `forward`, so it is not a
    module global; `_capacity_warm` needs its own import. It lacked one, every
    launch raised NameError into `forward`'s non-fatal handler, and the captures
    stayed cold while the logs looked like the warm was running. This pins the
    lookup, and the two geometries the engine captures (PIECEWISE at each
    `cudagraph_capture_sizes` entry, FULL padded to the capacity).
    """
    import sys
    import types

    from thunder_vllm.attention import backend as B

    launched = []
    fake = types.ModuleType("thunder_vllm.attention.cute_kernel")
    fake.launch_thunder_attention = lambda *a, **k: launched.append((a, k))
    monkeypatch.setitem(sys.modules, "thunder_vllm.attention.cute_kernel", fake)

    class _Cfg:
        onepass = False
        reg_rescale = False
        causal_bound = False

    class _Kernel:
        tile_m, tile_n = 32, 32

    class _Md:
        capture_sizes = (1, 2)
        max_num_reqs_capacity = 1024
        max_blocks_per_req = 264
        block_table = torch.zeros((1, 264), dtype=torch.int32)

    impl = object.__new__(B.ThunderAttentionImpl)  # __init__ needs a CUDA config
    impl.head_size, impl.num_heads, impl.num_kv_heads = 128, 32, 8
    impl.num_kv_groups, impl.scale, impl.cfg = 4, 1.0, _Cfg()
    impl.get_kernel = lambda *a, **k: _Kernel()

    B._WARM_CAPACITY_DONE.clear()
    q = torch.zeros((4, 32, 128), dtype=torch.float16)
    impl._capacity_warm(q, None, None, _Md(), None, None, False)

    assert [k["num_splits"] for _, k in launched]  # one per split count, per size
    assert len(launched) == 3 * 7  # sizes 1, 2 and the capacity, x 7 split counts
    # The capture's own q/o shape, not the step's storage-derived view.
    assert launched[0][0][1].shape == (1, 32, 128)
    assert launched[0][0][3].shape == (1, 32, 128)
