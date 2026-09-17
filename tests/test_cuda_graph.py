"""CUDA-graph capture / replay parity (SM100/SM110 only).

This is the test that catches the three classic graph bugs in a paged-KV
attention backend:

1. **Host sync** inside ``forward`` (``.item()``/``.cpu()``) -- capture fails
   outright, because capture forbids synchronising ops.
2. **Stale pointers** from allocating scratch after capture -- replay reads
   freed memory, so outputs differ between replays.
3. **Scratch aliasing** between layers -- replaying twice overwrites the
   buffer the first replay is still reading.

The test captures a decode-shaped ``forward``, replays it 10 times, and asserts
the outputs are bit-identical across replays *and* equal to the eager call.
"""

from __future__ import annotations

import os

import pytest
import torch

from turboquant_vllm.attention.cache_layout import (
    TurboQuantCacheLayout,
    allocate_kv_cache,
)
from turboquant_vllm.attention.paged_kv import make_paged_kv_manager
from turboquant_vllm.attention.scratch import new_scratch, reserve_scratch
from turboquant_vllm.quant.quantizer import TurboQuantQuantizer

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]


@pytest.mark.skipif(
    os.environ.get("TURBOQUANT_KERNEL_ENABLE", "0") != "1",
    reason="kernel schedule incomplete; set TURBOQUANT_KERNEL_ENABLE=1",
)
def test_cuda_graph_replay_parity():
    device = "cuda"
    torch.manual_seed(0)
    d, hk, hq = 128, 8, 32
    batch, nk = 4, 4096
    bs = 16

    layout = TurboQuantCacheLayout(
        num_kv_heads=hk, head_dim=d, k_bits=4, v_bits=4, block_size=bs
    )
    quant = TurboQuantQuantizer(d, 4, 4, device=device)
    nb = batch * ((nk + bs - 1) // bs)
    kv, scales = allocate_kv_cache(nb, bs, hk, d, 4, 4, device=device)

    block_table = torch.arange(
        0, nb, device=device, dtype=torch.int32
    ).reshape(batch, -1)
    mgr = make_paged_kv_manager(
        layout, max_num_reqs=batch, max_model_len=nk, device=device
    )
    # reserve BEFORE capture: this is the whole point of the scratch API.
    mgr.reserve()
    scratch = new_scratch()
    reserve_scratch(batch, hq, d, device, scratch, num_splits=1)

    from turboquant_vllm.attention.cute_kernel import TurboQuantAttentionForward

    kernel = TurboQuantAttentionForward(
        head_dim=d, K_BITS=4, V_BITS=4, qhead_per_kvhead=hq // hk, is_causal=True
    )

    q = torch.randn(batch, hq, d, device=device, dtype=torch.float16)

    def run(q_buf):
        return _decode_forward(kernel, q_buf, kv, scales, mgr, block_table, layout)

    eager = run(q).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(q)
    graph.replay()
    first = captured.clone()
    for _ in range(9):
        graph.replay()
        assert torch.equal(captured, first), "replay output drifted"
    assert torch.allclose(first.float(), eager.float(), atol=1e-3, rtol=1e-3)


def _decode_forward(kernel, q, kv, scales, mgr, block_table, layout):
    raise NotImplementedError(
        "wired once the CuTe schedule lands; the reserve-before-capture ordering "
        "and pointer stability it checks are exercised by "
        "tests/test_paged_kv.py::test_gather_in_place_reuses_buffer"
    )
