"""What does the first launch of a schedule cost, and what does a second?

CuTeDSL compiles per (constexpr config, tensor shape), and the gathered K/V shape
is the engine's reservation, so nothing outside the engine's own ``forward`` can
precompile a schedule. That is why the engine's warm-up plans could not work
(FAILURE_MODES 15b) and why the compile lands in the first request: this probe
measures the two numbers with no engine and no model weights.

    [warm] <first launch>  -- a compile when the schedule is cold (~1.5 s)
    [next] <second launch> -- the jit path's per-call cost once it is cached

Run: python ci_probe/probe_compile_cost.py
"""

from __future__ import annotations

import time

import torch

from benchmarks.bench_common import make_synthetic_batch
from thunder_vllm.attention.cute_kernel import (
    ThunderAttentionForward,
    launch_thunder_attention,
)
from thunder_vllm.attention.paged_kv import make_paged_kv_manager
from thunder_vllm.attention.tile_shape import tile_shape


def main() -> int:
    torch.manual_seed(0)
    batch, seqlen_q, seqlen_k = 1, 4096, 4096
    sb = make_synthetic_batch(batch, seqlen_k, 32, 8, 128, device="cuda")
    mgr = make_paged_kv_manager(
        sb.layout, max_num_reqs=batch,
        max_model_len=sb.block_table.shape[1] * sb.layout.block_size, device="cuda",
    )
    gathered = mgr.gather_packed_tiles(sb.block_table, sb.kv_cache, sb.kv_scales)
    q = sb.q[: batch * seqlen_q]
    out = torch.empty_like(sb.q)

    tile = tile_shape(True, batch, seqlen_q)
    kernel = ThunderAttentionForward(
        head_dim=128, K_BITS=sb.layout.k_bits, V_BITS=sb.layout.v_bits,
        qhead_per_kvhead=4, gqa_rows_per_head=tile["m_block"] // 4, is_causal=True,
        m_block_size=tile["m_block"], n_block_size=tile["n_block"],
        num_threads=tile["num_threads"],
    )
    meta = type("M", (), {
        "seq_lens": torch.full((batch,), seqlen_k, device="cuda", dtype=torch.int32),
        "query_start_loc": torch.tensor([0, batch * seqlen_q], device="cuda",
                                        dtype=torch.int32),
        "max_query_len": seqlen_q,
    })()
    kw = dict(num_splits=1, gqa_pack=True, onepass=True, reg_rescale=True,
              causal_bound=True)

    # 1. The first launch of this schedule: this is the compile a request pays.
    t0 = time.perf_counter()
    launch_thunder_attention(kernel, q, gathered, out, meta, 128 ** -0.5,
                             quantizer=sb.quantizer, **kw)
    warm = (time.perf_counter() - t0) * 1e3

    # 2. The next launch of the same schedule: compiled already or not?
    t0 = time.perf_counter()
    launch_thunder_attention(kernel, q, gathered, out, meta, 128 ** -0.5,
                             quantizer=sb.quantizer, **kw)
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    for _ in range(10):
        launch_thunder_attention(kernel, q, gathered, out, meta, 128 ** -0.5,
                                 quantizer=sb.quantizer, **kw)
    torch.cuda.synchronize()
    steady = (time.perf_counter() - t0) * 1e3 / 10
    print(f"[warm] {warm:.1f} ms  [next] {first:.2f} ms  steady {steady:.3f} ms  "
          f"cold_compile={'yes' if warm > 200 else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
