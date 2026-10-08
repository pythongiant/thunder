"""Does the warm-up precompile actually happen?

``launch_thunder_attention(compile_only=True)`` exists so the CuTeDSL compile
(~1.5 s) happens during warm-up instead of inside the first request. It passed
the wrong argument list (no ``debug``, no schedule constexprs), so the trailing
CUstream landed on ``num_splits`` and every call raised
``ARG_ANNOTATION_MISMATCH`` -- caught upstream and logged as non-fatal, so the
only symptom was the compile still landing in the first request.

This measures it directly, with no engine and no model weights: precompile a
schedule, then time the FIRST real launch of that schedule. ~0.15 ms means the
precompile landed; ~1.5 s means it did not (and the raise was swallowed).

Run: python ci_probe/probe_compile_only.py
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

    # 1. The precompile itself: does it raise?
    t0 = time.perf_counter()
    try:
        launch_thunder_attention(kernel, q, gathered, out, meta, 128 ** -0.5,
                                 quantizer=sb.quantizer, compile_only=True, **kw)
        print(f"[compile_only] ok in {(time.perf_counter() - t0) * 1e3:.1f} ms",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[compile_only] RAISED {type(exc).__name__}: {str(exc)[:200]}",
              flush=True)

    # 2. The first real launch of that same schedule: compiled already or not?
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
    print(f"[launch] first={first:.2f} ms steady={steady:.3f} ms "
          f"verdict={'PRECOMPILED' if first < 50 else 'COMPILED IN REQUEST'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
