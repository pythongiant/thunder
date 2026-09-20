"""Time launch_thunder_attention (P0-a A/B: scalar vs vectorized loads).

    THUNDER_N_BLOCK=32 python ci_probe/kernel_bench.py

Reports median ms and effective KV bandwidth for prefill and decode shapes.
Absolute numbers are GPU-specific; use it for before/after on one machine.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thunder_vllm.attention.cache_layout import (  # noqa: E402
    ThunderCacheLayout,
    allocate_kv_cache,
    reshape_and_cache,
)
from thunder_vllm.attention.cute_kernel import (  # noqa: E402
    ThunderAttentionForward,
    launch_thunder_attention,
)
from thunder_vllm.attention.paged_kv import make_paged_kv_manager  # noqa: E402
from thunder_vllm.quant.quantizer import ThunderQuantizer  # noqa: E402

HD, HK, BS = 128, 8, 16
MB = int(os.environ.get("THUNDER_M_BLOCK", 64))
NB = int(os.environ.get("THUNDER_N_BLOCK", 64))
ITERS = int(os.environ.get("BENCH_ITERS", 30))
dev = "cuda"


def bench(tag: str, k_bits: int, v_bits: int, nt: int, causal: bool) -> None:
    layout = ThunderCacheLayout(num_kv_heads=HK, head_dim=HD, k_bits=k_bits,
                                v_bits=v_bits, block_size=BS)
    quant = ThunderQuantizer(HD, k_bits, v_bits, device=dev)
    nblk = math.ceil(nt / BS)
    key = torch.randn(nt, HK, HD, device=dev, dtype=torch.float16)
    val = torch.randn(nt, HK, HD, device=dev, dtype=torch.float16)
    kv, sc = allocate_kv_cache(nblk, BS, HK, HD, k_bits, v_bits, device=dev)
    reshape_and_cache(key, val, torch.arange(nt, device=dev, dtype=torch.long),
                      kv, sc, quant, layout)
    mgr = make_paged_kv_manager(layout, max_num_reqs=1, max_model_len=nblk * BS,
                                device=dev)
    bt = torch.arange(nblk, device=dev, dtype=torch.int32).reshape(1, nblk)
    g = mgr.gather_packed_tiles(bt, kv, sc,
                                torch.tensor([nt], device=dev, dtype=torch.int32))
    nq = nt if causal else 1
    q = torch.randn(nq, HK, HD, device=dev, dtype=torch.float16)
    qr = (q.float() @ quant.rotation.matrix.float()).to(torch.float16)
    out = torch.zeros_like(qr)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([nt], device=dev, dtype=torch.int32),
        query_start_loc=torch.tensor([0, nq], device=dev, dtype=torch.int32),
        max_blocks_per_req=nblk, max_query_len=nq)
    fwd = ThunderAttentionForward(head_dim=HD, K_BITS=k_bits, V_BITS=v_bits,
                                  qhead_per_kvhead=1, is_causal=causal,
                                  m_block_size=MB, n_block_size=NB, num_threads=128)

    def call():
        launch_thunder_attention(fwd, qr, g, out, meta, HD ** -0.5, quantizer=quant)

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ms = statistics.median(ts)
    per_tok = HK * (layout.k_packed_bytes + layout.v_packed_bytes + 4)
    nbytes = float(per_tok * nt)  # all KV heads, one request
    print(f"  {tag:22s} n={nt:6d} med={ms:8.3f}ms  "
          f"kv_bw={nbytes / (ms / 1e3) / 1e9:7.2f} GB/s", flush=True)


def main() -> int:
    print(f"device: {torch.cuda.get_device_name(0)} m_block={MB} n_block={NB}",
          flush=True)
    for nt in (512, 2048, 8192):
        bench("prefill 4b/4b", 4, 4, nt, causal=True)
        bench("decode  4b/4b", 4, 4, nt, causal=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
