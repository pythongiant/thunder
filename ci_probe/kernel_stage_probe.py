"""Stage decomposition for the attention kernel (no kernel changes).

Uses existing knobs to attribute time:
  * onepass      : removes PASS-1 (a second K load + dequant per tile)
  * reg_rescale  : removes an SMEM round-trip + 2 barriers per tile
  * causal_bound : removes fully-masked tiles (causal prefill)

If `onepass` cuts a lot, the K load+dequant path dominates; if `reg_rescale`
does, barrier/SMEM traffic dominates; if neither, MMA/softmax dominates.

    THUNDER_N_BLOCK=32 python ci_probe/kernel_stage_probe.py
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
# Query-head count for the probe. Default 8 preserves the old microbench;
# set THUNDER_BENCH_HQ=32 for the engine-like GQA shape (32 Q heads, 8 KV heads).
HQ = int(os.environ.get("THUNDER_BENCH_HQ", 8))
MB = int(os.environ.get("THUNDER_M_BLOCK", 64))
NB = int(os.environ.get("THUNDER_N_BLOCK", 64))
ITERS = int(os.environ.get("BENCH_ITERS", 20))
dev = "cuda"


def make_case(nt: int, causal: bool, k_bits=4, v_bits=4):
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
    q = torch.randn(nq, HQ, HD, device=dev, dtype=torch.float16)
    qr = (q.float() @ quant.rotation.matrix.float()).to(torch.float16)
    out = torch.zeros_like(qr)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([nt], device=dev, dtype=torch.int32),
        query_start_loc=torch.tensor([0, nq], device=dev, dtype=torch.int32),
        max_blocks_per_req=nblk, max_query_len=nq)
    per_tok = HK * (layout.k_packed_bytes + layout.v_packed_bytes + 4)
    return quant, g, qr, out, meta, float(per_tok * nt)


def timeit(quant, g, qr, out, meta, causal, onepass, reg_rescale, causal_bound,
           gqa_pack=False):
    fwd = ThunderAttentionForward(head_dim=HD, K_BITS=4, V_BITS=4,
                                  qhead_per_kvhead=HQ // HK, is_causal=causal,
                                  m_block_size=MB, n_block_size=NB,
                                  num_threads=128)

    def call():
        launch_thunder_attention(fwd, qr, g, out, meta, HD ** -0.5,
                                 quantizer=quant, gqa_pack=gqa_pack,
                                 onepass=onepass,
                                 reg_rescale=reg_rescale,
                                 causal_bound=causal_bound)

    for _ in range(4):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def sweep(tag: str, nt: int, causal: bool) -> None:
    quant, g, qr, out, meta, nbytes = make_case(nt, causal)
    print(f"\n[{tag}] n={nt} causal={causal} bytes={nbytes/1e6:.2f}MB", flush=True)
    base = timeit(quant, g, qr, out, meta, causal, False, False, False)
    configs = [
        ("onepass", True, False, False, False),
        ("reg_rescale (onepass)", True, True, False, False),
        ("causal_bound", False, False, True, False),
    ]
    if not causal:
        # GQA-packed decode (plan steps 6+8). Needs HQ > HK to take effect;
        # GPU validation pending.
        configs.append(("gqa_pack", True, True, False, True))
    print(f"  {'base':26s} {base:8.3f}ms  "
          f"bw={nbytes / (base/1e3) / 1e9:6.2f} GB/s", flush=True)
    for name, op, rr, cb, gqa in configs:
        t = timeit(quant, g, qr, out, meta, causal, op, rr, cb, gqa)
        print(f"  {name:26s} {t:8.3f}ms  x{base/t:4.2f}  "
              f"bw={nbytes / (t/1e3) / 1e9:6.2f} GB/s", flush=True)


def main() -> int:
    print(f"device: {torch.cuda.get_device_name(0)} m_block={MB} n_block={NB}",
          flush=True)
    from thunder_vllm.utils.telemetry import system_line  # noqa: E402
    print(system_line(), flush=True)
    sweep("decode", 2048, causal=False)
    sweep("prefill", 2048, causal=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
