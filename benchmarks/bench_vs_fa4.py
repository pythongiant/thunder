"""Kernel-level benchmark: Thunder-CuTe vs FA4 (``FlashAttentionForwardSm100``).

FA4 exposes no public ``AttentionBackend`` in vLLM, so this benchmark is at the
kernel level. A *dense fp16* KV cache is built alongside the packed TurboQuant
cache so both kernels consume the same logical inputs (FA4 reads fp16, ours
reads packed), and both are timed with triton's ``do_bench``.

Reports latency, TFLOPs/s, effective bandwidth, FA4 bandwidth, the ratio
columns, and the capacity story (``memory_tokens_per_gb``).

Run:
    python -m benchmarks.bench_vs_fa4 --out benchmarks/results/vs_fa4.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_common import (  # noqa: E402
    bandwidth_fwd_bytes,
    do_bench_stats,
    effective_kv_bytes,
    flops,
    fp16_tokens_per_gb,
    make_synthetic_batch,
    memory_tokens_per_gb,
)

SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
BATCHES = [1, 8, 32]
NHEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
CAUSAL = True


def _fa4_callable(q, k, v, causal):
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

    fwd = FlashAttentionForwardSm100(
        head_dim=HEAD_DIM,
        head_dim_v=HEAD_DIM,
        qhead_per_kvhead=NHEADS // NUM_KV_HEADS,
        is_causal=causal,
    )

    def run():
        return fwd(q, k, v)

    return run


def bench_one(batch: int, seqlen: int) -> dict:
    device = "cuda"
    sb = make_synthetic_batch(
        batch, seqlen, NHEADS, NUM_KV_HEADS, HEAD_DIM, device=device
    )
    layout = sb.layout

    # Dense fp16 cache for FA4.
    k_dense = sb.key.repeat_interleave(NHEADS // NUM_KV_HEADS, dim=1).contiguous()
    v_dense = sb.value.repeat_interleave(NHEADS // NUM_KV_HEADS, dim=1).contiguous()
    q_dense = sb.q.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()

    result: dict = {"batch": batch, "seqlen": seqlen, "fa4_oom": False}

    try:
        fa4 = _fa4_callable(q_dense, k_dense, v_dense, CAUSAL)
        result["fa4"] = do_bench_stats(fa4)
    except torch.cuda.OutOfMemoryError:
        result["fa4_oom"] = True
    except Exception as exc:  # noqa: BLE001
        result["fa4_error"] = repr(exc)

    # Our kernel: launcher is wired once the schedule lands.
    try:
        from thunder_vllm.attention.cute_kernel import launch_thunder_attention

        result["ours"] = _time_ours(sb, launch_thunder_attention)
    except Exception as exc:  # noqa: BLE001
        result["ours_error"] = repr(exc)

    fl = flops(batch, NHEADS, seqlen, seqlen, HEAD_DIM, HEAD_DIM, CAUSAL)
    result["flops"] = fl
    result["eff_kv_bytes"] = effective_kv_bytes(layout, seqlen, batch)
    result["fp16_kv_bytes"] = bandwidth_fwd_bytes(
        batch, NHEADS, seqlen, seqlen, HEAD_DIM, HEAD_DIM, nheads_kv=NUM_KV_HEADS
    )
    result["tokens_per_gb_tq"] = memory_tokens_per_gb(layout, 40.0)
    result["tokens_per_gb_fp16"] = fp16_tokens_per_gb(NUM_KV_HEADS, HEAD_DIM, 40.0)
    return result


def _time_ours(sb, launcher):
    from thunder_vllm.attention.cute_kernel import ThunderAttentionForward
    from thunder_vllm.attention.paged_kv import make_paged_kv_manager

    kernel = ThunderAttentionForward(
        head_dim=HEAD_DIM,
        K_BITS=sb.layout.k_bits,
        V_BITS=sb.layout.v_bits,
        qhead_per_kvhead=NHEADS // NUM_KV_HEADS,
        is_causal=CAUSAL,
    )
    mgr = make_paged_kv_manager(
        sb.layout,
        max_num_reqs=sb.block_table.shape[0],
        max_model_len=sb.block_table.shape[1] * sb.layout.block_size,
        device=sb.q.device,
    )
    gathered = mgr.gather_packed_tiles(sb.block_table, sb.kv_cache, sb.kv_scales)
    meta = type(
        "M",
        (),
        {
            "num_reqs": sb.block_table.shape[0],
            "num_actual_tokens": sb.q.shape[0],
            "max_query_len": sb.q.shape[0],
            "is_prefill": sb.q.shape[0] > 1,
            "max_blocks_per_req": sb.block_table.shape[1],
            "block_table": sb.block_table,
            "seq_lens": sb.seq_lens,
            "slot_mapping": torch.arange(sb.q.shape[0], device=sb.q.device),
        },
    )()

    def run():
        out = torch.empty_like(sb.q)
        launcher(kernel, sb.q, gathered, out, meta, HEAD_DIM**-0.5, quantizer=sb.quantizer)
        return out

    return do_bench_stats(run)


def _fmt(stats) -> str:
    if not isinstance(stats, dict):
        return "n/a"
    return f"{stats['median_ms']:.4f} / {stats['p20_ms']:.4f} / {stats['p80_ms']:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results/vs_fa4.md")
    args = ap.parse_args()

    rows = []
    for batch in BATCHES:
        for seqlen in SEQLENS:
            print(f"[vs_fa4] batch={batch} seqlen={seqlen}", flush=True)
            try:
                rows.append(bench_one(batch, seqlen))
            except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
                print(f"  OOM: {exc}")
                rows.append({"batch": batch, "seqlen": seqlen, "oom": True})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Thunder-CuTe vs FA4 (kernel level)\n\n")
        f.write(
            "| batch | seqlen | FA4 latency ms (med/p20/p80) | ours latency | "
            "TFLOPs ours | TFLOPs FA4 | eff BW ours GB/s | BW FA4 GB/s | "
            "lat ours/fa4 | tokens/GB ours | tokens/GB fp16 |\n"
        )
        f.write("|---" * 11 + "|\n")
        for r in rows:
            if r.get("oom"):
                f.write(f"| {r['batch']} | {r['seqlen']} | OOM | OOM | | | | | | | |\n")
                continue
            ours = r.get("ours")
            fa4 = r.get("fa4")
            fl = r.get("flops", float("nan"))
            e = r.get("eff_kv_bytes", float("nan"))
            fk = r.get("fp16_kv_bytes", float("nan"))
            our_t = fl / ours["median_ms"] / 1e9 if ours else float("nan")
            fa_t = fl / fa4["median_ms"] / 1e9 if fa4 else float("nan")
            our_bw = e / (ours["median_ms"] * 1e-3) / 1e9 if ours else float("nan")
            fa_bw = fk / (fa4["median_ms"] * 1e-3) / 1e9 if fa4 else float("nan")
            ratio = (
                f"{ours['median_ms'] / fa4['median_ms']:.3f}" if ours and fa4 else "n/a"
            )
            f.write(
                f"| {r['batch']} | {r['seqlen']} | {_fmt(fa4)} | {_fmt(ours)} | "
                f"{our_t:.1f} | {fa_t:.1f} | {our_bw:.0f} | {fa_bw:.0f} | {ratio} | "
                f"{r['tokens_per_gb_tq']:.0f} | {r['tokens_per_gb_fp16']:.0f} |\n"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
