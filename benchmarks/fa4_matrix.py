"""Frozen FA4-vs-ours benchmark matrix (apples-to-apples gate).

Compares FA4 BF16 (``FlashAttentionForwardSm100``) against the TurboQuant
kernel across head_dim x GQA x causal x {prefill, decode} x seqlen x splits,
with IDENTICAL timing methodology on both sides: ``do_bench_stats`` (25
warmup + 100 CUDA-event-timed reps). FA4 reads a dense fp16 cache, ours reads
the packed cache; both consume the same logical inputs.

Reported per cell: ms (med/p20/p80), us, TFLOP/s-equivalent (standard
attention FLOPs), GB/s per side (each side's own bytes), ours/FA4 ratio,
analytic CTA count (ours), and optional ncu counters (SM/tensor activity,
DRAM) for selected cells.

Frozen means: the cell list, metric set, and methodology in this file ARE the
acceptance gate (ours >= FA4 per workload). Change them only with a commit
that says why.

Run:
    python -m benchmarks.fa4_matrix --out benchmarks/results/fa4_matrix.md
    python -m benchmarks.fa4_matrix --quick          # small subset
    python -m benchmarks.fa4_matrix --only decode   # substring filter
    python -m benchmarks.fa4_matrix --only-cell 3   # single cell (ncu use)
    python -m benchmarks.fa4_matrix --ncu-cells 3,10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_common import (  # noqa: E402
    bandwidth_fwd_bytes,
    do_bench_stats,
    effective_kv_bytes,
    flops,
    make_synthetic_batch,
)
from thunder_vllm.utils.telemetry import system_line  # noqa: E402

HK = 8
HEAD_DIMS = (64, 128)
GQAS = (1, 4)
PREFILL_SEQLENS = (1024, 4096, 16384)
DECODE_BATCHES = (1, 16)
DECODE_SEQLENS = (4096, 16384)
DECODE_SPLITS = (1, 4)
MB = int(os.environ.get("THUNDER_M_BLOCK", 64))
NB = int(os.environ.get("THUNDER_N_BLOCK", 64))


def _env_on(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def _ours_flags() -> dict:
    return {
        "onepass": _env_on("THUNDER_ONEPASS", True),
        "reg_rescale": _env_on("THUNDER_REG_RESCALE", True),
        "causal_bound": _env_on("THUNDER_CAUSAL_BOUND", True),
        "gqa_pack": _env_on("THUNDER_GQA_PACK", False),
        "indirect": _env_on("THUNDER_8B_INDIRECT", False),
        "m_block": MB,
        "n_block": NB,
    }


def cells() -> list[dict]:
    out = []
    for hd in HEAD_DIMS:
        for gqa in GQAS:
            for causal in (True, False):
                for s in PREFILL_SEQLENS:
                    out.append({"head_dim": hd, "gqa": gqa, "causal": causal,
                                "mode": "prefill", "batch": 1, "seqlen_q": s,
                                "seqlen_k": s, "splits": 1})
                for b in DECODE_BATCHES:
                    for s in DECODE_SEQLENS:
                        for sp in DECODE_SPLITS:
                            out.append({"head_dim": hd, "gqa": gqa,
                                        "causal": causal, "mode": "decode",
                                        "batch": b, "seqlen_q": 1,
                                        "seqlen_k": s, "splits": sp})
    for i, c in enumerate(out):
        tag = ("hd%d-gqa%d-%s-%s-B%d-S%d-sp%d"
               % (c["head_dim"], c["gqa"],
                  "causal" if c["causal"] else "ncausal",
                  c["mode"], c["batch"], c["seqlen_k"], c["splits"]))
        c["name"] = tag
        c["idx"] = i
    return out


def _fa4_callable(q, k, v, head_dim, gqa, causal):
    """FA4 baseline callable.

    flash-attn-4 >= 4.0.0b31 moved ``FlashAttentionForwardSm100.__call__`` to a
    raw cute-tensor signature (``mQ, mK, mV, mO, mLSE, softmax_scale, ...``), so
    the old ``fwd(q, k, v)`` no longer type-checks (``Required argument 'mO' is
    missing``). Use the supported torch-level entrypoint ``flash_attn_func``,
    which allocates ``mO``/``mLSE`` and handles GQA natively. Inputs are the
    standard ``(batch, seqlen, nheads, head_dim)`` torch layout; FA4's
    ``qhead_per_kvhead`` is derived from the q/k head counts, so K/V carry the
    real ``Hk`` heads (matching ``bandwidth_fwd_bytes(..., nheads_kv=HK)``).
    """
    from flash_attn.cute import flash_attn_func

    def run():
        return flash_attn_func(q, k, v, causal=causal)

    return run


def _fa4_vendored_callable(q, k, v, head_dim, gqa, causal):
    """Same as ``_fa4_callable`` but importing ``flash_attn_func`` from the
    vendored FA4 tree at ``thunder_vllm.attention.v2.fa4`` (frozen at
    flash-attn-4 4.0.0b31). Used to prove the v2 skeleton: the vendored,
    unmodified forward must reproduce the installed package's numbers before
    any TurboQuant producer work starts on it (validation ladder S2 in
    ``docs/PIPELINE_V2.md``).
    """
    from thunder_vllm.attention.v2.fa4.interface import flash_attn_func

    def run():
        return flash_attn_func(q, k, v, causal=causal)

    return run


def _ours_callable(sb, q, meta_q, num_splits, flags, head_dim, gqa, causal):
    from thunder_vllm.attention.cute_kernel import (
        ThunderAttentionForward,
        launch_thunder_attention,
    )
    from thunder_vllm.attention.paged_kv import make_paged_kv_manager

    kernel = ThunderAttentionForward(
        head_dim=head_dim, K_BITS=sb.layout.k_bits, V_BITS=sb.layout.v_bits,
        qhead_per_kvhead=gqa, is_causal=causal,
        m_block_size=MB, n_block_size=NB, num_threads=128,
    )
    mgr = make_paged_kv_manager(
        sb.layout, max_num_reqs=sb.block_table.shape[0],
        max_model_len=sb.block_table.shape[1] * sb.layout.block_size,
        device=sb.q.device,
    )
    gathered = mgr.gather_packed_tiles(
        sb.block_table, sb.kv_cache, sb.kv_scales, sb.seq_lens)
    out = torch.empty_like(q)

    def run():
        launch_thunder_attention(
            kernel, q, gathered, out, meta_q, head_dim ** -0.5,
            quantizer=sb.quantizer, num_splits=num_splits,
            gqa_pack=flags["gqa_pack"] and meta_q.max_query_len == 1,
            onepass=flags["onepass"], reg_rescale=flags["reg_rescale"],
            causal_bound=flags["causal_bound"])
        return out

    return run


def _meta(batch: int, blocks: int, seqlen_k: int, mode: str):
    from types import SimpleNamespace

    if mode == "prefill":
        qsl = torch.tensor([0, seqlen_k], device="cuda", dtype=torch.int32)
        slot = torch.arange(seqlen_k, device="cuda")
        mql, is_prefill = seqlen_k, True
    else:
        qsl = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        slot = torch.arange(batch, device="cuda")
        mql, is_prefill = 1, False
    return SimpleNamespace(
        num_reqs=batch, num_actual_tokens=int(qsl[-1].item()),
        max_query_len=mql, is_prefill=is_prefill,
        max_blocks_per_req=blocks,
        block_table=None, seq_lens=None,  # filled by caller
        query_start_loc=qsl, slot_mapping=slot,
    )


def bench_cell(c: dict) -> dict:
    device = "cuda"
    hq = HK * c["gqa"]
    hd = c["head_dim"]
    b, sq, sk = c["batch"], c["seqlen_q"], c["seqlen_k"]
    row: dict = dict(c)
    row["ours_flags"] = _ours_flags()

    sb = make_synthetic_batch(b, sk, hq, HK, hd, device=device)
    if c["mode"] == "prefill":
        q_ours = sb.q
    else:
        q_ours = torch.randn(b, hq, hd, device=device, dtype=torch.float16)
    meta = _meta(b, sb.block_table.shape[1], sk, c["mode"])
    meta.block_table = sb.block_table
    meta.seq_lens = sb.seq_lens

    # FA4 side: dense fp16 in the standard (batch, seqlen, nheads, head_dim)
    # torch layout. K/V keep the real Hk KV heads: FA4 handles GQA natively via
    # pack_gqa, matching the Hk used in bandwidth_fwd_bytes(..., nheads_kv=HK).
    k_dense = sb.key.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
    v_dense = sb.value.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
    if c["mode"] == "prefill":
        q_fa4 = sb.q.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
    else:
        q_fa4 = torch.randn(b, 1, hq, hd, device=device, dtype=torch.float16)

    side = os.environ.get("TQ_MATRIX_SIDE", "both")
    if side in ("both", "fa4"):
        fa4_fn = _fa4_vendored_callable if os.environ.get(
            "TQ_FA4_MODULE", "") == "vendored" else _fa4_callable
        try:
            row["fa4"] = do_bench_stats(
                fa4_fn(q_fa4, k_dense, v_dense, hd, c["gqa"], c["causal"]))
        except torch.cuda.OutOfMemoryError:
            row["fa4_oom"] = True
        except Exception as exc:  # noqa: BLE001
            row["fa4_error"] = repr(exc)
    if side in ("both", "ours"):
        try:
            row["ours"] = do_bench_stats(
                _ours_callable(sb, q_ours, meta, c["splits"], row["ours_flags"],
                               hd, c["gqa"], c["causal"]))
        except torch.cuda.OutOfMemoryError:
            row["ours_oom"] = True
        except Exception as exc:  # noqa: BLE001
            row["ours_error"] = repr(exc)

    fl = flops(b, hq, sq, sk, hd, hd, c["causal"])
    row["flops"] = fl
    row["fp16_bytes"] = bandwidth_fwd_bytes(b, hq, sq, sk, hd, hd, nheads_kv=HK)
    kv = effective_kv_bytes(sb.layout, sk, b)
    qo = b * hq * sq * hd * 2 * 2  # Q read + O write, fp16
    row["ours_bytes"] = kv + qo
    # Analytic CTA count for ours: q-blocks x head axis x reqs x splits.
    qb = math.ceil((sq if c["mode"] == "prefill" else 1) / MB)
    heads = HK if row["ours_flags"]["gqa_pack"] else hq
    row["cta_ours"] = qb * heads * b * c["splits"]
    return row


NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]


def _ncu_cell(idx: int, out_dir: str) -> dict:
    """Profile one cell per side under ncu (process-wide counters)."""
    res: dict = {"idx": idx}
    for side in ("fa4", "ours"):
        prefix = os.path.join(out_dir, f"ncu_cell{idx}_{side}")
        env = dict(os.environ, TQ_MATRIX_SIDE=side)
        cmd = ["ncu", "--csv", "--metrics", ",".join(NCU_METRICS),
               "--target-processes", "all", "-o", prefix,
               sys.executable, "-m", "benchmarks.fa4_matrix",
               "--only-cell", str(idx)]
        print("ncu running", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=False, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            res[side] = {"ncu_error": repr(exc)}
            continue
        csv_path = prefix + ".csv"
        try:
            vals: dict[str, list[float]] = {}
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    m = r.get("Metric Name", "")
                    try:
                        vals.setdefault(m, []).append(float(r.get("Metric Value", "nan")))
                    except ValueError:
                        pass
            agg = {}
            for m, xs in vals.items():
                xs = [x for x in xs if x == x]
                if not xs:
                    continue
                agg[m] = sum(xs) / len(xs) if "pct" in m or "avg" in m else sum(xs)
            res[side] = agg
        except Exception as exc:  # noqa: BLE001
            res[side] = {"ncu_error": repr(exc)}
    return res


def _summarize(r: dict) -> dict:
    s: dict = {"name": r["name"]}
    for side in ("fa4", "ours"):
        st = r.get(side)
        if not isinstance(st, dict):
            s[side + "_ms"] = None
            continue
        s[side + "_ms"] = st["median_ms"]
    fl = r["flops"]
    out = dict(s)
    for side in ("fa4", "ours"):
        ms = s[side + "_ms"]
        out[side + "_tflops"] = fl / (ms * 1e-3) / 1e12 if ms else None
        b = r["fp16_bytes" if side == "fa4" else "ours_bytes"]
        out[side + "_gbps"] = b / (ms * 1e-3) / 1e9 if ms else None
    if s["fa4_ms"] and s["ours_ms"]:
        out["ratio"] = s["ours_ms"] / s["fa4_ms"]
    else:
        out["ratio"] = None
    out["cta_ours"] = r.get("cta_ours")
    out["ncu"] = r.get("ncu")
    for k in ("fa4_oom", "ours_oom", "fa4_error", "ours_error"):
        if r.get(k):
            out[k] = r[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results/fa4_matrix.md")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--only-cell", default="")
    ap.add_argument("--ncu-cells", default="")
    args = ap.parse_args()

    print(system_line(), flush=True)
    all_cells = cells()
    if args.only_cell != "":
        sel = [all_cells[int(args.only_cell)]]
    elif args.quick:
        sel = [c for c in all_cells
               if c["head_dim"] == 128 and c["gqa"] == 4 and c["causal"]
               and ((c["mode"] == "prefill" and c["seqlen_k"] == 4096)
                    or (c["mode"] == "decode" and c["batch"] == 1
                        and c["seqlen_k"] == 4096))]
    elif args.only:
        sel = [c for c in all_cells if args.only in c["name"]]
    else:
        sel = all_cells
    print(f"[fa4_matrix] {len(sel)}/{len(all_cells)} cells", flush=True)

    ncu_idx = {int(x) for x in args.ncu_cells.split(",") if x.strip()}
    rows = []
    for c in sel:
        print(f"[fa4_matrix] {c['name']}", flush=True)
        try:
            r = bench_cell(c)
        except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
            print(f"  OOM: {exc}")
            r = dict(c, oom=True)
        if c["idx"] in ncu_idx:
            r["ncu"] = _ncu_cell(c["idx"],
                                 os.path.dirname(args.out) or ".")
        rows.append(r)

    summ = [_summarize(r) for r in rows]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# FA4 vs TurboQuant kernel matrix\n\n")
        f.write("Identical methodology both sides: `do_bench_stats`, 25 warmup + "
                "100 CUDA-event-timed reps. ncu counters (when present) are "
                "process-wide per side.\n\n")
        f.write("| cell | FA4 ms | ours ms | ours/FA4 | TFLOPs fa4/ours | "
                "GB/s fa4/ours | CTA ours | SM% fa4/ours |\n")
        f.write("|---" * 8 + "|\n")
        for r, s in zip(rows, summ):
            def fmt(x, nd=3):
                return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"
            sm = ""
            ncu = s.get("ncu") or {}
            for side in ("fa4", "ours"):
                m = (ncu.get(side) or {}).get(
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed")
                sm += ("%.0f" % m) if isinstance(m, (int, float)) else "n/a"
                sm += "/"
            f.write(f"| {s['name']} | {fmt(s['fa4_ms'], 4)} | {fmt(s['ours_ms'], 4)} | "
                    f"{fmt(s['ratio'])} | {fmt(s['fa4_tflops'], 1)}/{fmt(s['ours_tflops'], 1)} | "
                    f"{fmt(s['fa4_gbps'], 0)}/{fmt(s['ours_gbps'], 0)} | "
                    f"{r.get('cta_ours', 'n/a')} | {sm.rstrip('/')} |\n")
            for k in ("fa4_oom", "ours_oom", "fa4_error", "ours_error", "oom"):
                if r.get(k):
                    f.write(f"| | | | | | | | {k}={r[k]} |\n")
    with open(args.out + ".json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"wrote {args.out} (+ .json)")


if __name__ == "__main__":
    main()
