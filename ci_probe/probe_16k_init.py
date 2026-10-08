"""Localize the 16k illegal access: it happens during ``LLM(...)`` init.

The previous probe (``probe_16k_geom.py``) showed the fault is NOT in the served
16k prefill but in ``vLLM``'s own warmup: ``kernel_warmup`` ->
``_run_flashinfer_autotune_dummy_runs`` -> ``runner._dummy_run`` -> the model's
attention. ``_dummy_run`` is a *synthetic* batch, so its metadata is nothing like
a served step: ``num_tokens = max_num_batched_tokens`` split over
``min(num_tokens, max_num_seqs)`` requests, with ``seq_lens`` set to
``max_query_len`` (the whole token count) rather than a per-request length.

This driver runs several ablations in separate subprocesses (a CUDA fault poisons
the context, so each config needs its own process) and prints, per config, the
traceback origin and the geometry our backend was handed:

    A  baseline, CUDA_LAUNCH_BLOCKING=1   -> names the faulting op
    B  THUNDER_SKIP_KERNEL=1              -> kernel launch vs everything else
    C  THUNDER_SKIP_GATHER=1              -> gather vs everything else
    D  THUNDER_8B_INDIRECT=1              -> request-major vs CSR gather
    E  baseline, no launch blocking       -> is it deterministic?

Run: CTX=16384 python ci_probe/probe_16k_init.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

WORKER = r'''
import json
import os
import sys
import time

import torch

CTX = int(os.environ.get("CTX", "16384"))
MAXLEN = int(os.environ.get("MAXLEN", "0")) or (CTX + 64)

from thunder_vllm.model.registry import configure, register

configure(k_bits=int(os.environ.get("K_BITS", "3")),
          v_bits=int(os.environ.get("V_BITS", "4")))
register()

N = {"fwd": 0, "kv": 0, "gather": 0, "prefill": 0}
_IMPL = {}
LIMIT = int(os.environ.get("DUMP_LIMIT", "6"))


def _sync(where, step):
    """Surface a pending async fault HERE, and say whether it is this op's."""
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"[FAULT] before {where} step={step}: {type(exc).__name__}: {exc}",
              flush=True)
        raise


def _tstats(t):
    c = t.detach().to("cpu")
    return {"shape": tuple(int(x) for x in c.shape), "min": int(c.min()),
            "max": int(c.max()), "zeros": int((c == 0).sum()),
            "head": [int(x) for x in c.reshape(-1)[:8]]}


def _dump(tag, md, query, kv_rows, bt_rows):
    if torch.cuda.is_current_stream_capturing():
        return
    sl = _tstats(md.seq_lens)
    qsl = _tstats(md.query_start_loc)
    bt = md.block_table
    mblk = int(bt.shape[1])
    num_reqs = int(md.seq_lens.shape[0])
    n_reqs_pad = int(getattr(md, "num_reqs", 0) or 0)
    mgr = _IMPL.get("impl")._paged if _IMPL.get("impl") else None
    page_rows = int(mgr.page_rows) if mgr is not None else -1
    bs = int(mgr.layout.block_size) if mgr is not None else 16
    weak = None
    if mgr is not None:
        weak = {
            "mgr_max_num_reqs": int(mgr.max_num_reqs),
            "mgr_max_blocks_per_req": int(mgr.max_blocks_per_req),
            "mgr_page_rows": page_rows,
        }
    total_kv = page_rows * bs
    stride = mblk * bs
    kv_max_row = (num_reqs - 1) * stride + sl["max"]
    mql = int(getattr(md, "max_query_len", 0) or 0)
    rows_per_head = 64 // 4  # full-height prefill tile (max_query_len > 32)
    line = {
        "tag": tag, "step": N[tag],
        "num_reqs": num_reqs, "num_reqs_md": n_reqs_pad,
        "max_query_len": mql, "is_prefill": bool(getattr(md, "is_prefill", False)),
        "num_actual_tokens": int(getattr(md, "num_actual_tokens", 0) or 0),
        "q_shape": tuple(int(x) for x in query.shape),
        "bt": tuple(int(x) for x in bt.shape),
        "kv_stride": stride, "total_kv_rows": total_kv,
        "kv_max_row": kv_max_row, "kv_oob": bool(kv_max_row >= total_kv),
        "num_q_blocks_full": -(-mql // rows_per_head),
        "sl": sl, "qsl": qsl,
        "q_len_zero": int((md.query_start_loc[1:] == md.query_start_loc[:-1]).sum()),
        "sl_gt_stride": int((md.seq_lens > stride).sum()),
        "qsl_gt_q": int((md.query_start_loc > int(query.shape[0])).sum()),
    }
    if weak is not None:
        line.update(weak)
    b = bt.detach().to("cpu")
    line["bt_max"] = int(b.max())
    line["bt_row0"] = [int(x) for x in b[0, :6]]
    print("[GEOM] " + json.dumps(line), flush=True)


def _install():
    from thunder_vllm.attention import backend as be
    from thunder_vllm.attention import paged_kv as pk

    orig_fwd = be.ThunderAttentionImpl.forward
    orig_kv = be.ThunderAttentionImpl.do_kv_cache_update
    orig_gather = pk.PagedKVManager.gather_packed_tiles

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, **kw):
        _IMPL["impl"] = self
        N["fwd"] += 1
        step = N["fwd"]
        _sync("forward", step)
        if attn_metadata is not None:
            mql = int(getattr(attn_metadata, "max_query_len", 0) or 0)
            kind = "prefill" if mql > 1 else "decode"
            if kind == "prefill" and N["prefill"] < LIMIT:
                N["prefill"] += 1
                _dump("fwd", attn_metadata, query, None, None)
        out = orig_fwd(self, layer, query, key, value, kv_cache, attn_metadata, **kw)
        _sync("forward", step)
        return out

    def do_kv(self, layer, key, value, kv_cache, attn_metadata, *a, **kw):
        N["kv"] += 1
        _sync("kv_update", N["kv"])
        out = orig_kv(self, layer, key, value, kv_cache, attn_metadata, *a, **kw)
        _sync("kv_update", N["kv"])
        return out

    def gather(self, block_table, kv_cache, kv_scales, seq_lens=None, live_blocks=None):
        _sync("gather", N["gather"])
        N["gather"] += 1
        out = orig_gather(self, block_table, kv_cache, kv_scales, seq_lens,
                          live_blocks=live_blocks)
        _sync("gather", N["gather"])
        return out

    be.ThunderAttentionImpl.forward = forward
    be.ThunderAttentionImpl.do_kv_cache_update = do_kv
    pk.PagedKVManager.gather_packed_tiles = gather


_install()

from vllm import LLM, SamplingParams

t0 = time.perf_counter()
try:
    llm = LLM(model="Qwen/Qwen3-8B", max_model_len=MAXLEN, dtype="float16",
              gpu_memory_utilization=0.85, enable_prefix_caching=False,
              enforce_eager=os.environ.get("EAGER", "1") == "1",
              attention_config={"backend": "CUSTOM"})
except Exception as exc:
    print(f"[WORKER] INIT FAILED after {time.perf_counter()-t0:.0f}s: "
          f"{type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(f"[WORKER] counters={N}", flush=True)
    sys.exit(3)
print(f"[WORKER] init ok in {time.perf_counter()-t0:.0f}s counters={N}", flush=True)
try:
    prompt = " ".join(["token"] * CTX)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1))
    torch.cuda.synchronize()
    print(f"[WORKER] generate ok counters={N}", flush=True)
except Exception as exc:
    print(f"[WORKER] GENERATE FAILED: {type(exc).__name__}: {str(exc)[:300]}",
          flush=True)
    sys.exit(4)
'''


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    path = "/tmp/probe_16k_worker.py"
    with open(path, "w") as fh:
        fh.write(WORKER)

    base = dict(os.environ)
    base["PYTHONPATH"] = "/opt/thunder_vllm"
    base.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    base.setdefault("CTX", "16384")
    configs = [
        ("A_blocking", {"CUDA_LAUNCH_BLOCKING": "1"}, 1800),
        ("B_skip_kernel", {"CUDA_LAUNCH_BLOCKING": "1", "THUNDER_SKIP_KERNEL": "1"}, 1800),
        ("C_skip_gather", {"CUDA_LAUNCH_BLOCKING": "1", "THUNDER_SKIP_GATHER": "1"}, 1800),
        ("D_indirect", {"CUDA_LAUNCH_BLOCKING": "1", "THUNDER_8B_INDIRECT": "1"}, 1800),
        ("E_async", {}, 1800),
    ]
    only = os.environ.get("ONLY", "")
    for tag, over, timeout in configs:
        if only and tag not in only:
            continue
        print(f"\n{'=' * 70}\n== {tag}: {over}\n{'=' * 70}", flush=True)
        env = dict(base)
        env.update(over)
        p = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             cwd="/opt/thunder_vllm", env=env)
        try:
            tail = []
            for ln in p.stdout:
                tail.append(ln.rstrip())
                if ln.startswith(("[GEOM]", "[BT]", "[WORKER]", "[FAULT]")) or any(
                    k in ln for k in ("Traceback", '  File "', "Error", "error",
                                      "thunder_vllm/", "torch.equal")
                ):
                    print(f"  | {ln.rstrip()}", flush=True)
            p.wait(timeout=timeout)
        except Exception:
            p.kill()
            print(f"[driver] {tag} TIMED OUT", flush=True)
        if p.returncode != 0:
            print(f"[driver] {tag} tail:", flush=True)
            for ln in tail[-25:]:
                print(f"  > {ln}", flush=True)
        print(f"[driver] {tag} rc={p.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
