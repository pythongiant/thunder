"""Dump the geometry the engine actually hands the kernel at ctx 16384.

The kernel takes an illegal memory access at that context and nothing in the
grid reproduces it deterministically, so the missing datum is the *metadata*:
which requests the step carries, how long each is, and whether the addresses the
kernel derives from ``seq_lens``/``query_start_loc`` stay inside the buffers.

This wraps ``ThunderAttentionImpl.forward`` and prints, per step, the kernel's
own index arithmetic evaluated on the host side:

    max kv row = (num_reqs-1)*kv_row_stride + max(seq_lens)     < gathered rows?
    max q  row = max(query_start_loc)                            < q rows?

so the step *before* the fault names the overflow, instead of inferring it from
a traceback that points at the first sync after the launch.

Run (inside the CI container):
    CTX=16384 python ci_probe/probe_16k_geom.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

CTX = int(os.environ.get("CTX", "16384"))
GEN = int(os.environ.get("GEN", "32"))
MAXLEN = int(os.environ.get("MAXLEN", "0")) or (CTX + 64)

from thunder_vllm.model.registry import configure, register  # noqa: E402

configure(k_bits=int(os.environ.get("K_BITS", "3")),
          v_bits=int(os.environ.get("V_BITS", "4")))
register()

_SEEN = {"prefill": 0, "decode": 0, "capturing": 0}


def _stats(t: torch.Tensor) -> dict:
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return {}
    c = t.detach().to("cpu")
    return {
        "shape": tuple(int(x) for x in c.shape),
        "min": int(c.min()),
        "max": int(c.max()),
        "sum": int(c.sum()),
        "zeros": int((c == 0).sum()),
        "head": [int(x) for x in c.reshape(-1)[:6]],
        "tail": [int(x) for x in c.reshape(-1)[-6:]],
    }


def _dump(tag: str, md, query) -> None:
    """Evaluate the kernel's addressing on the host side for this step."""
    if torch.cuda.is_current_stream_capturing():
        _SEEN["capturing"] += 1
        return
    try:
        sl = _stats(md.seq_lens)
        qsl = _stats(md.query_start_loc)
        bt = md.block_table
        num_reqs = int(getattr(md, "num_reqs", 0) or 0)
        mblk = int(getattr(md, "max_blocks_per_req", 0) or 0)
        mql = int(getattr(md, "max_query_len", 0) or 0)
        bs = 16
        kv_row_stride = mblk * bs
        # Kernel-side derivation, verbatim from launch_thunder_attention /
        # cute_kernel.__call__.
        n_req_k = int(md.seq_lens.shape[0])
        n_q_blocks = -(-mql // 8)  # chunked-prefill tile packs 8 rows/head
        rows = int(query.shape[0])
        max_kv_row = (num_reqs - 1) * kv_row_stride + sl["max"]
        # A live request's gathered slot is [req*kv_row_stride, req*kv_row_stride
        # + seq_len); the reservation is max_num_reqs * max_blocks_per_req*bs.
        print("[GEOM] " + json.dumps({
            "tag": tag,
            "num_reqs_md": num_reqs,
            "num_reqs_kernel": n_req_k,
            "max_query_len": mql,
            "num_q_blocks": n_q_blocks,
            "is_prefill": bool(getattr(md, "is_prefill", False)),
            "q_rows": rows,
            "kv_row_stride": kv_row_stride,
            "block_table": tuple(int(x) for x in bt.shape),
            "seq_lens": sl,
            "qsl": qsl,
            "sl_gt_kv_row_stride": int((md.seq_lens > kv_row_stride).sum()),
            "sl_gt_max_model_len": int((md.seq_lens > MAXLEN).sum()),
            "qsl_gt_q_rows": int((md.query_start_loc > rows).sum()),
            "max_kv_row": max_kv_row,
            "max_q_row": qsl["max"],
            "q_len_zero_reqs": int(
                (md.query_start_loc[1:] == md.query_start_loc[:-1]).sum()
            ),
            "slot_mapping": _stats(md.slot_mapping),
        }), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[GEOM] dump failed: {type(exc).__name__}: {exc}", flush=True)


def _install() -> None:
    from thunder_vllm.attention import backend as be

    orig = be.ThunderAttentionImpl.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, **kw):
        if attn_metadata is not None:
            kind = "prefill" if int(getattr(attn_metadata, "max_query_len", 0) or 0) > 1 else "decode"
            if kind == "prefill" or _SEEN["decode"] < 2:
                _SEEN[kind] += 1
                _dump(kind, attn_metadata, query)
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, **kw)

    be.ThunderAttentionImpl.forward = forward


_install()

from vllm import LLM, SamplingParams  # noqa: E402

print(f"[probe] ctx={CTX} maxlen={MAXLEN} k={os.environ.get('K_BITS', '3')}"
      f" v={os.environ.get('V_BITS', '4')}", flush=True)
t0 = time.perf_counter()
llm = LLM(model="Qwen/Qwen3-8B", max_model_len=MAXLEN, dtype="float16",
          gpu_memory_utilization=0.85, enable_prefix_caching=False,
          enforce_eager=os.environ.get("EAGER", "1") == "1",
          attention_config={"backend": "CUSTOM"})
print(f"[probe] init={time.perf_counter() - t0:.1f}s", flush=True)

prompt = " ".join(["token"] * CTX)
try:
    t0 = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1))
    torch.cuda.synchronize()
    print(f"[probe] ttft={time.perf_counter() - t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=GEN))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[probe] gen {GEN} in {dt:.2f}s text={out[0].outputs[0].text[:24]!r}",
          flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[probe] FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"[probe] seen={_SEEN}", flush=True)
    sys.exit(3)
print(f"[probe] OK seen={_SEEN}", flush=True)
