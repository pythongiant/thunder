"""Reproduce and bisect the dense gather's CUDA-graph capture failure (FM14).

FAILURE_MODES 14 says the dense (CSR) gather cannot be captured: vLLM's cudagraph
memory profiling dies with `cudaErrorStreamCaptureUnsupported`, in both of its
forms, after its allocations were removed. The engine is an expensive place to
bisect that, so this probe captures the same code path on synthetic buffers.

What the engine's forward does inside the captured region, in order:

    build_csr_device(...)            device CSR metadata (persistent buffers)
    gather_csr_payload(..., nrows=)  4 x index_select(..., out=<reservation view>)
    launch_thunder_attention(...)    the kernel
    _merge_splits(...)               triton merge (split-K decode only)

The modes below remove one piece at a time. `no_build` substitutes a static CSR
index (identity selection), `no_gather` reserves without selecting, `no_kernel`
and `no_merge` use the launcher's own knobs.

Run: python ci_probe/probe_dense_capture.py          # sweep, one subprocess per mode
     python ci_probe/probe_dense_capture.py --worker # one mode (env: DC_MODE)
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback

WORKER = r'''
import os
import sys

import torch

from benchmarks.bench_common import make_synthetic_batch
from thunder_vllm.attention.cute_kernel import (
    ThunderAttentionForward,
    launch_thunder_attention,
)
from thunder_vllm.attention.paged_kv import CsrIndex, make_paged_kv_manager
from thunder_vllm.attention.tile_shape import tile_shape

MODE = os.environ.get("DC_MODE", "full")
BATCH, Q_LEN, LIVE_BLOCKS, BT_W = 1024, 1, 8, 1032
HD, HQ, HK, BS = 128, 32, 8, 16

torch.manual_seed(0)
# `seqlen_k` sizes the q/o buffers; the live cache is `LIVE_BLOCKS` per request.
sb = make_synthetic_batch(BATCH, 16384, HQ, HK, HD, device="cuda")
live = LIVE_BLOCKS
sb.kv_cache = sb.kv_cache[: BATCH * live]
sb.kv_scales = sb.kv_scales[: BATCH * live]
bt = torch.arange(BATCH * live, dtype=torch.int32, device="cuda").reshape(BATCH, live)
bt = torch.cat([bt, torch.zeros((BATCH, BT_W - live), dtype=torch.int32, device="cuda")], 1)
sb.block_table = bt
sb.seq_lens = torch.full((BATCH,), live * BS, dtype=torch.int32, device="cuda")

mgr = make_paged_kv_manager(sb.layout, max_num_reqs=BATCH,
                            max_model_len=BT_W * BS, device="cuda")
meta = type("M", (), {
    "seq_lens": sb.seq_lens, "block_table": sb.block_table,
    "query_start_loc": torch.arange(0, BATCH * Q_LEN + 1, Q_LEN, device="cuda",
                                    dtype=torch.int32),
    "max_query_len": Q_LEN,
})()
nb = int(sb.kv_cache.shape[0])
indirect = MODE != "no_gather_indirect_off"

# --- eager warm-up: allocations and compiles must be behind us -------------
csr = mgr.csr_for_step(meta, nb)
gathered = mgr.gather_csr_payload(csr, sb.kv_cache, sb.kv_scales)
q = sb.q[: BATCH * Q_LEN]
out = torch.empty_like(sb.q)
tile = tile_shape(False, BATCH, None)
kernel = ThunderAttentionForward(
    head_dim=HD, K_BITS=sb.layout.k_bits, V_BITS=sb.layout.v_bits,
    qhead_per_kvhead=4, gqa_rows_per_head=1, is_causal=False,
    m_block_size=tile["m_block"], n_block_size=tile["n_block"],
    num_threads=tile["num_threads"],
)
from thunder_vllm.attention.splits import decode_split_count
S = 1 if MODE == "no_merge" else decode_split_count(
    16384, is_prefill=False, num_kv_groups=4, num_reqs=BATCH,
    tile_n=tile["n_block"],
)
kw = dict(num_splits=S, gqa_pack=True, onepass=True, reg_rescale=True, causal_bound=True,
          indptr=csr.indptr, indirect=indirect)

def launch(g):
    launch_thunder_attention(kernel, q, g, out, meta, HD ** -0.5,
                             quantizer=sb.quantizer, **kw)

for _ in range(3):
    launch(mgr.gather_csr_payload(csr, sb.kv_cache, sb.kv_scales))
torch.cuda.synchronize()
print(f"[{MODE}] warm ok S={S} indirect={indirect} layers={os.environ.get('DC_LAYERS', '36')}", flush=True)

# --- capture ---------------------------------------------------------------
cap_rows = max(min(BATCH * BT_W, mgr.page_rows), 0)
static = CsrIndex(
    indptr=torch.arange(0, (BATCH + 1) * live * BS, live * BS, dtype=torch.int32,
                        device="cuda"),
    sel=torch.arange(cap_rows, dtype=torch.int64, device="cuda") % max(nb, 1),
    nrows=cap_rows, num_blocks=nb,
)

def build():
    if MODE == "no_build":
        return static
    return mgr.build_csr_device(meta, nb, compute_nrows=False)

def gather(idx):
    if MODE == "no_gather":
        return mgr.reserve(nb)
    return mgr.gather_csr_payload(idx, sb.kv_cache, sb.kv_scales, nrows=cap_rows)

graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        idx = build()
        g = gather(idx)
        launch(g)
        torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001
    print(f"[{MODE}] CAPTURE FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    sys.exit(3)
graph.replay()
torch.cuda.synchronize()
print(f"[{MODE}] capture ok", flush=True)
'''


def main() -> int:
    if "--worker" in sys.argv:
        exec(compile(WORKER, "worker", "exec"), {"__name__": "__main__"})
        return 0
    modes = ["full", "no_build", "no_gather", "no_kernel", "no_merge"]
    path = "/tmp/dense_capture_worker.py"
    with open(path, "w") as fh:
        fh.write(WORKER)
    base = dict(os.environ)
    base["PYTHONPATH"] = "/opt/thunder_vllm"
    base.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    for mode in modes:
        env = dict(base)
        env["DC_MODE"] = mode
        if mode == "no_kernel":
            env["THUNDER_SKIP_KERNEL"] = "1"
        print(f"\n== {mode} ==", flush=True)
        p = subprocess.run([sys.executable, path], env=env, cwd="/opt/thunder_vllm",
                           capture_output=True, text=True, timeout=900)
        for ln in p.stdout.splitlines():
            if ln.startswith("[") or "Error" in ln or "error" in ln:
                print(f"  | {ln}", flush=True)
        if p.returncode != 0:
            print("  > " + "\n  > ".join(p.stdout.splitlines()[-6:]), flush=True)
        print(f"[driver] {mode} rc={p.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
