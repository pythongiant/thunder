"""Tracked P3 batched worker.

Runs one config per process (clean allocator, single tenant) with exact-token
natural prompts and greedy decode. Driven by env:

  PB_CFG: fp16 | ours-eager | ours-graph
  PB_PAIRS: e.g. "1:4096,16:4096,1:16384,16:16384"

This is the canonical worker source. `studio_batched.py` spawns it directly;
the Modal driver may reuse this same file so both paths measure the same thing.
"""

import json
import os
import threading
import time

import torch
from vllm import LLM, SamplingParams
from thunder_vllm.utils.telemetry import system_info

_SYS = system_info()  # machine, versions, env flags, git rev: one per config

CFG = os.environ["PB_CFG"]
PAIRS = [tuple(int(x) for x in p.split(":")) for p in os.environ["PB_PAIRS"].split(",")]
CORPUS = ("The quick brown fox jumps over the lazy dog. " * 4000)
GEN = 64

if CFG != "fp16":
    from thunder_vllm.model.registry import configure, register
    configure(k_bits=3, v_bits=4)
    register()

kw = dict(model="Qwen/Qwen3-8B", max_model_len=40960, dtype="float16",
          gpu_memory_utilization=0.75, block_size=16, enable_prefix_caching=False)
if CFG == "fp16":
    llm = LLM(**kw)
elif CFG == "ours-eager":
    llm = LLM(**kw, enforce_eager=True, attention_config={"backend": "CUSTOM"})
else:
    llm = LLM(**kw, enforce_eager=False, attention_config={"backend": "CUSTOM"},
              compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE",
                                  "cudagraph_capture_sizes": [1],
                                  "max_cudagraph_capture_size": 1})

tok = llm.get_tokenizer()
_base = tok.encode(CORPUS)

from benchmarks.bench_common import (  # noqa: E402
    effective_bw_gbps, fp16_per_token_bytes, kv_bytes_per_step,
    kv_bytes_per_token_from_layout, memory_tokens_per_gb,
)
Hk, D = 8, 128
if CFG == "fp16":
    per_token = fp16_per_token_bytes(Hk, D)
    capacity_per_gib = (1024 ** 3) / per_token  # tokens per GiB
else:
    from thunder_vllm.attention.cache_layout import ThunderCacheLayout
    cfg = llm.llm_engine.vllm_config.cache_config
    bs = int(getattr(cfg, "block_size", 16) or 16)
    lay = ThunderCacheLayout(num_kv_heads=Hk, head_dim=D, block_size=bs,
                             k_bits=3, v_bits=4)
    per_token = kv_bytes_per_token_from_layout(lay)
    capacity_per_gib = memory_tokens_per_gb(lay, budget_gb=1.0)
cc = llm.llm_engine.vllm_config.cache_config
nb = int(getattr(cc, "num_gpu_blocks", 0) or 0)
bs = int(getattr(cc, "block_size", 0) or 16)
kv_mb = nb * bs * per_token / 1e6
print(f"[PB] cfg={CFG} per_token={per_token} num_blocks={nb} kv_mb={kv_mb:.0f} "
      f"capacity_tokens_per_GiB={capacity_per_gib:.0f}", flush=True)


class Util:
    """Sample GPU utilisation/memory on a background thread."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.max_util = 0.0
        self.max_mem = 0.0
        self._stop = threading.Event()
        self._th = None

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                self.max_util = max(self.max_util, float(u.gpu))
                self.max_mem = max(self.max_mem, m.used / 1e6)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
        return False


def prompts(batch, ctx):
    ids = (_base * (ctx // len(_base) + 1))[:ctx]
    return [{"prompt_token_ids": list(ids)} for _ in range(batch)]


def gen(p, mt):
    t0 = time.perf_counter()
    o = llm.generate(p, SamplingParams(temperature=0.0, max_tokens=mt,
                                       min_tokens=mt, ignore_eos=True))
    torch.cuda.synchronize()
    return time.perf_counter() - t0, o


for B, CTX in PAIRS:
    try:
        p = prompts(B, CTX)
        gen(p, 8)                                   # warmup (compile/capture)
        torch.cuda.reset_peak_memory_stats()
        with Util() as u:
            t1, _ = gen(p, 1)
            t64, o = gen(p, GEN)
        itl = (t64 - t1) / (GEN - 1)
        ntokens = sum(len(x.outputs[0].token_ids) for x in o)
        kv_step = kv_bytes_per_step(per_token, B, CTX)
        row = dict(cfg=CFG, batch=B, ctx=CTX,
                   model="Qwen/Qwen3-8B", k_bits=3, v_bits=4,
                   ttft_ms=t1 * 1e3, itl_ms=itl * 1e3,
                   agg_tok_s=ntokens / t64, ntokens=ntokens,
                   kv_bytes_per_step=kv_step,
                   kv_bw_gbps=effective_bw_gbps(kv_step, itl),
                   util_pct=u.max_util, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                   text=o[0].outputs[0].text[:24],
                   sys=_SYS)
        print("[PB-JSON] " + json.dumps(row), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[PB] FAIL B={B} ctx={CTX}: {type(exc).__name__}: {exc}", flush=True)
