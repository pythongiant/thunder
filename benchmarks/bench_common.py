"""Shared benchmark utilities.

Wraps FA4's ``flash_attn/cute/bench_utils.py`` where it is usable as-is, and
adds the TurboQuant-specific numbers:

* :func:`effective_kv_bytes` -- the bytes the *packed* cache actually reads, the
  denominator for the only bandwidth number that matters for this kernel.
* :func:`run_with_graph` -- the only honest way to time decode.
* :func:`make_synthetic_batch` -- a realistic paged cache + block table.
* :func:`memory_tokens_per_gb` -- the capacity story.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch

try:  # pragma: no cover - optional dependency
    from triton.testing import do_bench as triton_do_bench
except Exception:  # noqa: BLE001
    triton_do_bench = None

from thunder_vllm.attention.cache_layout import (
    ThunderCacheLayout,
    allocate_kv_cache,
)
from thunder_vllm.quant.quantizer import ThunderQuantizer


# ---------------------------------------------------------------------------
# Theoretical bytes / FLOPs
# ---------------------------------------------------------------------------
def flops(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    headdim: int,
    headdim_v: int | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
) -> float:
    """Attention FLOPs (2 * the multiply-adds), matching FA4's signature.

    ``headdim_v`` defaults to ``headdim``; FA4 requires it explicitly, this
    wrapper does not, so call sites read the same as the spec's.
    """
    if headdim_v is None:
        headdim_v = headdim
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    elif window_size == (None, None):
        avg_seqlen = seqlen_k
    else:
        row = torch.arange(seqlen_q)
        left = (
            torch.clamp(row + seqlen_k - seqlen_q - window_size[0], min=0)
            if window_size[0] is not None
            else torch.zeros_like(row)
        )
        right = (
            torch.clamp(row + seqlen_k - seqlen_q + window_size[1], max=seqlen_k - 1)
            if window_size[1] is not None
            else torch.full_like(row, seqlen_k - 1)
        )
        avg_seqlen = (right - left + 1).float().mean().item()
    return float(batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v))


def bandwidth_fwd_bytes(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    headdim: int,
    headdim_v: int | None = None,
    dtype_bytes: int = 2,
    shared_kv: bool = False,
    nheads_kv: int | None = None,
) -> float:
    """HBM traffic for an fp16 attention pass: read Q,K,V + write O."""
    headdim_v = headdim_v or headdim
    nheads_kv = nheads_kv or nheads
    q = batch * nheads * seqlen_q * headdim
    k = batch * nheads_kv * seqlen_k * headdim
    v = 0 if shared_kv else batch * nheads_kv * seqlen_k * headdim_v
    o = batch * nheads * seqlen_q * headdim_v
    return float((q + k + v + o) * dtype_bytes)


def effective_kv_bytes(layout: ThunderCacheLayout, seqlen_k: int, batch: int = 1) -> float:
    """Bytes the TurboQuant kernel reads for the KV cache.

    Per (token, kv-head) position: packed K + packed V codes plus two fp16
    norms. This is the number the packed cache reads, and it is what the
    effective bandwidth is computed from.
    """
    norm_bytes = 2 * torch.tensor([], dtype=layout.scale_dtype).element_size()
    per_token = layout.k_packed_bytes + layout.v_packed_bytes + norm_bytes
    return float(batch * layout.num_kv_heads * seqlen_k * per_token)


def memory_tokens_per_gb(
    layout: ThunderCacheLayout, budget_gb: float = 40.0
) -> float:
    """How many KV positions fit in ``budget_gb`` for this layout.

    The capacity metric: this is the whole point of packing K and V.
    """
    bytes_per_token_per_head = (
        layout.k_packed_bytes
        + layout.v_packed_bytes
        + 2 * torch.tensor([], dtype=layout.scale_dtype).element_size()
    )
    total = budget_gb * (1024**3)
    return total / (bytes_per_token_per_head * layout.num_kv_heads)


def fp16_tokens_per_gb(num_kv_heads: int, head_dim: int, budget_gb: float = 40.0) -> float:
    per_token = 2 * num_kv_heads * head_dim * 2  # K and V, fp16
    return budget_gb * (1024**3) / per_token


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def do_bench(fn, warmup: int = 25, rep: int = 100, **kwargs) -> float:
    """Median wall time in **milliseconds**, via triton's do_bench."""
    if triton_do_bench is None:  # pragma: no cover
        raise RuntimeError("triton.testing.do_bench is unavailable")
    return float(triton_do_bench(fn, warmup=warmup, rep=rep, **kwargs))


def do_bench_stats(fn, warmup: int = 25, rep: int = 100) -> dict[str, float]:
    """Median plus the p20/p80 spread, over individual timed iterations."""
    times: list[float] = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return {
        "median_ms": float(t.median()),
        "p20_ms": float(t.quantile(0.2)),
        "p80_ms": float(t.quantile(0.8)),
        "min_ms": float(t.min()),
        "n": rep,
    }


@dataclass
class GraphTiming:
    median_ms: float
    p20_ms: float
    p80_ms: float


def run_with_graph(fn, warmup: int = 25, iters: int = 100) -> GraphTiming:
    """Capture ``fn`` into a CUDA graph and time only the replays.

    Eager timing of decode is dominated by launch overhead; a graph is what a
    serving stack actually runs, so this is the number to report.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return GraphTiming(
        median_ms=float(t.median()),
        p20_ms=float(t.quantile(0.2)),
        p80_ms=float(t.quantile(0.8)),
    )


# ---------------------------------------------------------------------------
# Synthetic batch
# ---------------------------------------------------------------------------
@dataclass
class SyntheticBatch:
    q: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    kv_cache: torch.Tensor
    kv_scales: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    layout: ThunderCacheLayout
    quantizer: ThunderQuantizer


def make_synthetic_batch(
    batch: int,
    seqlen: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    *,
    k_bits: int = 4,
    v_bits: int = 4,
    block_size: int = 16,
    seed: int = 0,
) -> SyntheticBatch:
    """Build a batch with a realistic paged KV cache and block table."""
    from thunder_vllm.attention.cache_layout import reshape_and_cache_ref

    dev = torch.device(device)
    torch.manual_seed(seed)
    layout = ThunderCacheLayout(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        k_bits=k_bits,
        v_bits=v_bits,
        block_size=block_size,
    )
    quant = ThunderQuantizer(head_dim, k_bits, v_bits, device=dev)

    blocks_per_req = math.ceil(seqlen / block_size)
    num_blocks = batch * blocks_per_req
    kv_cache, kv_scales = allocate_kv_cache(
        num_blocks, block_size, num_kv_heads, head_dim, k_bits, v_bits, dtype, dev
    )
    block_table = torch.arange(
        num_blocks, device=dev, dtype=torch.int32
    ).reshape(batch, blocks_per_req)

    key = torch.randn(seqlen, num_kv_heads, head_dim, device=dev, dtype=dtype)
    value = torch.randn(seqlen, num_kv_heads, head_dim, device=dev, dtype=dtype)
    slots = torch.arange(seqlen, device=dev, dtype=torch.long)
    reshape_and_cache_ref(key, value, slots, kv_cache, kv_scales, quant, layout)

    return SyntheticBatch(
        q=torch.randn(seqlen, num_heads, head_dim, device=dev, dtype=dtype),
        key=key,
        value=value,
        kv_cache=kv_cache,
        kv_scales=kv_scales,
        block_table=block_table,
        seq_lens=torch.full((batch,), seqlen, device=dev, dtype=torch.int32),
        layout=layout,
        quantizer=quant,
    )


def attention_ref(q, k, v, causal: bool = False, scale: float | None = None):
    """PyTorch reference attention on ``(seqlen, nheads, headdim)`` tensors."""
    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    if k_t.shape[0] != q_t.shape[0]:
        group = q_t.shape[0] // k_t.shape[0]
        k_t = k_t.repeat_interleave(group, dim=0)
        v_t = v_t.repeat_interleave(group, dim=0)
    scale = scale if scale is not None else q_t.shape[-1] ** -0.5
    scores = torch.einsum("hqd,hkd->hqk", q_t.float() * scale, k_t.float())
    if causal:
        nq, nk = scores.shape[-2:]
        mask = torch.ones(nq, nk, device=scores.device, dtype=torch.bool).tril(
            diagonal=nk - nq
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,hkd->qhd", attn, v_t.float())


@contextlib.contextmanager
def nvml_sampler(interval_s: float = 0.1):
    """Yield a dict that, on exit, holds the max GPU utilisation sampled."""
    stats = {"util_gpu": 0.0, "mem_used_mb": 0.0}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        import threading
        import time

        stop = threading.Event()

        def loop():
            while not stop.is_set():
                u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats["util_gpu"] = max(stats["util_gpu"], float(u.gpu))
                stats["mem_used_mb"] = max(stats["mem_used_mb"], m.used / 2**20)
                time.sleep(interval_s)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        try:
            yield stats
        finally:
            stop.set()
            t.join(timeout=1.0)
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        yield stats


def assert_parity(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2):
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Batched-benchmark helpers (P3 harness)
# ---------------------------------------------------------------------------
def fp16_per_token_bytes(num_kv_heads: int, head_dim: int) -> int:
    """fp16 KV bytes per token across all KV heads (K + V)."""
    return 2 * int(num_kv_heads) * int(head_dim) * 2


def kv_bytes_per_step(per_token_bytes: int, batch: int, seqlen: int) -> float:
    """KV bytes the attention kernel reads in one decode step.

    ``per_token_bytes`` already spans all KV heads for one token.
    """
    return float(int(per_token_bytes) * int(batch) * int(seqlen))


def effective_bw_gbps(nbytes: float, seconds: float) -> float:
    """Effective bandwidth (GB/s) for ``nbytes`` moved in ``seconds``.

    ``nbytes`` uses 1e9-byte GB so it reads as a fraction of HBM spec numbers.
    Returns 0.0 for a non-positive interval.
    """
    if seconds is None or seconds <= 0:
        return 0.0
    return float(nbytes) / float(seconds) / 1e9


def kv_bytes_per_token_from_layout(layout) -> int:
    """Packed KV bytes per token (all KV heads) incl. fp16 norms."""
    norm_bytes = 2 * torch.tensor([], dtype=layout.scale_dtype).element_size()
    return int(layout.num_kv_heads * (layout.k_packed_bytes + layout.v_packed_bytes + norm_bytes))
