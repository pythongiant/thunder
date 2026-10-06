# thunder

A quantized-attention architecture for Blackwell GPUs. A TurboQuant-compressed
KV cache — 3/4-bit codes plus per-head norms — flows through a fused CuTeDSL
attention kernel straight into the tensor cores. No fp16 KV is ever
materialized: the cache is 3.9x smaller than fp16 and attention reads it in
place.

The vLLM integration is the serving vehicle and the end-to-end harness, not the
product: it exists so kernel throughput can be checked against serving
throughput.

**Author:** Srihari Unnikrishnan · [@pythongiant](https://github.com/pythongiant) · srihari.unnikrishnan@gmail.com

## Benchmarks

All numbers: B200, `Qwen/Qwen3-8B` (32 Q heads / 8 KV heads / head_dim 128,
GQA 4:1), batch 1, greedy, 32 generated tokens, fp16 weights.

### What this pass changed

Every row is a measured attention launch on the engine-like shapes, in the
configuration the engine ships (single-pass online softmax, register rescale,
causal bound, split-K policy), `k_bits=4`/`v_bits=4`, CUDA-graph medians.

| workload | before | after | win |
|---|---|---|---|
| decode, 4k context | 0.2732 ms | **0.1843 ms** | **1.48x** |
| decode, 32k context | 1.9351 ms | **1.3887 ms** | **1.39x** |
| prefill, 4k | 7.3042 ms | **5.6884 ms** | **1.28x** |
| decode 4k vs dequant-fp16 reference | 2.498 ms | 0.184 ms | **13.6x** |
| decode 32k vs dequant-fp16 reference | 5.602 ms | 1.389 ms | **4.0x** |
| KV cache at 32k, one request | 128.0 MiB (fp16) | **33.0 MiB** | **3.9x smaller** |

The win came from occupancy, not from the arithmetic: shared memory was 121 KB,
just over the line for two CTAs per SM, so each SM ran one CTA and sat ~75%
idle. Writing the output staging buffer as fp16 took the kernel to ~89 KB, and
the split-K knee moved from 4 to 8 as the extra CTAs became resident (plus a
two-phase packed load and a fused split-K merge). Details in *Why it is fast*.

The dequant-fp16 column is a kernel-level sanity bound, not the competition —
it measures the value of fusing dequantization into attention, not a win over
another implementation. The competition is the next table.

### Against upstream TurboQuant KV

Upstream's GQA/MHA KV path is **stock vLLM** since
[vllm#38479](https://github.com/vllm-project/vllm/pull/38479) — vLLM 0.25.1 with
`--kv-cache-dtype turboquant_3bit_nc`, no plugin — which is why the two kernels
are measured on two pins.

| ctx | kernel | KV cache | KV size @ ctx | TTFT | ITL | output tok/s |
|---|---|---|---|---|---|---|
| 4096 | upstream vLLM 0.25.1 | fp16 | 16.0 MiB | 88.9 ms | 3.10 ms | 172.9 |
| 4096 | upstream vLLM 0.25.1 | `turboquant_3bit_nc` | 3.4 MiB | **74.3 ms** | 7.03 ms | 109.5 |
| 4096 | thunder (pinned vLLM) | fp16, control | 16.0 MiB | 99.0 ms | 3.04 ms | 165.6 |
| 4096 | thunder (pinned vLLM) | packed k4v4 | 4.1 MiB | *205 ms* | *6.6 ms* | ***151*** |
| 32768 | upstream vLLM 0.25.1 | fp16 | 128.0 MiB | 706.3 ms | 3.78 ms | 38.9 |
| 32768 | upstream vLLM 0.25.1 | `turboquant_3bit_nc` | 27.0 MiB | 1380.2 ms | 12.23 ms | 18.2 |
| 32768 | thunder (pinned vLLM) | fp16, control | 128.0 MiB | 724.1 ms | 3.60 ms | 38.3 |
| 32768 | thunder (pinned vLLM) | packed k4v4 | 33.0 MiB | *—* | *50.0 ms* | ***20*** |

*Italic = derived, not measured*: this plugin cannot yet be served end to end
(graph capture faults — see *Known issues*), so its rows are the measured
attention launch times 36 layers, which excludes weight GEMMs, sampling and
engine overhead and is therefore a best case. The 32k TTFT has no measurement to
derive from. Every other row is measured.

**KV size** is one request's cache at that context, from the format rather than
from a running engine: bytes per token (all 8 KV heads) x context. Per
token-head, fp16 is 512 B; this plugin's 4-bit slot is 64 B K + 64 B V + 2 B
norms = 132 B, i.e. **1056 B/token, 3.9x smaller than fp16** (3-bit K gives
928 B/token); upstream's `turboquant_3bit_nc` is 108 B per token-head = 864
B/token, per their published per-vector table. Ours is computed by
`benchmarks/bench_common.py::kv_bytes_per_token_from_layout`, so it tracks the
layout code rather than a hand-written constant. Note the two compressed formats
are not the same size: ours is ~22% larger than upstream's at equal context,
which the throughput comparison should be read against.

Reading the table:

- **The fp16 rows are the control**, and they agree across the two stacks within
  a few percent (3.04 vs 3.10 ms ITL at 4k, 3.60 vs 3.78 ms at 32k), so the two
  environments are comparable.
- **Upstream's compression buys memory, not speed at batch 1**: 2.3x (4k) and
  3.2x (32k) more inter-token latency than its own fp16 KV. TTFT improves at 4k
  (74.3 vs 88.9 ms) and degrades 2x at 32k.
- **Thunder's decode is now at parity with upstream at 4k, and still behind at
  32k.** Its rows are attention only — 36 layers of the measured launch — so they
  are a *lower bound* on its step time; upstream's column is a whole step, with
  weights and engine overhead included. On that footing: 6.6 ms against 7.03 ms
  at 4k (attention alone now costs less than their entire step, so parity is
  plausible but not demonstrated — our step also carries non-attention work), and
  50.0 ms against 12.23 ms at 32k, a 4x deficit that no amount of the same
  optimisation closes. Prefill is 205 ms vs 74.3 ms TTFT.

The remaining distance is the schedule, not the compression: FlashAttention-4
does the same 16k decode shape in 0.196 ms per layer — 7.0 ms per 36-layer step,
reading *dense fp16* — still about 4x faster than this kernel while moving 3.9x
more bytes. That measured gap, not the format, is what the v2 work is for.

At the kernel level, one attention launch in the configuration the engine
actually launches (single-pass online softmax, register-local rescale, causal
bound, engine split-K policy = 4, `k_bits=4` / `v_bits=4`, CUDA-graph medians)
takes **0.184 ms** at 4k decode, **1.389 ms** at 32k decode and **5.688 ms** at
4k prefill, against **2.498 / 5.602 / 14.769 ms** for the same cache
dequantized to fp16 and run through SDPA — 13.6x, 4.0x and 2.6x. That reference
is a kernel-level sanity bound: it says fusing dequantization into attention is
worth this much, not that the kernel is competitive. Run-to-run noise on these
medians is about ±1%. Full method, caveats and raw rows:
`benchmarks/results/upstream_vs_ours.md`.

### Correctness

The kernel is checked against a dequantized-fp16 oracle at `atol=rtol=1e-2`
(`tests/test_correctness.py`): prefill 128x2048, decode 1x8192, several bit
widths, and split-K decode at S=2/4. On B200 the suite is green except
`test_cuda_graph_replay_parity`, which is an unwired stub. The CPU suite is
`python -m pytest tests/`.

## Installation

Requirements: a Blackwell GPU (sm_100/sm_110), CUDA 12.8+, PyTorch 2.8+,
`nvidia-cutlass-dsl`, Triton. The end-to-end harness additionally needs a vLLM
build with the pinned commit (see `docs/RUNBOOK.md`).

```sh
pip install -e .
```

Register the backend before constructing the engine:

```python
from thunder_vllm.model import registry

registry.configure(k_bits=4, v_bits=4)
registry.register()

# then select it at engine construction:
#   attention_config={"backend": "CUSTOM"}
```

Kernel-level checks and A/Bs:

```sh
PYTHONPATH=. python3 ci_probe/correctness_check.py     # store, gather, kernel vs oracle
PYTHONPATH=. python3 ci_probe/kernel_bench.py          # kernel A/B
PYTHONPATH=. python3 ci_probe/kernel_stage_probe.py    # per-stage decomposition
python -m benchmarks.fa4_matrix --quick                 # FA4 vs ours, frozen gate
python -m pytest tests/ -q                              # CPU suite
```

GPU work runs on Modal (there is no local CUDA). These runners are local
tooling, not tracked sources:

```sh
modal run ci/modal_app.py --mode loop                  # kernel latency, engine config
modal run ci/modal_app.py --mode test                  # whole suite on B200
modal run ci/modal_app.py --mode e2e                   # serving measurement
modal run ci/modal_app.py --mode grid --grid "..."     # config attribution grid
modal run ci_probe/modal_upstream_baseline.py          # upstream TurboQuant KV baseline
```

End-to-end, on a vLLM build that can select the backend:

```sh
python -m benchmarks.bench_vs_thunder_vllm --model Qwen/Qwen3-8B
```

## Why it is fast

**The cache is 3.9x smaller and attention reads it as-is.** A token's KV slot is
64 B of packed K codes + 64 B of packed V codes + two fp16 norms = 132 B at
4-bit, against 512 B for fp16 (`k_bits=3` gives 116 B, 4.4x). Nothing is expanded
into HBM: the codes travel from the cache to SMEM to the tensor cores, so a
decode step streams a quarter of the bytes.

**Dequantization is fused into the tile pipeline, and it is cheap.** Per KV tile
the kernel loads packed bytes, unpacks nibbles and gathers through a per-layer
LUT into an SMEM fp16 tile, then feeds the QK and PV MMAs. That stage measures
14-19% of decode time (K and V together, 4k to 32k) — less than the 3.9x traffic
it saves.

**The MMAs never see quantized operands.** Dequantized fp16 tiles feed `mma.sync`
with fp32 accumulators, so the arithmetic is an ordinary fp16 attention kernel;
the compression changes the traffic, not the math. That is why the kernel can
match a dequantized-fp16 oracle to 1e-2 while reading 3.9x fewer bytes.

**Batch-1 decode fills the machine with split-K.** A 32-head decode grid is only
32 CTAs on 148 SMs, so the KV range is split 8 ways: 256 CTAs, measuring **3.5x**
(4k) and **3.9x** (32k) over the unsplit schedule. The knee is at 8 rather than
4, and that follows from the next point — extra CTAs only help once the SM can
keep two resident. The partials are reduced by an online-softmax rescale in one
fused kernel; the ten-launch torch chain it replaced cost 13% of 4k decode.

**Two CTAs per SM, not one.** Shared memory was 121 KB, just over the 113 KB
line, so each SM ran a single CTA and sat largely idle — about 29k cycles per KV
tile against ~7k of modelled work. Writing the output staging buffer as fp16
instead of fp32 (it is pure staging: written from the accumulator fragment, read
back one row per thread) took the kernel to ~89 KB and fits two CTAs. Prefill,
which has 2048 CTAs, fell 21% on that change alone; decode needed the split-count
change to turn the same headroom into CTAs.

**The packed load no longer waits on itself.** It was one dependent load->store
chain per byte, 32 iterations per thread per buffer, and ablating the load
entirely measured 22% of decode. Every global load now issues before any store,
so the latencies overlap instead of serializing (-2.4%).

**The KV is traversed once.** Single-pass online softmax removes the separate
row-max pass — otherwise every tile loads and dequantizes K twice — and
register-local accumulator rescale removes two SMEM round-trips per tile.
Causal tile skipping drops fully-masked tiles in prefill. All three are on by
default.

**A GQA group shares one reconstruction.** With 32 Q heads over 8 KV heads the
QK and PV math differ per head but the KV reconstruction does not, so the
GQA-packed schedule reconstructs each tile once for the four heads that consume
it.

**Host overhead is flat.** The fast-launch cache keys the compiled CuTeDSL
function by kernel config instead of re-tracing MLIR per call: a launch costs
~0.35 ms of host work rather than ~407 ms, which is what makes eager usable and
graph capture cheap.

Where this is *not* fast yet: against FlashAttention-4 on the same shapes we are
2.25x off at 4k decode and about 4x at 16k
(`benchmarks/results/fa4_matrix_b200_full.md`). The remaining gap is
schedule-level, not parameter-level — the MMA M tile is bound to the warp count
(`m_block == num_warps * 16`), so a one-row decode pays for a 64-row tile, and
the operand copies are universal 16-bit SMEM copies rather than `ldmatrix`, which
this CuTeDSL build does not expose. Closing it means deriving the pipeline from
FA4 (async/TMA movement, UMMA/TMEM accumulators, device-side reduction) rather
than tuning this schedule further; the frozen FA4 tree to build on is vendored at
`thunder_vllm/attention/v2/fa4/`.

## Known issues

- **CUDA-graph capture of the `CUSTOM` backend faults** with
  `cudaErrorStreamCaptureInvalidated` (reproduced at 4k and 32k). Graph capture
  is the path vLLM serves with, so this blocks every end-to-end number.
- **Eager is host-bound**: 515 ms per token at 4k on Qwen3-8B, dominated by the
  CuTeDSL launcher rather than the GPU. Eager numbers are not comparable.
- vLLM pads its block table past `ceil(max_model_len / block_size)` (260 → 264
  columns at `block_size=16`, `max_model_len=4160`). The gather reservation now
  follows the table the engine hands over
  (`make_paged_kv_manager(..., max_blocks_per_req=...)`), so serving at arbitrary
  `max_model_len` no longer raises `block_table ... exceeds reserved`.

## Repository layout

```text
thunder_vllm/
  attention/      fused CuTeDSL kernel, paged KV, metadata, split-K, backends
  attention/v2/   FA4-derived execution skeleton (vendored) + TurboQuant producer
  quant/          rotation, Lloyd-Max codebooks, packing
  model/          vLLM registry and integration
  utils/          logging, telemetry
benchmarks/        FA4 matrix, end-to-end harness, shared metrics
benchmarks/results/ frozen FA4 matrix and the upstream comparison
ci_probe/          correctness checks, kernel/stage probes, GPU probes
docs/              roadmap, pipeline v2, ablation, benchmarks, runbook
tests/             test suite (CPU here; GPU cases are marked and run on Modal)
```

## License

Apache-2.0. The vendored FlashAttention-4 tree under
`thunder_vllm/attention/v2/fa4/` is BSD-1-Clause, © Dao-AILab.
