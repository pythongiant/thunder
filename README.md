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

## Where it stands

- **Kernel**: one attention launch at the engine's shapes measures 0.352 ms
  (decode, batch 16, 4k), 2.73 ms (decode, batch 16, 32k), 0.217 ms (decode,
  batch 1, 32k) and 4.66 ms (prefill, 4k) — against 5.61 ms and 14.92 ms for the
  same cache dequantized to fp16 and run through SDPA at the last two.
- **Serving**: batch 1 at 4k runs under CUDA graphs end to end at 7.3-7.5
  output tok/s. 16k and 32k do not yet — see *Known issues*.
- **Correctness**: the kernel matches a dequantized-fp16 oracle at
  `atol=rtol=1e-2`, and the GPU suite is green except one unwired stub.

## Benchmarks

All numbers: B200, `Qwen/Qwen3-8B` (32 Q heads / 8 KV heads / head_dim 128,
GQA 4:1), batch 1 unless stated, greedy, 32 generated tokens, fp16 weights.

### Kernel

Every row is a measured attention launch on the engine-like shapes, in the
configuration the engine ships (single-pass online softmax, register rescale,
causal bound, the split-K and tile policies in `thunder_vllm/attention/`, GQA
packing), `k_bits=4` / `v_bits=4`, CUDA-graph medians. Run-to-run noise on these
medians is about ±1%.

| workload | thunder | dequant-fp16 reference | ratio |
|---|---|---|---|
| decode, batch 16, 4k context | **0.352 ms** | — | — |
| decode, batch 16, 32k context | **2.73 ms** | — | — |
| decode, batch 1, 32k context | **0.217 ms** | 5.61 ms | **25.9x** |
| prefill, 4k | **4.66 ms** | 14.92 ms | **3.20x** |

The dequant-fp16 column is a kernel-level sanity bound — the same cache
dequantized to fp16 and run through SDPA — not the competition. It measures the
value of fusing dequantization into attention, not a win over another
implementation. The competition is the next table.

### Against upstream TurboQuant KV

Upstream's GQA/MHA KV path is **stock vLLM** since
[vllm#38479](https://github.com/vllm-project/vllm/pull/38479) — vLLM 0.25.1 with
`--kv-cache-dtype turboquant_3bit_nc`, no plugin — which is why the two stacks
are measured on two pins.

| ctx | stack | KV cache | KV size @ ctx | TTFT | ITL | output tok/s |
|---|---|---|---|---|---|---|
| 4096 | upstream vLLM 0.25.1 | fp16 | 16.0 MiB | 88.9 ms | 3.10 ms | 172.9 |
| 4096 | upstream vLLM 0.25.1 | `turboquant_3bit_nc` | 3.4 MiB | **74.3 ms** | 7.03 ms | 109.5 |
| 4096 | thunder (pinned vLLM) | fp16, control | 16.0 MiB | 99.0 ms | 3.04 ms | 165.6 |
| 4096 | thunder (pinned vLLM) | packed k4v4 | 4.1 MiB | 1.67-2.81 s | 50-87 ms | **7.3-7.5** |
| 32768 | upstream vLLM 0.25.1 | fp16 | 128.0 MiB | 706.3 ms | 3.78 ms | 38.9 |
| 32768 | upstream vLLM 0.25.1 | `turboquant_3bit_nc` | 27.0 MiB | 1380.2 ms | 12.23 ms | 18.2 |
| 32768 | thunder (pinned vLLM) | fp16, control | 128.0 MiB | 724.1 ms | 3.60 ms | 38.3 |
| 32768 | thunder (pinned vLLM) | packed k4v4 | 33.0 MiB | — | — | *blocked* |

Reading the table:

- **The fp16 rows are the control**, and they agree across the two stacks within
  a few percent (3.04 vs 3.10 ms ITL at 4k, 3.60 vs 3.78 ms at 32k), so the two
  environments are comparable.
- **Upstream's compression buys memory, not speed at batch 1**: 2.3x (4k) and
  3.2x (32k) more inter-token latency than its own fp16 KV. TTFT improves at 4k
  (74.3 vs 88.9 ms) and degrades 2x at 32k.
- **Thunder is behind on serving time, and the kernel is not why.** At 4k the
  served step is 50-87 ms while 36 attention launches at that shape measure
  ~2.8 ms per token, so the per-token cost is the host path (launcher, gather,
  merge), not the GPU. That gap, not the kernel table, is what the serving work
  is for — which is also why the two tables are quoted separately.
- 32k has no serving row: it cannot be served under CUDA graphs yet
  (*Known issues*), and the eager path is host-bound.

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

At the kernel level the remaining distance is the schedule, not the compression:
FlashAttention-4 does the same 16k decode shape in 0.196 ms per layer — 7.0 ms
per 36-layer step, reading *dense fp16* — still about 4x faster than this kernel
while moving 3.9x more bytes. Full method, caveats and raw rows:
`benchmarks/results/upstream_vs_ours.md`.

## Correctness

The kernel is checked against a dequantized-fp16 oracle at `atol=rtol=1e-2`
(`tests/test_correctness.py`): prefill 128x2048, decode 1x8192, several bit
widths, and split-K decode at S=2/4 with and without GQA packing. On B200 the
suite is green except `test_cuda_graph_replay_parity`, which is an unwired stub.
The CPU suite is `python -m pytest tests/`.

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

## How it works

**The cache is 3.9x smaller and attention reads it as-is.** A token's KV slot is
64 B of packed K codes + 64 B of packed V codes + two fp16 norms = 132 B at
4-bit, against 512 B for fp16 (`k_bits=3` gives 116 B, 4.4x). Nothing is expanded
into HBM: the codes travel from the cache to SMEM to the tensor cores, so a
decode step streams a quarter of the bytes.

**Dequantization is fused into the tile pipeline, and it is cheap.** Per KV tile
the kernel loads packed bytes, unpacks nibbles and gathers through a per-layer
LUT into an SMEM fp16 tile, then feeds the QK and PV MMAs. Measured by ablation
at the batch-16/32k shape, that stage is 26% of decode time (18% of it the LUT
gather) and the packed KV load is a further 22% — together still less than the
3.9x traffic they save.

**The MMAs never see quantized operands.** Dequantized fp16 tiles feed `mma.sync`
with fp32 accumulators, so the arithmetic is an ordinary fp16 attention kernel;
the compression changes the traffic, not the math. That is why the kernel can
match a dequantized-fp16 oracle to 1e-2 while reading 3.9x fewer bytes.

**Batch-1 decode fills the machine with split-K.** One request's decode grid is
8 CTAs (one per KV head) on 148 SMs, so the KV range is split; the count comes
from the grid, not a constant — 64 at 32k context for batch 1, capped at 16 from
batch 16 up, and floored to powers of two so the number of distinct captured
launches stays small. The partials are reduced by an online-softmax rescale in
one fused kernel.

**A GQA group shares one reconstruction.** With 32 Q heads over 8 KV heads the
QK and PV math differ per head but the KV reconstruction does not, so the
GQA-packed schedule reconstructs each tile once for the four heads that consume
it. It is the decode default (`THUNDER_GQA_PACK=0` opts out).

**The KV is traversed once.** Single-pass online softmax removes the separate
row-max pass — otherwise every tile loads and dequantizes K twice — and
register-local accumulator rescale removes two SMEM round-trips per tile.
Causal tile skipping drops fully-masked tiles in prefill. All three are on by
default.

**The packed load does not wait on itself.** Every global load in a tile is
issued before any store, so the latencies overlap instead of forming one
dependent chain per byte.

**Host overhead is flat.** The fast-launch cache keys the compiled CuTeDSL
function by kernel config instead of re-tracing MLIR per call: a launch costs
~0.35 ms of host work rather than ~407 ms, which is what makes eager usable and
graph capture cheap.

Where this is *not* fast yet: FlashAttention-4 still beats this kernel on the
same shapes while moving 3.9x more bytes, and the gap is schedule-level, not
parameter-level. The MMA M tile is bound to the warp count
(`m_block == num_warps * 16`), so a one-row decode pays for the tile it is given,
and at the batched tile the CTA is a single warp, which leaves latency hiding to
occupancy rather than to instruction-level parallelism; the operand copies are
also universal 16-bit SMEM copies rather than `ldmatrix`, which this CuTeDSL
build does not expose. Closing it means deriving the pipeline from FA4
(async/TMA movement, UMMA/TMEM accumulators, device-side reduction) rather than
tuning this schedule further; the frozen FA4 tree to build on is vendored at
`thunder_vllm/attention/v2/fa4/`. The measured matrix
(`benchmarks/results/fa4_matrix_b200_full.md`) predates the current tile and
split policies, so re-run it before quoting a ratio.

## Known issues

- **16k serving faults in the kernel.** At ctx 16384 the engine launches a
  many-request prefill — vLLM chunks the prompt into 1024 requests of 16 query
  rows — and the kernel takes an illegal memory access. It is pre-existing (the
  pre-session tile configuration fails identically), and it is in the kernel
  rather than the gather or the metadata. `docs/FAILURE_MODES.md` 15 has the
  launch geometry and the evidence.
- **32k cannot be served under CUDA graphs.** The gather reservation that fits
  long context (the dense/CSR path, 3.0 GiB at 32k) is not capturable in this
  stack, and the layout that is capturable (request-major) would need a 30.7 GiB
  reservation. 32k runs eager only, and eager is host-bound. `docs/FAILURE_MODES.md`
  14.
- **Eager is host-bound**: ~134 ms per token at 4k on Qwen3-8B, dominated by the
  CuTeDSL launcher rather than the GPU, so eager numbers are not comparable to
  graph-captured ones.

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
