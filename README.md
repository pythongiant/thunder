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

### Against `turboquant-vllm` (upstream TurboQuant KV)

Upstream's GQA/MHA KV path is **stock vLLM** since
[vllm#38479](https://github.com/vllm-project/vllm/pull/38479): vLLM 0.25.1 with
`--kv-cache-dtype turboquant_3bit_nc`, no plugin. Measured on that stack:

| ctx | KV cache | TTFT | ITL | output tok/s |
|---|---|---|---|---|
| 4096 | fp16 | 88.9 ms | 3.10 ms | 172.9 |
| 4096 | `turboquant_3bit_nc` | **74.3 ms** | 7.03 ms | 109.5 |
| 32768 | fp16 | 706.3 ms | 3.78 ms | 38.9 |
| 32768 | `turboquant_3bit_nc` | 1380.2 ms | 12.23 ms | 18.2 |

At batch 1 the compression buys memory, not speed: upstream's compressed path
costs **2.3x** (4k) and **3.2x** (32k) more inter-token latency than its own fp16
KV, while TTFT improves at 4k and degrades 2x at 32k.

The same workload on this plugin's pin with fp16 KV — the control that shows the
two stacks are comparable — gives TTFT 99.0 ms / ITL 3.04 ms / 165.6 tok/s at
4k, and 724.1 ms / 3.60 ms / 38.3 tok/s at 32k, within a few percent of
upstream's fp16 rows. Full method, caveats and raw rows:
`benchmarks/results/upstream_vs_ours.md`.

### This kernel

One attention launch, in the configuration the engine actually launches
(single-pass online softmax, register-local rescale, causal bound, engine
split-K policy = 4), `k_bits=4` / `v_bits=4`, CUDA-graph medians:

| workload | ours | dequant-fp16 reference |
|---|---|---|
| decode, 4k context | **0.273 ms** | 2.498 ms |
| decode, 32k context | **1.935 ms** | 5.602 ms |
| prefill, 4k | **7.304 ms** | 14.769 ms |

The reference column dequantizes the same cache to fp16 and runs SDPA on it: it
is a kernel-level sanity bound, never the competition. Against it this kernel is
9.1x faster at 4k decode, 2.9x at 32k decode and 2.0x at prefill. Marginal
shares of decode time, by ablation: MMAs 38%, packed-KV load 22%, K+V dequant
19%, split-K merge 1.5%. Run-to-run noise on these medians is about ±1%.

**End-to-end we cannot publish a measured column yet** — see *Known issues*.
Derived from the launch above, attention alone costs ~9.8 ms/token at 4k
(~102 tok/s) and ~69.7 ms/token at 32k (~14 tok/s) over 36 layers: roughly at
parity with upstream's compressed path at 4k and ~25% behind at 32k, while
prefill is far behind (~263 ms vs 74.3 ms TTFT at 4k). Those figures exclude
weight GEMMs, sampling and engine overhead, so they are a best case, not a
measurement.

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
32 CTAs on 148 SMs, so the KV range is split: 4 ways gives 128 CTAs and measures
**3.5x** (4k) and **3.9x** (32k) over the unsplit schedule. The split partials
are reduced by an online-softmax rescale, so the extra CTAs cost one small merge
(1.5% of decode at 32k, 10% at 4k).

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
3.3x off at 4k decode and about 6x at 16k
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
