# thunder

A quantized-attention architecture for Blackwell GPUs. TurboQuant-compressed
KV cache — 3/4-bit codes plus per-head norms — flows through a fused CuTeDSL
attention kernel straight into the tensor cores. No fp16 KV is ever
materialized.

The vLLM integration is the serving vehicle and end-to-end harness, not the
product: it lets the kernel run inside a real serving stack so kernel
throughput can be validated against serving throughput.

## How it works

A KV token is stored as a rotation, a handful of Lloyd-Max code indices, and a
per-head norm. Attention consumes those codes directly: each tile is unpacked,
looked up through the codebook, and fed to the QK and PV MMAs, with the
rotation folded into the output projection. A 4:1 GQA group therefore reads one
compressed KV tile instead of four fp16 ones, and the bytes that move shrink by
roughly the compression ratio.

The pipeline lives in four places:

- **Representation** (`thunder_vllm/quant/`) — rotation, Lloyd-Max codebooks,
  bit packing, and per-head norms. This is the differentiating data path and it
  is fixed; the work is in how it is executed, not in how it is encoded.
- **Attention kernel** (`thunder_vllm/attention/cute_kernel.py`) — a fused
  CuTeDSL forward pass. Per KV tile: packed codes are loaded, unpacked, and
  dequantized through a per-layer LUT into an SMEM code ring, then consumed
  directly by the QK and PV MMAs with online softmax. Fast paths default on:
  single-pass online softmax, register-local accumulator rescale, and causal
  tile skipping. Decode uses split-K to fill the machine at batch 1.
- **KV store** (`thunder_vllm/attention/cache_layout.py`) — a Triton kernel
  that performs rotation, exact fp32 quantization, and bit packing inside the
  cache write, including the non-contiguous value views the engine hands out.
- **Paged addressing** (`thunder_vllm/attention/paged_kv.py`) — a request-major
  gather and a CSR/indirect gather whose per-step metadata is built once and
  shared across layers, with a capture-safe device build for CUDA graphs.

Serving integration registers as a `CUSTOM` attention backend. The engine's
native cache-write path conflicts with the packed byte cache, so a separate
KV-cache-update hook owns the write, and a config-keyed fast-launch cache keeps
per-launch host overhead flat.

## The v2 pipeline

The `mma.sync` schedule is the measurement baseline and the end-to-end vehicle.
The target execution architecture is derived from FlashAttention-4: the same
Blackwell pipeline shape — KV-head/GQA tile ownership, async/TMA movement,
UMMA/TMEM accumulators, overlapped softmax, SplitKV scheduling, device-side
reduction, direct paged-KV consumption — with the TurboQuant packed data path
as the one new producer stage.

The frozen FA4 tree lives at `thunder_vllm/attention/v2/fa4/` (vendor of
flash-attn-4, import-rewritten and pinned). The plan is to prove the vendored
pipeline reproduces dense FA4 first, then swap its KV load for the packed
load/dequant producer, so the only new code is the data path and everything
downstream — descriptors, MMA, softmax, scheduler — is known-good.

## Installation

```sh
pip install -e .
```

Requirements: a Blackwell GPU (sm_100/sm_110), CUDA 12.8+, PyTorch 2.8+,
`nvidia-cutlass-dsl`, and Triton. The end-to-end harness additionally needs a
vLLM build with the pinned commit (see `docs/RUNBOOK.md`).

## Usage

```python
from thunder_vllm.model import registry

registry.configure(k_bits=4, v_bits=4)
registry.register()

# Then select the backend at engine construction:
#   attention_config={"backend": "CUSTOM"}
```

Kernel-level checks and A/Bs:

```sh
PYTHONPATH=. python3 ci_probe/correctness_check.py     # store, gather, kernel vs oracle
PYTHONPATH=. python3 ci_probe/kernel_bench.py          # kernel A/B
PYTHONPATH=. python3 ci_probe/kernel_stage_probe.py    # per-stage decomposition
python -m benchmarks.fa4_matrix --quick                 # FA4 vs ours, frozen gate
```

End-to-end:

```sh
python -m benchmarks.bench_vs_thunder_vllm --model Qwen/Qwen3-8B
PYTHONPATH=. python3 ci_probe/studio_batched.py "1:4096,16:4096" "fp16,ours-eager"
```

## Repository layout

```text
thunder_vllm/
  attention/      fused CuTeDSL kernel, paged KV, metadata, split-K, backends
  attention/v2/   FA4-derived execution skeleton (vendored) + TurboQuant producer
  quant/          rotation, Lloyd-Max codebooks, packing
  model/          vLLM registry and integration
  utils/          logging, telemetry
benchmarks/        FA4 matrix, end-to-end harness, shared metrics
ci_probe/          correctness checks, kernel/stage probes, GPU probes
docs/              roadmap, pipeline v2, ablation, benchmarks, runbook
tests/             CPU test suite
```
## License

Apache-2.0. The vendored FlashAttention-4 tree under
`thunder_vllm/attention/v2/fa4/` is BSD-1-Clause, © Dao-AILab.
