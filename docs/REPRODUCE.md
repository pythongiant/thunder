# Reproducing the kernel benchmark (no engine in the loop)

This file documents the one claim this repository stands behind today, and how to
reproduce it in one command. Read the *What is NOT claimed* section before quoting
any number.

## The claim

**Our TurboQuant attention kernel is faster than the kernel upstream TurboQuant
runs on** (FA4's SM100 forward, `flash_attn.cute.flash_attn_func`) at batch-1
decode with a 32k context: **1.40x** on identical inputs with identical timing.

Across the five engine-shaped cells the geometric mean is **0.114x**, i.e. we are
*behind overall*. The win is confined to batch-1 decode; see the table below.

| cell | ours | FA4 | speedup |
|---|---|---|---|
| decode b1 32k | 0.220 ms | 0.307 ms | **1.397** |
| decode b16 4k | 0.335 ms | 0.072 ms | 0.213 |
| decode b16 32k | 2.590 ms | 0.345 ms | 0.133 |
| prefill 4k | 5.212 ms | 0.113 ms | 0.0217 |
| prefill 16k | 70.43 ms | 1.621 ms | 0.0230 |
| **geomean** | | | **0.1138** |

Both sides run the same `q`/`k`/`v` tensors, the same CUDA-event methodology
(25 warmup + 50 timed reps, median), on Qwen3-8B's geometry: head_dim 128,
32 Q heads / 8 KV heads (GQA 4:1), causal. Our side uses the **shipped**
configuration: bit widths K=3/V=4, the tile/split/GQA-packing policies from
`thunder_vllm/attention/tile_shape.py` and `splits.py`, and the kernel's fast
flags (`onepass`, `reg_rescale`, `causal_bound`).

## One command

```bash
bash autoresearch.sh
```

Prerequisites: a [Modal](https://modal.com) account with GPU access, and a local
`python3 -m compileall`-clean tree. The script builds the benchmark container
itself (CUDA 12.8, vLLM deps, the in-repo vendored FA4) and runs on a B200.

It prints one `METRIC` line per cell plus the aggregate:

```
METRIC speedup_vs_fa4=0.113811
METRIC grid_decode_b16_32k_policy_policy_ms=2.589056
METRIC grid_decode_b16_32k_policy_policy_fa4_ms=0.337516
METRIC grid_decode_b16_32k_policy_policy_speedup_fa4=0.130362
...
```

Under the hood that is
`modal run ci/modal_app.py --mode grid --grid "<cells>"`; cells are
`<shape>|<splits>|<gqa>|<m>|<threads>|<n>|<env K=V,...>` where an empty field or
`policy` means "the engine's own policy" (which is what makes the measurement the
shipped configuration rather than a strawman). Shapes come from
`ci/modal_bench_smoke.py::SHAPES`; `run_smoke` times FA4 on the same tensors.

## What is NOT claimed

- **No engine.** This is a kernel-vs-kernel measurement. Nothing here is served
  throughput or TTFT.
- **No vLLM/SGLang/TensorRT-LLM integration claim.** The kernel is wired into
  vLLM only; the served path has open problems (see below).
- **Not "we beat upstream".** We beat upstream's *kernel* at one shape (batch-1
  decode) and are behind at batched decode and prefill.
- **No parity claim against the naive reference.** `attention_ref` (dequantize +
  fp32 einsum) is a sanity lower bound only, per `docs/BENCHMARKS.md`; the FA4
  comparison above is the one that matters.

## Why the gaps split the way they do (measured, not inferred)

Running the *same kernel* with the other layout (K=4/V=4, i.e. nibble-aligned
instead of 3-bit-packed) moves the two regimes in opposite directions:

| cell | K=3/V=4 (shipped) | K=4/V=4 |
|---|---|---|
| decode b16 32k | **2.590 ms** | 2.719 ms |
| prefill 4k | 5.212 ms | **4.644 ms** |

3-bit adds unpack instructions and removes bytes, so **prefill is
instruction-bound and decode is traffic-sensitive**. That is also why our
compressed representation wins where bandwidth dominates (batch-1 decode) and
loses where per-element unpack dominates (prefill). Refuted levers and the
remaining ones (KV-stationary prefill, async multi-stage KV movement, a
tensor-core-native 4-bit format) are recorded in
`docs/PIPELINE_V2.md` and `thunder_vllm/attention/tile_shape.py`.

## What an engine adapter must provide (the "backbone" contract)

The kernel is engine-agnostic; it takes a gathered KV buffer and per-step
metadata, and it is launched through one entry point:

- `launch_thunder_attention(kernel, q, gathered, out, attn_metadata, softmax_scale, ...)`
  in `thunder_vllm/attention/cute_kernel.py`.
- **`gathered`**: request-major packed KV — `k_packed` `(page_rows, block_size, Hk, k_packed_bytes)`,
  `v_packed` (same with `v_packed_bytes`), plus `k_norm`/`v_norm` `(page_rows, block_size, Hk)`.
  Produced by `thunder_vllm/attention/paged_kv.py` from a block table + packed cache.
- **`attn_metadata`**: `seq_lens`, `query_start_loc`, `block_table`,
  `max_blocks_per_req`, `max_query_len`, `num_reqs`.
- **Split-K** writes partial `(m, l, o)` per (request, head, split) and the merge
  is `_merge_splits` (Triton) in the same module.
- The **producer** (rotation + Lloyd-Max + packing) and the cache layout are in
  `thunder_vllm/quant/` and `thunder_vllm/attention/cache_layout.py`; the store
  hook is `do_kv_cache_update`.

An adapter for another engine needs: its block table and per-step lengths, a
gathered buffer of that layout, and a stream-ordered launch. Everything else is
in this repository.

## Known open problems (do not mistake for done)

1. **The served path is not solved.** Our plugin's per-layer host overhead makes
   eager serving far slower than upstream; CUDA graphs are the intended fix and
   the post-capture fault documented in `docs/FAILURE_MODES.md` (FM14) blocks
   them.
2. **Prefill is ~40x behind FA4** and needs a structural change (the dequant is
   re-run per q-block), not tuning.
3. **No SGLang or TensorRT-LLM adapter exists.** TensorRT-LLM has no TurboQuant
   implementation at all (its KV quantization is FP8/FP4/INT4/INT8), so a
   "TurboQuant vs TurboQuant" comparison there would require porting both sides.
