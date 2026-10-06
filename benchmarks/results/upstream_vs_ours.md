# Upstream TurboQuant KV vs this plugin

Two stacks, because they cannot share one environment:

| | upstream baseline | ours |
|---|---|---|
| vLLM | **0.25.1** (stock) | pinned `0.29.1rc1.dev159+gdffbb714e` |
| KV path | `--kv-cache-dtype turboquant_3bit_nc` (upstreamed in vllm-project/vllm#38479; no plugin needed) | `attention_config={"backend": "CUSTOM"}`, this plugin |
| runner | `ci_probe/modal_upstream_baseline.py` | `ci/modal_app.py --mode e2e` |
| GPU | B200 | B200 |

Workload (identical on both sides): `Qwen/Qwen3-8B`, batch 1, prompt `"token" * ctx`,
greedy, 32 generated tokens, `dtype=float16`, TTFT from a `max_tokens=1` pass.

## Measured: upstream baseline (vLLM 0.25.1, stock)

| ctx | kv cache | TTFT | ITL | output tok/s |
|---|---|---|---|---|
| 4096 | fp16 | 88.9 ms | 3.10 ms | 172.9 |
| 4096 | `turboquant_3bit_nc` | **74.3 ms** | **7.03 ms** | 109.5 |
| 32768 | fp16 | 706.3 ms | 3.78 ms | 38.9 |
| 32768 | `turboquant_3bit_nc` | 1380.2 ms | **12.23 ms** | 18.2 |

`k3v4_nc` (the bit-matched dtype) is **not accepted** by vLLM 0.25.1
(`ValidationError: 1 validation error for CacheConfig`), so `turboquant_3bit_nc`
is the closest upstream row available.

**Upstream's compressed-KV path is a memory play, not a speed play at B=1.** It
costs 2.3x (4k) to 3.2x (32k) more inter-token latency than fp16 KV. It does help
prefill at 4k (74.3 vs 88.9 ms TTFT) and hurts it at 32k (1380 vs 706 ms).

## Cross-stack control

fp16 KV on *our* pin, same workload, measures 99.0 ms TTFT / 3.04 ms ITL /
165.6 tok/s (4k) and 724.1 ms / 3.60 ms / 38.3 tok/s (32k) — within a few percent
of upstream's fp16 rows, so the two stacks are comparable and the rows above are
not an artifact of one environment. (fp16 control spread across three runs:
3.04-3.87 ms ITL at 4k, 3.60-4.41 ms at 32k.)

## Ours: kernel-level, and why there is no e2e row

The plugin currently has **no meaningful end-to-end number** on this workload:

- **CUDA-graph capture of the CUSTOM backend faults**:
  `torch.AcceleratorError: CUDA error: operation failed due to a previous error
  during capture` (`cudaErrorStreamCaptureInvalidated`), reproduced at ctx 4096
  and 32768. This is the capture fault the repo's own notes track; it blocks the
  path vLLM actually serves with.
- **Eager is host-bound**: 4724 ms TTFT / **515.6 ms ITL** / 1.5 tok/s at 4k —
  the CuTeDSL launcher dominates, matching `ci_probe/results/tracka_notes.md`
  ("eager decode is ~1 s/forward; only graph-captured numbers are meaningful").

What is measurable is the attention launch itself, from `./.auto/measure.sh`
(engine config: `onepass`/`reg_rescale` on, engine split-K policy = 4, B=1,
32 Q heads / 8 KV heads / head_dim 128):

| shape | ours (launch) | dequant-fp16 ref |
|---|---|---|
| decode 4k | 0.2732 ms | 2.4984 ms |
| decode 32k | 1.9351 ms | 5.6024 ms |
| prefill 4k | 7.3042 ms | 14.7688 ms |

Deriving e2e from that (36 layers, attention as the only cost — an upper bound on
our throughput, not a measurement):

| ctx | ours (derived) | upstream tq3 | upstream fp16 |
|---|---|---|---|
| 4096 | ~9.8 ms ITL, ~102 tok/s | 7.03 ms, 109.5 tok/s | 3.10 ms, 172.9 tok/s |
| 32768 | ~69.7 ms ITL, ~14 tok/s | 12.23 ms, 18.2 tok/s | 3.78 ms, 38.9 tok/s |

So on decode the kernel is roughly **at parity with upstream's compressed-KV path
at 4k and ~25% behind it at 32k** — despite being 3.3-6x off FA4 on the same
shapes (`benchmarks/results/fa4_matrix_b200_full.md`), because upstream's own
compressed path is itself ~3x off its fp16 baseline. Prefill is the reverse: our
prefill launch (7.30 ms) times 36 layers exceeds upstream's *entire* 4k TTFT.

## Bugs found while building this

1. **Gather reservation sized from the token limit** — vLLM pads its block table
   past `ceil(max_model_len / block_size)` (260 blocks -> 264 columns at
   `block_size=16`, `max_model_len=4160`), and the manager rejected the table it
   was handed. Fixed: `make_paged_kv_manager(..., max_blocks_per_req=...)`, with
   `ThunderAttentionImpl.forward` passing the engine's own table width and row
   count. Regression test in `tests/test_paged_kv.py`.
2. **Graph-capture fault** (above) — reproduced, not fixed. Until it is, this
   plugin cannot produce an e2e number under `enforce_eager=False`, and every
   e2e comparison against upstream is blocked.

## Caveats

- Different vLLM pins; the plugin cannot load on 0.25.1 (it needs the
  `AttentionMetadataBuilder`/`CUSTOM` surface of main), so the two sides cannot be
  measured in one process.
- Upstream row is 3-bit K/V + norm correction; ours is 3-bit K / 4-bit V
  (116 B per token-head vs their 108 B).
- Our rows are kernel-level; the derived e2e figures assume attention is the only
  per-layer cost and are labelled as such.
- Numbers are single-run medians of one generation; the fp16 control spread is
  shown above so the reader can judge.
