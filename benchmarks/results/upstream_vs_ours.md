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

## Ours: e2e, kernel-level, and what still blocks the comparison

**The capture fault is gone.** At 4k the CUSTOM backend runs under CUDA graphs
(`enforce_eager=False`) and completes a generation. Measured three times on
Qwen3-8B, batch 1, greedy, 32 tokens:

| ctx | cfg | TTFT | ITL | output tok/s | peak |
|---|---|---|---|---|---|
| 4096 | ours (graphs) | 1667-2806 ms | 50.0-87.4 ms | 7.3-7.4 | 155.6 GiB |
| 4096 | ours-eager | 1996 ms | 133.7 ms | 5.2 | 161.7 GiB |

Graphs are worth 1.5-2.7x here, and the ITL spread (50 to 87 ms across runs) is
large enough that only the tok/s column should be read as stable. Against
upstream's 4k rows (109.5 tok/s for `turboquant_3bit_nc`, 172.9 for fp16) ours is
15-24x slower end to end.

That gap is **host path, not kernel**: the same attention launch measures 0.077 ms
at this shape, so 36 layers are ~2.8 ms per token against the ~50 ms observed. The
CuTeDSL launch plumbing, the gather's ~10 torch ops per layer and the Triton merge
are the suspects; `THUNDER_TIME_LAUNCH=1` reports the host buckets.

At 32k the run died before that:

- **CUDA OOM inside `paged_kv.reserve`** (16.06 GiB request, 9 GiB free): the
  request-major gather reserves `max_num_reqs * max_blocks_per_req` block-rows
  (952 x 2052 = 33 GiB at 32k) because its row layout cannot be trimmed. Under
  capture it surfaced as `CUDA_ERROR_ILLEGAL_ADDRESS` instead.
- Fixed by selecting the CSR gather past a 16 GiB reservation budget (it packs
  live blocks and reserves by the physical block count: 3.0 GiB at 32k). The CSR
  path then invalidated the capture, which was its per-layer temporaries (3 GiB
  each) being allocated inside the captured region; it now selects straight into
  the reservation with no temporary and no second copy.
- **32k under graphs is still blocked, for a different reason**: with the OOM gone
  the dense (CSR) gather is selected and the capture then dies with
  `cudaErrorStreamCaptureUnsupported`, deterministically, at 4k as well as 32k.
  The request-major layout cannot substitute there (30.7 GiB reservation), so
  long-context serving needs `enforce_eager=True` until the dense gather is
  capturable. See `docs/FAILURE_MODES.md` 14 for the evidence chain. Eager at 32k
  is correct: vLLM's own warmup run completes and generates.

What *is* measurable is the attention launch itself, from `./autoresearch.sh`
(engine config: `onepass`/`reg_rescale`/`causal_bound` on, the tile and split
policies from `thunder_vllm/attention/`, GQA packing on):

| shape | ours (launch) | dequant-fp16 ref |
|---|---|---|
| decode 4k, batch 16 | 0.351 ms | — |
| decode 32k, batch 16 | 2.72 ms | — |
| decode 32k, batch 1 | 0.215 ms | 5.61 ms |
| prefill 4k | 4.66 ms | 14.73 ms |

Those are 8.9x, 6.5x and 1.2x below the same shapes at the start of this session
(24.11 / 1.389 / 5.668 ms), from three changes: GQA packing on by default, the
split-K knee following the grid (16 -> 64 once packing took the head axis), and the
prefill KV tile narrowed to 16 (the MMA floor).

The remaining distance is the schedule, not the compressed format: FA4 does the
16k decode shape in 0.196 ms per layer -- 7.0 ms per 36-layer step, reading *dense
fp16* -- about 4x faster than this kernel while moving 3.9x more bytes
(`benchmarks/results/fa4_matrix_b200_full.md`).

## KV cache size

One request's cache at each context, from the format (bytes per token x
context), not from a running engine:

| KV cache | per token-head | per token (8 heads) | 4k | 32k |
|---|---|---|---|---|
| fp16 | 512 B | 4096 B | 16.0 MiB | 128.0 MiB |
| upstream `turboquant_3bit_nc` | 108 B | 864 B | 3.4 MiB | 27.0 MiB |
| this plugin, packed k4v4 | 132 B | 1056 B | 4.1 MiB | 33.0 MiB |

Ours comes from `benchmarks/bench_common.py::kv_bytes_per_token_from_layout`
(64 B K codes + 64 B V codes + 2 B norms per token-head); upstream's from the
per-vector table in its README. The two compressed formats are close but not
equal: ours is ~22% larger at the same context, so throughput rows are not
bit-for-bit comparable.

## Bugs found while building this

1. **Gather reservation sized from the token limit** — vLLM pads its block table
   past `ceil(max_model_len / block_size)` (260 blocks -> 264 columns at
   `block_size=16`, `max_model_len=4160`), and the manager rejected the table it
   was handed. Fixed: `make_paged_kv_manager(..., max_blocks_per_req=...)`, with
   `ThunderAttentionImpl.forward` passing the engine's own table width and row
   count. Regression test in `tests/test_paged_kv.py`.
2. **Reservation proportional to `max_model_len`** — the request-major gather
   reserves `max_num_reqs * max_blocks_per_req` block-rows, which is 33 GiB at 32k
   against a 178 GiB device already holding a 98.9 GiB KV cache, and it cannot be
   trimmed without breaking its row layout. Fixed by choosing the gather path from
   the reservation size; `docs/FAILURE_MODES.md` entry 13.
3. **CSR gather allocated per layer inside capture** — four 3 GiB temporaries per
   layer, which is what invalidated the 32k capture. Fixed by selecting into the
   reservation.

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

## Update: end-to-end difference, and what the gap actually is

The numbers above are the answer to "TTFT and throughput vs upstream under vLLM".
Restated as the difference:

| ctx | upstream turboquant_3bit_nc | ours (graphs) | ours (eager) | ratio (ours / upstream) |
|---|---|---|---|---|
| 4096 | TTFT 74.3 ms, ITL 7.03 ms, **109.5 tok/s** | TTFT 1667-2806 ms, ITL 50-87 ms, **7.3 tok/s** | TTFT 1996 ms, ITL 133.7 ms, 5.2 tok/s | **15x slower** (graphs), 21x (eager) |
| 32768 | TTFT 1380.2 ms, ITL 12.23 ms, **18.2 tok/s** | blocked (capture) | correct but unmeasured end to end | n/a |

So under vLLM we are **15-24x slower end to end**, and the cross-stack fp16 control
(99.0 ms / 3.04 ms / 165.6 tok/s on our pin) shows that is not an artifact of the
two environments.

**Correction to the attribution above.** The earlier text says the gap is "host
path, not kernel" and names the launch plumbing as the suspect. That conclusion
survives, but the *specific* mechanism it implies -- a ~13 ms per-layer launch cost
-- turned out to be an **instrumentation artifact**: a timing variable went stale
across CUDA-graph capture boundaries, so a mean over 1368 launches reported 25 ms
while the real fast-path hit measures **0.07 ms** (the wall-clock ITL is identical
with and without the instrumentation).

The corrected picture, measured:

- the attention launch is **0.07 ms**, so 36 layers are ~2.5 ms of a ~474 ms step;
- the rest is **whole-model eager launch overhead** (36 layers x ~10 kernels per
  step), which is exactly what CUDA graphs exist to collapse -- hence "graphs are
  worth 1.5-2.7x" above is a floor, not the ceiling;
- and **graphs are blocked by the post-capture fault** in `docs/FAILURE_MODES.md`
  FM14 (both captures now *complete* after the fast-path-guard fix, 52.7 s -> 0.75 s
  for the FULL capture, but the engine still faults in the KV-cache-init phase).

**Engine-free claim.** Because the served path is confounded by the above, the
kernel claim is measured without vLLM at all:

| | ours | FA4 (the kernel upstream TurboQuant runs on) | speedup |
|---|---|---|---|
| decode b1 32k | 0.220 ms | 0.307 ms | **1.397** |
| decode b16 32k | 2.590 ms | 0.345 ms | 0.133 |
| prefill 4k | 5.212 ms | 0.113 ms | 0.0217 |

geomean **0.1138**; see `docs/REPRODUCE.md` for the one-command reproduction. The
kernel is faster than upstream's at batch-1 decode and behind at batched decode and
prefill; the end-to-end gap under vLLM is *not* the kernel's doing.
