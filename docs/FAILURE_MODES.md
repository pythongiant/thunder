# Failure modes of the decision tree

Each gate in `docs/ROADMAP.md` can lie. This file lists how, with detection
and mitigation for each. Several entries are lived experience, not theory.

## 1. Contaminated measurement accepted as signal — LIVED

A killed worker leaves `VLLM::EngineCore` alive holding most of the GPU;
later runs start with little free memory and produce absurd timings
(216 s for an 8-token gen) that look like kernel pathology.

- Detection: check free memory before every run; distrust any cell with
  wild variance or superlinear blowups.
- Mitigation: `docs/RUNBOOK.md` hygiene is a hard gate, not advice.
  Every result carries its `[TQ-SYS]` fingerprint; medians are never
  reported without p20/p80.

## 2. Wrong-level comparison — LIVED

Comparing the 8-Q-head single-layer microbench against whole-model ITL
(32 heads × 36 layers + gather/rot/store) understates the kernel term by
roughly two orders of magnitude.

- Detection: ask of every comparison — same heads? same layer count?
  same path (gather/rot/store included)?
- Mitigation: `THUNDER_BENCH_HQ=32` engine-like shapes; stage-to-e2e
  reconciliation required before any claim.

## 3. Host time mistaken for GPU time — LIVED

`[STAGE]` buckets use `perf_counter` around async launches with no sync;
they measure dispatch, not execution. The `do_kv_cache_update` store path
sits outside them entirely.

- Detection: buckets summing far below e2e step time.
- Mitigation: documented host-only; GPU attribution via skip-flag A/B
  deltas (`THUNDER_SKIP_KERNEL/GATHER/ROT`), never via stage buckets.

## 4. Config aliasing in launch caches — LIVED

The fast-launch key once omitted `indirect` (and `gqa_pack`), so different
kernel specializations could share a key whenever shapes coincided.

- Detection: `THUNDER_DIAG` arm/hit/fallback counts — zero hits means
  the key is broken, not that the cache is cold.
- Mitigation: pure `_fast_key()` helper with unit tests covering every
  launch-time constexpr (`tests/test_fast_key.py`).

## 5. Correctness PASS but quality FAIL — EXPECTED

Kernel-vs-oracle parity does not imply model quality: boundary-value
quantization, missing QJL/norm-correction/sink handling, and error
accumulation across 36 layers are all invisible to it.

- Detection: only a quality eval (PPL/judge) detects this class.
- Mitigation: STAGE 9 quality gate is terminal — never ship on kernel
  parity alone. Upstream's quality features exist for exactly this reason.

## 6. Confounded A/B

Changing two variables at once (e.g. GQA + split-K from baseline) yields
an improvement with no attributable cause.

- Detection: ask which single-variable parent each combined cell has.
- Mitigation: the 2×2 (`baseline | GQA-only | SplitKV-only | GQA+SplitKV`)
  in `docs/ABLATION.md`; split-count tuning only after architecture choice.

## 7. Fork mis-resolution

Variant A ≈ dense-QK could mean "dequant is hidden" — or that the fork
shapes don't stress the real bottleneck (too small to expose dequant
cost), producing a false verdict for dequant-then-MMA.

- Detection: require tensor-active % alongside time; run fork shapes at
  decode-representative AND prefill-tile scales.
- Mitigation: fork runs on both shape classes before the rebuild decision;
  record raw numbers in matrix JSON, never prose conclusions alone.

## 8. Baseline drift

FA4, vLLM pins, torch/CUDA move under a frozen matrix; old wins rot
silently into stale comparisons.

- Detection: telemetry fingerprint mismatch across compared runs.
- Mitigation: re-run the matrix whenever pins change; never compare
  numbers with different `[TQ-SYS]` fingerprints.

## 9. Goodhart on the gate

Optimizing to `fa4_matrix` cells (tile sizes overfit to matrix shapes)
while e2e regresses — the table becomes the target instead of the measure.

- Detection: kernel-table wins without e2e movement.
- Mitigation: terminal acceptance is e2e + quality, never the kernel
  table alone.

## 10. Premature stage-skipping

Declaring a milestone done on one shape (e.g. 4k only), then building
the next stage on sand.

- Detection: exit criteria must span the shape set, not one point.
- Mitigation: each milestone's A/B runs the matrix subset for that
  stage (decode shapes for M1/M2, both for later stages).

## 11. No-GPU staleness

The tree stalls so long on GPU access that pins drift and harnesses bit-rot.

- Detection: CPU CI still green is necessary but not sufficient.
- Mitigation: on GPU day, re-verify imports first (`fa4_error` /
  `ours_error` columns exist for exactly this) before trusting any number.

## 12. Success theater

Ratios without absolutes (`ours%` with no tok/s), or TTFT-led reporting
on a decode story — numbers that persuade without informing.

- Detection: any table missing absolutes, ITL, or variance.
- Mitigation: mandated table format (absolutes + ITL-led + p20/p80);
  capacity/batch-scaling reported alongside B=1.

## 13. A path whose reservation scales with the model length — LIVED

The gather had two paths and the default one reserved by `max_num_reqs *
max_blocks_per_req` block-rows because it keeps request identity in the row stride
(`row = req * max_blocks_per_req + block`) and therefore cannot be trimmed without
changing the layout. That is a worst case proportional to the context: 4.2 GiB at
4k, 33 GiB at 32k, against a 178 GiB device already holding a 98.9 GiB KV cache.
The 32k e2e died inside `reserve()` with CUDA OOM, and under CUDA-graph capture it
surfaced as `CUDA_ERROR_ILLEGAL_ADDRESS` instead — which made it look like a graph
bug and cost a session's worth of bisecting the wrong thing.

- Detection: read the *allocation* traceback, not the error class. `OutOfMemoryError`
  and `ILLEGAL_ADDRESS` at the same frame mean the fault is the allocator, not the
  kernel. Also: any buffer sized from `max_model_len` is a suspect at long context.
- Mitigation: the CSR path packs live blocks densely and reserves by the physical
  block count, so the path is now chosen from the reservation size
  (`_use_indirect_gather`, 16 GiB budget: 4.2 GiB at 4k and 8.2 GiB at 8k stay on
  request-major, 33 GiB at 32k switches) rather than from a flag default; the
  explicit `THUNDER_8B_INDIRECT` override still wins, so the OOM stays reproducible.
  The budget is a constant and `reserve()` refuses to grow after the first
  allocation, because the two paths share one reservation: a path that flipped
  with memory pressure would make the other path write past the buffer.

## 14. A gather path that cannot be captured — LIVED

The dense (CSR/indirect) gather packs the live blocks and reserves by the physical
block count, which is the only layout that fits a long context: the request-major
layout reserves `max_num_reqs * max_blocks_per_req` block-rows (30.7 GiB at 32k on
Qwen3-8B) because its row stride is the engine's table width and cannot be
trimmed. But that dense path cannot be captured in this stack: vLLM's cudagraph
memory profiling dies with `cudaErrorStreamCaptureUnsupported`, deterministically,
in both of its forms (select-into-the-reservation and temporaries + `copy_`) --
after its allocations were removed (the build is fully temporary-free, the gather
selects into the reservation) and with no sync left anywhere in the path. Ablating
the build body alone, or the gather alone, does not help; ablating the kernel
launch and the merge (`THUNDER_SKIP_KERNEL=1`) does not either. `THUNDER_SKIP_BACKEND=1`
makes the capture succeed, so the fault is in this backend and in the dense path.

- Detection: `cudaErrorStreamCaptureUnsupported` (not `...Invalidated`) raised at
  `profile_cudagraph_memory`, at 4k as well as 32k, i.e. it is the path and not the
  context length. Two identical runs fail identically, so it is not a race.
- Mitigation: the path is chosen by reservation size with a 24 GiB budget, so 4k,
  8k and 16k (15.5 GiB) stay on the capturable request-major path and only 32k
  attempts the dense one. Long-context serving must run with `enforce_eager=True`
  until the dense gather is capturable -- the eager path is correct (vLLM's own
  warmup run completes at 32k). Do not "fix" this by lowering the budget: that
  trades a working context length for a startup failure.

## 15. An illegal address at a many-request prefill — OPEN

At ctx 16384 the plugin faults with `cudaErrorIllegalAddress`, in eager and in graph
mode, reproducibly (3 of 3 engine runs). The error surfaces at the first
synchronising op *after* the prefill (`torch.equal` inside `HadamardRotation.__init__`,
reached from `do_kv_cache_update`), which is why the traceback points at the
quantizer instead of at the fault.

What the engine actually launches there, from `THUNDER_DEBUG_LAUNCH=1`:

```
q=(16384, 32, 128) fp16   n=16384
kv=(243585, 8, 16, 128)   bt=(1024, 1032)   sl=(1024,)   qsl=(1025,)
num_reqs=1024   max_blocks_per_req=1032   is_prefill=True   max_query_len=16
```

So vLLM chunks the 16k prompt into **1024 requests of 16 query rows each** -- a
multi-request prefill. Nothing in the loop covered that: the only prefill shape was
batch 1 at 4k, and the 4k path is fine.

- `THUNDER_SKIP_BACKEND=1` runs (17.3 tok/s) and `THUNDER_SKIP_KERNEL=1` runs
  (9.3 tok/s), so the fault is in the **kernel launch**, not the gather or the
  metadata.
- The grid reproduces it with the engine's geometry: a new `prefill-b1024-16k`
  shape (batch 1024, seqlen_q 16, seqlen_k 16384) faults with the same error --
  **but not every time**: a four-cell batch sweep (2/64/256/1024) passed all four,
  while the same cell alone failed twice. Treat a single grid cell at this geometry
  as flaky, and prefer repeated runs before believing a pass.
- Tile shape is not the trigger: n=16/32/64 and m=128/t=256 all fault alike, so the
  prefill KV-tile change this session is not the cause.

- Next: bisect inside the kernel for the many-request prefill (the row base
  `req * kv_row_stride` with `num_reqs` = 1024 and a 16-row query block is the prime
  suspect, e.g. an index or a grid bound that assumes one query row per request).
