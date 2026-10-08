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
- **ROOT CAUSE (measured live, this session): we could not tell we were being
  captured.** `torch.cuda.is_current_stream_capturing()` returns **False** during
  vLLM's capture — vLLM captures on a side stream, and our forward does not run on
  it. Caught with `ci_probe/probe_capture_watch.py`, which starts the engine itself
  and prints its log live (the e2e worker buffers it, so a hang shows nothing):
  every `[TQ-PATH]` line printed `capturing=0` while vLLM's progress bar read
  `Capturing CUDA graphs (FULL)`. So every capture-safe branch in this backend was
  dead code, and the capture ran the EAGER paths: `_scales_for` re-derived (and
  re-allocated) the norms buffer for vLLM's 1-block profiling cache *inside* the
  captured region, `_decode_split_count` could fall back to a host read, and the
  dense-gather refusal never fired. The same run shows PIECEWISE capturing fine
  (1/1) and FULL dying — and it dies with a real `cudaErrorIllegalAddress`, an
  out-of-bounds read, not a capture-API error, which is what the store's syncs had
  been masking. Fixed by an in-band signal: `build_for_cudagraph_capture` marks the
  metadata (`is_capture`), and every capture-sensitive branch keys on
  `_in_capture(attn_metadata)` instead of the stream query.
  Corrections to what this file claimed earlier: the captured launch DOES split
  (`_decode_split_count` uses vLLM's CPU seq-len mirror when it is present, so it is
  not 1 during capture — measured S=16 at the 4k geometry), so the Triton merge IS
  inside the captured region and the `THUNDER_SPLITS=1` row above is a different
  configuration, not a control; and with `cudagraph_num_of_warmups = 0` (vLLM's
  default) the capture is the FIRST execution of its geometry, so both the compile
  and the split-partial allocation happen inside it. The warm-up now aims at exactly
  that geometry (an ALL-ZERO metadata at each configured capture size, every split
  count the policy can pick) instead of the engine capacity.
- **Superseded hypothesis: the store's host syncs.**
  `reshape_and_cache_ref` -- the path 3-bit K uses, i.e. the engine's default --
  decides PAD_SLOT_ID ON THE HOST (`if not bool(keep.any())`, plus the `t[mask]`
  gathers next to it: two D2H syncs), and the store runs INSIDE the captured
  region because vLLM's KV-update hook is part of the layer's forward. A sync
  during capture is exactly `cudaErrorStreamCaptureUnsupported` ->
  `...Invalidated`. Nothing else in the forward touches device data on the host
  (the `seq_lens_cpu` mirror, the config flags and `is_current_stream_capturing()`
  are all host-side; the quantizer has no syncs). The store now scatters through
  `_scatter_codes_kernel`, which skips PAD_SLOT_ID in-kernel like vLLM's own cache
  kernels, so the 3-bit path has no host branch at all; the torch quantizer is
  unchanged and the parity probe (`ci_probe/modal_probe_store_parity.py`, 4/4,
  3/4, and both with PAD_SLOT_ID) reports `maxdiff == 0.0` on codes and norms.
  Verified end to end once the spend limit lifted: the parity probe still reports
  `maxdiff 0.0` in all four cases, the GPU suite is green (1 known stub failure,
  0 unexpected), and 16k eager serving measures ITL 320 ms / 3.1 tok/s against
  334 ms / 3.0 before the change -- so the scatter is invisible in serving and
  only removes the host syncs.
  Effect on capture (measured, this session): the failure MODE changes and the
  store is exonerated. Three 4k `ours` configurations, all `k=3/v=4`:

  | store scatter | split-K merge in the captured region | outcome |
  |---|---|---|
  | torch (syncs on `slot_mapping`) | yes | fails fast at `capture_model`, `cudaErrorStreamCaptureUnsupported` + `...Invalidated` |
  | Triton (no sync) | yes | runs long (a hang, cancelled at ~20 min) |
  | Triton (no sync), `THUNDER_SPLITS=1` | no | fails fast at init again |

  So removing the syncs lets the capture get further but does not fix it. **The
  remaining blocker is code-verified, and it is the shape-key problem of 15b.**
  The capture's kernel is not the step's kernel, in three independent ways:
  1. `_decode_split_count` returns 1 while capturing (its host read is illegal
     there), so the captured launch has NO split-K, while every eager decode step
     has S=16..64;
  2. `tile_shape(is_prefill=False, num_reqs)` picks `DECODE_TILE_BATCHED`
     (16 rows / 16-wide KV / 32 threads) at the engine's capacity >= 16 but
     `DECODE_TILE` (32/32/64) at batch 1 -- a DIFFERENT kernel object;
  3. vLLM's capture pads `num_reqs` to the capacity (`pad_attn`) while its own
     warm-up dummy for the same descriptor runs the batch's unpadded geometry.
  CuTeDSL's jit cache is keyed per (constexpr config, tensor shape, grid), so the
  captured launch is a cold compile -- and a cold split-partial allocation --
  inside the capture. Note that only FULL captures pad (`pad_attn = mode == FULL`),
  so PIECEWISE (prefill-shaped) captures run the same geometry their warm-up dummy
  did: the decode/FULL case is the one to warm. Its other launch scalars are fixed
  too -- `max_query_len = uniform_decode_query_len == 1` and `num_splits == 1`
  (point 1) -- which is what the capacity warm replicates. That fits every observation: eager-only failure before (the
  store's sync died first), a hang once the sync was removed (the compile blocks
  on the allocator), and `THUNDER_SPLITS=1` moving the symptom again (it changes
  which config is cold).
- **Correction (measured, not inferred): points 1 and 2 above are wrong, and the
  warm never ran at all.** Three separate errors, each caught by measurement:
  1. **The split count is NOT 1 under capture.** `_decode_split_count` only falls
     back to 1 when there is no host mirror; vLLM's capture metadata DOES carry
     `seq_lens_cpu_upper_bound` (= 4160 at this context), so the captured launch
     gets `choose_split_count(4160, tile_n=32, num_reqs=1)` = **16**. The captured
     decode is a split-K decode like any other.
  2. **The capture's geometry is the capture SIZE, not the capacity, and not the
     warm-up dummy's.** Its own launch, measured at `cudagraph_capture_sizes=[1]`:
     `q=(1, 32, 128)`, `sl=(1,)`, `qsl=(2,)`, `num_reqs=1`, `max_query_len=1`,
     `is_prefill=False`, `cap_reqs=1024`. `_fixed_rows` takes its view from the
     buffer's STORAGE, so the capture's q/o view is its own token count (1) while
     an eager step's is its own (16384 at the profile batch) -- different jit keys
     from the same code.
  3. **The warm raised `NameError` on every launch.** `launch_thunder_attention` is
     imported *inside* `forward`, so it is not a module global and `_capacity_warm`
     could not see it. The handler logged it and continued (`logger.exception`),
     which is why the captures stayed cold while the flag looked like it worked:
     every earlier negative result about warming is uninformative.
- Mitigation (experiment, OFF by default): `THUNDER_WARM_CAPACITY=1` issues eager
  launches of the capture geometries before vLLM captures them: each
  `cudagraph_capture_sizes` entry (PIECEWISE, `num_reqs == size`) and the capacity
  (FULL, `num_reqs == max_num_seqs` via `pad_attn`), each at every split count
  `1..64`, with the capture's own tile (`get_kernel` per `num_reqs`), GQA-packed,
  and -- the part that was missing -- q/o tensors of the CAPTURE's shape, allocated
  by the warm itself (a step's tensors yield a different `_fixed_rows` view). The
  metadata is all-zero (`tests/test_capacity_warm.py` pins that invariant: a stale
  entry would be an out-of-bounds read, and a zero entry makes `kv_len == 0` and
  `q_len == 0`, so the CTAs compile and allocate without touching a row). The warm
  uses its own q/o, so no save/restore of the step's output is needed, and a
  capacity-sized zeroed `indptr` copy covers the indirect path's `mIndptr[req]`
  indexing. `tests/test_capacity_warm.py` also pins that the warm REACHES the
  launcher, per capture size and at the capacity -- the NameError class of bug is
  invisible otherwise. Verify on the GPU with
  `THUNDER_WARM_CAPACITY=1 --e2e "4096|ours|3|4"`; if the capture then succeeds,
  the same launch belongs in the engine's warm-up path (not behind a flag).
- **The store's engine-level divergence is now REPRODUCED (this session).** The
  Triton `_scatter_codes` -- which the engine ran by default for 3-bit K -- fails
  `tests/test_cache_layout.py::test_triton_scatter_matches_reference_at_the_engine_geometry`
  (the engine's contract: 3-bit K, Hk=8, bs=16, PAD_SLOT_ID). The pre-existing parity case covered only the torch
  reference at 4-bit K, which takes the OTHER kernel, so the path the engine
  actually shipped had no coverage. (An `--mode e2e` run with the Triton store as
  the default also fails to initialize, but that config is `ours`, which captures,
  so it is NOT evidence about the eager path; the eager control is `ours-eager`.)
  The Triton scatter is therefore opt-in now
  (`THUNDER_STORE_TRITON=1`) and the torch reference is the default again: it
  syncs (fatal inside a capture) and it is correct. The capture path needs the
  sync-free store, and the capture path does not work yet either.
- **Final measured state (this session).** With `THUNDER_WARM_CAPACITY=1` BOTH
  captures now complete -- `PIECEWISE 1/1` in 13.5 s and, for the first time,
  `FULL 1/1` in 50.5 s (they previously hung or died in `capture_model`). The warm
  launches the capture sizes at both the policy's split count and 1, and the
  capacity at 1 only: warming the capacity at 16 hangs the FULL capture, whose
  padded batch has no CPU seq-len mirror and therefore runs S=1. What still kills
  the engine is an IMA *after* both captures (`XID 31 ... ACCESS_TYPE_VIRT_READ`),
  in the KV-cache-init phase, with a traceback that surfaces at the store's
  `reshape_and_cache` -- a reporting site, not a fault site. The store is the open
  item: for 3-bit K (the engine default) `do_kv_cache_update` routes to the Triton
  `_scatter_codes`, whose own docstring records an unresolved engine-level
  divergence, and the engine has never served with it. Its instrumented bounds
  (`THUNDER_DEBUG_KV=1` -> `[TQ-SCATTER]`, eager-only) were clean in the one step
  measured (`sm_max=-1`, a padded slot), so the fault is elsewhere in that phase.
  The Triton merge is NOT implicated: the capture never runs it (point 1).
- **Correction (measured again):** with the current harness config
  (`cudagraph_mode="FULL_AND_PIECEWISE"`, `cudagraph_capture_sizes=[1]`) capture
  fails at `capture_model` on the REQUEST-MAJOR path too, with BOTH error classes
  in one run (`cudaErrorStreamCaptureUnsupported`, "operation not permitted when
  stream is capturing", then `cudaErrorStreamCaptureInvalidated`), and it does so
  on the pre-session baseline commit `245eea2` as well -- so the dense-vs-request-
  major attribution above is not what fails today, and this is not a regression
  from the 16k work. What is still true: capture fails inside the backend's
  forward, `THUNDER_SKIP_BACKEND=1` makes it succeed, and the failure is in vLLM's
  *profiling* capture (`profile_cudagraph_memory` runs with a 1-block KV cache and
  a throwaway graph pool, and calls `capture_model(profile_only=True)`). The
  harness's own comment in `ci/modal_app.py` (`ours-eager`) has said this for
  longer than this session. Every context therefore serves with
  `enforce_eager=True` today; localizing it needs the CUDA API log
  (`CUDA_LOG_FILE=stderr`) or a commit bisect, not another path ablation.
- Mitigation: the path is chosen from the reservation -- the 24 GiB memory budget
  and, since FAILURE_MODES 15, `ADDRESSABLE_GATHER_BYTES` (4 GiB), because the
  request-major K/V tensor past that is not addressable by a 32-bit CuTeDSL
  index. 4k (2.06 GiB) keeps the capturable request-major path; 8k, 16k and 32k
  take the dense one, so those contexts serve with `enforce_eager=True` until the
  dense gather is capturable. The eager path is correct (vLLM's own warmup run
  completes at 32k, and 16k's init + generation complete on the dense path). Do
  not "fix" this by lowering the budget: that trades a working context length for
  a startup failure.

## 15. A gathered buffer past 32-bit addressing — LIVED (fixed)

The ctx-16384 fault was never a served prefill and never an index bug: it is
vLLM's own warm-up. `kernel_warmup` -> `_run_flashinfer_autotune_dummy_runs` ->
`runner._dummy_run` launches a prefill-shaped step (1024 requests of 16 query
rows, `seq_lens` 16 for every request, `max_query_len` 16), and the illegal
address is taken in *that* launch. The engine's geometry, from
`THUNDER_DEBUG_LAUNCH` plus a metadata dump wrapped around `forward`:

```
q=(16384, 32, 128) fp16   num_reqs=1024   max_query_len=16   is_prefill=True
kv=(278383, 8, 16, 128)   bt=(1024, 1032)   sl=(1024,) all 16   qsl=(1025,)
```

`bt` is 1032 columns wide (`ceil(16448/16)` rounded up to a multiple of 8), so
the request-major gather reserves `max_num_reqs * max_blocks_per_req` =
1024 x 1032 = 1,056,768 block-rows = 16.9M token rows. The K tensor the kernel
is handed is therefore `(16908288, 8, 48)` uint8 = **6.49e9 elements** and V is
8.65e9. CuTeDSL addresses those buffers with 32-bit offsets, so any request whose
row offset passes 2**32 bytes is addressed modulo the wrap: the read lands back
inside the same 6.5 GB buffer (silently wrong data) or outside it
(`cudaErrorIllegalAddress`). That is why the same grid cell could pass in a sweep
and fail alone, and why the engine's *decode* dummy steps at 16k never failed
first: they address few requests, so their row offsets stay far below the wrap.

- Detection: the fault follows the **buffer size**, not the rows read. One cell
  pair settles it (`prefill-dummy-16k`, the engine's geometry with `live_blocks=1`,
  so its `q` is a correct 16384 rows): the default policy takes the dense gather
  and measures 0.760 ms, while `THUNDER_8B_INDIRECT=0` +
  `THUNDER_ALLOW_UNADDRESSABLE=1` — the same launches on the request-major
  reservation — takes an illegal address. Every `batch=1` shape is clean because a
  1-request reservation is small, which is why `prefill-16k` always passed while
  the 1024-request shapes did not. On the engine, forcing the dense gather makes
  the whole 16k init *and* generation pass. The driver's XID MMU faults all sit
  inside a 4 GiB window (0x2a84..0x2b67) — that window is the wrap.
  `use_32bit_stride=False` on the dlpack conversion does **not** remove it
  (tested), so the wrap is not the dynamic-stride bitwidth: the only fix is to
  keep the tensor addressable.
- Mitigation: `ADDRESSABLE_GATHER_BYTES` (4 GiB) is now a second reason to take
  the dense gather (`_use_indirect_gather`), and `PagedKVManager.reserve` refuses
  an unaddressable reservation outright, naming the geometry, so the failure
  cannot come back as an illegal address thousands of launches later.
  `THUNDER_ALLOW_UNADDRESSABLE=1` reproduces the wrap on demand, the way
  `THUNDER_8B_INDIRECT` reproduces the OOM.
- Measured bracket for the 4 GiB constant: 4k's request-major tensors are
  2.06 GiB (V) and run; 16k's are 8.06 GiB and fault; the dense reservation at
  16k is 2.12 GiB and runs. 8k (4.06 GiB) switches to dense as well.

## 15b. A warm-up compile that never ran, and cannot — LIVED (removed)

The engine is supposed to precompile every (tile, causal) config the tile policy
can produce, so no CuTeDSL compile (~1.5 s) lands inside a request. It never
happened, for three separate reasons, and the mechanism is now gone.

1. `launch_thunder_attention(compile_only=True)` passed the wrong argument list:
   no `debug`, no schedule constexprs, so the trailing `CUstream` landed on
   `num_splits` and every call raised `ARG_ANNOTATION_MISMATCH`. The exception is
   caught and logged as "kernel warmup failed (non-fatal)", so the only symptom
   was the compile still happening inside the first request.
2. With the arguments fixed, `cutlass.cute.compile` still does not help: it
   populates a different cache key than the `@cute.jit` call path looks up.
   Measured (`ci_probe/probe_compile_cost.py`): the compile-only call took 2.8 s
   and the *next* launch of the same schedule took 2.4 s, i.e. it compiled again.
3. Doing the warm-up as REAL launches does precompile, but it breaks CUDA-graph
   capture. The first forward of a step happens inside vLLM's capture context
   (`profile_cudagraph_memory` -> `_warmup_and_capture`, which runs its warmup
   dummy on the capture stream inside the graph pool), so a launch that allocates
   there poisons the capture: the engine then dies at `capture_model` with
   `cudaErrorStreamCaptureUnsupported` ("operation not permitted when stream is
   capturing") and `cudaErrorStreamCaptureInvalidated`. Measured at 4k, which had
   captured fine before.

- Detection: a capture failure at `profile_cudagraph_memory` that appears only
  once the warm-up launches; and, for the compile itself, timing the first launch
  of a schedule (seconds means the compile is still in the request).
- Mitigation: the mechanism is removed. The compile stays in the first request,
  which is the TTFT number in 18. Making it cheap needs a shape-stable jit key
  (a shapes-free key was tried and reverted as unsafe) or a device-side compile
  cache -- the jit cache is keyed per tensor shape, and the gathered K/V shape is
  the engine's reservation, so nothing outside `forward` can precompile it.

## 16. Host timing buckets that average over capture — LIVED

`THUNDER_TIME_LAUNCH=1` accumulates process-wide sums and dumps means at exit, so
the numbers include the CUDA-graph capture and the engine warmup. `_FASTLAUNCH` is
deliberately disabled while capturing (`if _FASTLAUNCH and not capturing`), so
those calls re-trace MLIR (~0.4 s each) and swamp the mean: a 4k run reported
`kernel=141.99ms` over `n=540`, which would be 77 s of host time inside a 4.3 s
generation. The `plumbing` bucket is small and believable in the same dump
(0.25 ms), which is how the pollution is visible.

- Detection: bucket mean x call count far exceeding the wall time of the run.
- Mitigation: read `plumbing` only, or reset the counters after warmup before
  attributing a decode step. `docs/FAILURE_MODES.md` 3 already warns that these
  buckets measure dispatch, not execution.

## 17. Reading a measurement's geometry off the wrong thing — LIVED

I recorded that the grid's chunked-prefill shapes (batch 256, seqlen_q 16,
seqlen_k 4096) understated their causal work, because `make_synthetic_batch` fills
`q` from the start of a synthetic sequence and the rows therefore *look* like they
sit at the start of the KV. That was wrong, and it nearly cost a valid measurement:
the kernel anchors a request's query rows at the END of its context --
`_valid` masks with `kv_len - q_len + q_off + row` "because vLLM appends the query
tokens to the request's existing context" -- so those rows attend the whole
4096-token prefix, exactly as an engine chunk does. The -30% chunked-prefill tile
number is faithful.

- Detection: read the kernel's masking, not the tensor's contents. Synthetic values
  are random, so the *data* never tells you where the rows are; only the mask does.
- Mitigation: when a measurement's validity depends on geometry, cite the code that
  defines it (here `_valid`) in the same breath as the number.

## 18. A prefill target that the available levers do not reach — OPEN

The goal was request throughput >1.4x upstream's 109.5 tok/s, i.e. >=153 tok/s, i.e.
TTFT + 31 x ITL <= 209 ms. Measured floor with the packed prefill in place:
36 x 4.45 ms = 160 ms of prefill attention plus 31 x 5.5 ms of best-case ITL = 331 ms,
about 65 tok/s. The target needs the prefill KERNEL faster, and the cheap knobs are now
exhausted, all measured:

| lever | result |
|---|---|
| tile (m/t/n) | 64/128/16 is the optimum; 128/256/16 5.24 ms, 128/256/32 5.11, unpacked 5.41 |
| pipeline depth (num_stages / num_dequant_stages) | DEAD KNOB, not a result: stored on the kernel and never read (no circular pipeline in v0), so the "flat" measurement was noise |
| GQA packing the prefill | -6.8% at the served shape (4.77 -> 4.45), -33.6% at vLLM's profiling batch |
| 16-byte vectorized KV load / paired 4-bit dequant reads | regressed (+9.9%, +24%) |
| wider M for the packed schedule (rp = tile_m // G) | regressed (4.63 -> 5.24 ms) |
| wider KV tile at the same warp count (decode) | regressed: b16/32k 2.72 -> 3.73 (n=32) -> 5.91 (n=64); b16/4k 0.350 -> 0.466; b1/32k 0.215 -> 0.329 |

**Resolved (host side): the KV store ran twice per layer per step.** `forward`
guarded its fallback store with `if not getattr(layer, "_tq_cache_updated", False)`,
and `do_kv_cache_update` set that flag with
`for holder in args: if hasattr(holder, "_tq_cache_updated")` -- a fresh layer never
has the attribute, so the guard was never satisfied and EVERY layer stored its K/V
again on top of the hook's own store. `THUNDER_STAGE_TIMING=1` at 4k put the
forward's `prefix` bucket (which contains that store) at **p50 6.906 ms per layer**
against 0.125 gather / 0.075 qrot / 0.253 launch / 0.035 inverse; with the flag set
unconditionally the same bucket reads **p50 0.007 ms**, the forward's host total
falls from 7.407 to 0.465 ms (16x), and the decode rate in the serialized
diagnostic config goes 1.7 -> 3.3 tok/s. The store itself is still the largest
remaining host term (it is the exact torch reference path for 3-bit K, see the
store notes), but the duplicate is gone. `prefix` is the bucket to watch: it should
stay in the microseconds.

What is left is structural: the dequant is ~70% of the prefill's issue rate and every
cheap way to reduce its op count has been tried and measured. The other half of the
budget is unexplained and must be localized before anything else is attempted -- the
TTFT is 1.8-4.2 s against 36 x (4.45 ms kernel + ~1 ms host) = ~0.2 s, so ~1.5 s
happens once per request, and the ablation that removes the kernel launch
(THUNDER_SKIP_KERNEL) removes it. The GPU-event timer added for this returns -1 (its
own except catches something) and needs debugging: it is the one instrument that can
say whether that 1.5 s is GPU or host.

## 19. A guard that can never be satisfied — LIVED (fixed)

`forward` stores the layer's K/V only when vLLM's separate hook has not already
done it: `if not getattr(layer, "_tq_cache_updated", False)`. The hook set that
flag with `for holder in args: if hasattr(holder, "_tq_cache_updated")` -- and a
fresh attention layer never has the attribute, so the condition was false for
every holder, every step: the flag was never set, the guard was never satisfied,
and **every layer stored its K/V twice per step** (once through the hook, once
through the fallback). Nothing failed and nothing was wrong numerically -- the
store is idempotent for the same K/V -- so the only symptom was time, and it read
as "the CuTeDSL launcher is expensive" because the forward's own buckets were the
only instrument pointed at the host path.

- Detection: `THUNDER_STAGE_TIMING=1` at 4k (with
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the dump lands in the same process as the
  engine). The forward's `prefix` bucket -- the phase that contains that store --
  read **p50 6.906 ms per layer** against gather 0.125 / qrot 0.075 / launch 0.253
  / inverse 0.035. A phase that is 93% of a 7.4 ms forward is not a launch cost.
- Related (this session): the store's SCATTER is now a Triton kernel
  (`_scatter_codes_kernel`) because the torch one decided `PAD_SLOT_ID` on the
  host and this store runs inside vLLM's CUDA-graph capture (see 14). Bit-exact
  against the torch reference in all four probe cases; the torch quantizer is
  untouched.
- Mitigation: set the flag unconditionally on every positional holder, and CONSUME
  it in `forward` (`layer._tq_cache_updated = False` after the check) so a step
  whose hook does not run -- a KV-sharing layer with `key is None` -- still stores
  through the fallback. Measured after: `prefix` p50 **0.007 ms**, forward host
  total 7.407 -> 0.465 ms (16x), decode 1.7 -> 3.3 tok/s in the serialized
  diagnostic config. `prefix` is now a bucket to watch: it belongs in the
  microseconds, and anything above ~0.05 ms means a store is running where it
  should not.
