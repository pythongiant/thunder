# v2 pipeline blueprint: B200-native quantized attention

Architectural decision (recorded, not assumed): the measurements make
tuning the v1 kernel into FA4 implausible. v1 becomes the measurement
baseline and e2e vehicle. The target is a new pipeline whose architecture
is derived from FA4, with TurboQuant compression as the differentiating
data path.

```text
CURRENT
Q-head CTA
→ synchronous MMA
→ serial softmax
→ repeated KV reconstruction
→ separate movement

             ↓ REPLACE (not tune)

TARGET
KV-head/GQA tile ownership
→ compressed KV loaded once
→ async/TMA pipeline
→ UMMA/TMEM
→ QK ↔ softmax overlap
→ PV overlap
→ SplitKV scheduling
→ device-side reduction
→ direct paged KV
```

## What carries over from v1 (proven, keep)

- TurboQuant representation: rotation, Lloyd-Max codebooks, packing,
  per-head norms, byte layouts. The data path is the differentiator;
  it does not change.
- CSR/indirect metadata concepts (per-step shared index, per-layer
  payload) as the addressing model for direct paged consumption.
- Split-K merge algebra (online-softmax rescale of partials).
- Fast-launch discipline: config-keyed compiled-function cache, never
  `id()`-keyed; arm/hit/fallback counters.
- Telemetry + frozen matrix + fork spec as the validation harness.

## What is new in v2 (build, in ladder order)

1. **KV-head/GQA tile ownership from the start.** One CTA owns a KV
   tile and serves every Q head sharing it. No Q-head-grid fallback
   inside v2 (v1 keeps both grids for measurement).
2. **Async movement.** cp.async G2S staging first (idiom already probed);
   TMA descriptors via FA4's precomputed path second. Movement overlaps
   MMA from day one — never a separate phase.
3. **UMMA/TMEM.** Warp-specialized MMA with TMEM accumulators. Known
   prerequisites from the tcgen05 attempt: precomputed TMA descriptors
   (`declare_ptx_smem_desc` + `gemm_ptx_precomputed_varname`) and
   `PipelineUmmaAsync` participants for the completion path. v2 starts
   from the FA4 Blackwell GEMM shape, not from v1's `mma.sync` loop.
4. **Overlapped softmax.** Software exp + conditional rescale (FA4
   machinery); QK↔softmax and PV overlap by pipeline stage, so a faster
   QK does not expose softmax as the next wall.
5. **SplitKV scheduling.** KV-sequence partitioning with load-balanced,
   cache-aware CTA ordering (FA4's causal-imbalance/L2-locality policies
   as the starting point, tuned by the 2×2). Split counts chosen per
   workload, never hard-coded.
6. **Device-side reduction.** Split partials reduce on-device; no host
   merge in the decode loop.
7. **Direct paged-KV consumption.** The kernel walks the block table
   (TMA or fused indirect addressing by measurement); the gather-into-
   temporary path is deleted, not optimized.

## Validation ladder (each rung gates the next)

1. v1 matrix frozen as the baseline to beat (per workload).
2. v2 stage 1 (pipeline shell, dense operands): parity with FA4 first —
   proves the execution skeleton before quantization enters.
3. v2 stage 2 (+ packed path): correctness vs rotated oracle, then
   `fa4_matrix` — must meet or beat v1 everywhere before proceeding.
4. v2 stage 3 (+ GQA ownership, SplitKV, direct paged): full matrix +
   e2e + quality gates per `docs/ROADMAP.md` M7.

## Explicit non-goals for v2

- Prefill stays on v1 until decode wins. v2 is decode-first.
- No second grid mode inside v2 (keeps the schedule honest).
- No new quantization representation until the pipeline wins on the
  current one (representation is proven; execution is not).

## Measured delta vs FA4 (read from the FA source, not assumed)

Source: `flash-attention/` (the FA repo, incl. `flash_attn/cute/` = FA4 in the
SAME CuTeDSL stack we use), `benchmarks/results/fa4_matrix_b200_full.md` (frozen,
same methodology both sides), and our own code.

The worst cell is the one that matters most, hd128/gqa4 (Qwen3-8B's shape):

| cell | FA4 ms | ours ms (frozen) | ratio | GB/s FA4 / ours |
|---|---|---|---|---|
| hd128-gqa4-causal-decode-B16-S16384-sp4 | 0.2012 | 14.1858 | **70.5x** | **5339 / 20** |
| hd128-gqa4-causal-decode-B16-S4096-sp4 | 0.0860 | 3.7374 | 43.5x | 3124 / 19 |
| hd128-gqa4-causal-decode-B1-S16384-sp4 | 0.1955 | 1.1788 | 6.0x | 343 / 15 |
| hd128-gqa4-causal-prefill-B1-S16384 | 1.6823 | 105.456 | 62.7x | 199 / 3 |

Two corrections to how this table is usually quoted:

- It is **frozen** (pre-session). The shipped kernel has since moved b16/32k
  from 24.11 to **2.72 ms**, so the *current* ratio on that cell is ~7x, not 70x
  (FA4 at S32768 ~= 0.40 ms by linear extrapolation, since it is bandwidth-bound
  there -- inference, not a measurement).
- The **GB/s column is the whole story**. FA4 runs that cell at 5339 GB/s, about
  67% of B200 HBM peak; we run it at 20 GB/s (frozen) and ~350 GB/s now, a few
  percent. Ours moves LESS data (112 packed bytes per token-head vs 256 fp16) and
  is still far slower. So this is not a data-volume or arithmetic problem: it is
  a **latency-hiding / issue-rate** problem.

### Mechanism-by-mechanism (ours vs FA4)

| | ours (`thunder_vllm/attention/cute_kernel.py`, shipped) | FA4 (`flash_attn/cute/flash_fwd_sm100.py`) |
|---|---|---|
| KV movement | `_load_kv_packed_full` (cute_kernel.py:967): global -> registers -> smem, **two-phase but synchronous, single-buffered** | `cpasync.CopyBulkTensorTileG2SOp` (TMA bulk G2S, flash_fwd_sm100.py:656) |
| async copies | `cpasync` is **imported (cute_kernel.py:49) and never called** | TMA + `PipelineTmaUmma` with byte-count mbarrier (1050) |
| stages in flight | 1 | `kv_stage = min((224*1024 - smem_size_q_o)//smem_size_kv_per_stage, 32)` (425) |
| softmax | serial, in the MMA warps | dedicated softmax + correction warps (1021-1027) |
| accumulators | registers | TMEM, ping-pong S/P slots (338-375), `TmemAllocator` (994) |
| scheduling | static grid | CLC dynamic persistent, 2-CTA MMA clusters |
| split-KV reduce | separate Triton `_merge_splits_kernel` | device-side `flash_fwd_combine.py` |
| paged KV | gather into a temporary, then kernel reads it | consumed in-kernel via TMA |

The `num_stages`/`num_dequant_stages` knobs measuring DEAD is *consistent* with
this: a "stage" knob cannot help when no copy in the loop is asynchronous.

### Porting ladder (each rung verified by the matrix, not by inspection)

We are not starting from zero: `thunder_vllm/attention/cute_kernel_tcgen05.py`
already builds `PipelineUmmaAsync` (621), `NamedBarrier` (631), `TmemAllocator`
(634) and `declare_ptx_smem_desc` (662) -- but at **`num_stages=1`** (351), and it
is not the shipped kernel. Rungs, in dependency order:

1. **Async, multi-stage KV pipeline** (the missing machinery). TMA descriptors for
   the packed K/V tiles; feed the KV pipeline asynchronously; raise
   `num_stages` and size it from the smem budget as FA4 does. This is the rung
   that targets the measured bandwidth collapse directly.
2. **Warp specialization**: softmax/correction warps + named barriers, so dequant
   and softmax stop serializing behind the MMA.
3. **TMEM accumulators** (tcgen05), taking the existing attempt past `num_stages=1`.
4. **Direct paged-KV consumption** (delete the gather; item 7 above).
5. **Device-side split-KV reduction** (replace the Triton merge).
6. **Register/exp2 tuning**: FA4 spends 184-192 registers on softmax warps and
   *emulates* exp2 (`ex2_emu`, SM100's native ex2 is slow), with the tuning table
   keyed by (2cta, causal, hdim, sm103).

Verification for rung 1: `python -m benchmarks.fa4_matrix --quick` on
hd128-gqa4-causal-decode-B16-S16384-sp4, and the GB/s column must rise before any
time number is believed. A rung that does not move GB/s is not the bottleneck.

## Target: >= 1.4x the upstream TurboQuant inside vLLM (served, not kernel)

Goal restated by the user: the *served* system must be 40% faster than upstream
`TURBOQUANT` (`turboquant_3bit_nc` in vLLM), not merely close the FA4 kernel gap.

### First-principles floors (Qwen3-8B, B200 8 TB/s, 2.25 PFLOPS dense)

| quantity | floor | upstream | ours |
|---|---|---|---|
| decode b1/4k ITL | 2.07 ms/step (484 tok/s) | 9.13 ms (109.5 tok/s) = 4.4x off | 333 ms (3.0 tok/s) = 161x off |
| decode b1/32k ITL | 2.18 ms/step | 55 ms (18.2 tok/s) | blocked |
| TTFT prefill 4096 | 32 ms (72 TFLOP) | 74.3 ms = 2.3x off | 1817 ms = 57x off |

Target = 1.4 x 109.5 = **>= 153 tok/s @4k, i.e. ITL <= 6.5 ms**, which is 3.2x off
the floor -- i.e. the target does NOT require beating a near-optimal baseline.
Upstream itself is 4.4x off the roofline, and our representation has a 4.6x byte
advantage (31.5 KiB vs 144 KiB of KV per token), so the target is reachable
without beating FA4.

### Where the served step actually goes (MEASURED)

`--e2e "4096|ours-eager|3|4|0|VLLM_ENABLE_V1_MULTIPROCESSING=0,THUNDER_TIME_LAUNCH=1"`:

```
[TQ-LAUNCH] n=1368  plumbing=0.34ms  kernel=25.95ms  merge=0.16ms  gpu=-1.00ms  total=26.45ms
```

`n` = 1368 launches = 36 layers x 38 forwards. The `kernel` bucket (the
compiled-function invocation, i.e. `_t2 - _t1`) is **98% of the launch cost at
25.95 ms per launch**; plumbing (which covers the reshapes, `.contiguous()`s and
dlpack wrapping) is 0.34 ms and merge 0.16 ms. `gpu=-1.00` means the GPU-event
path produced no timing.

This is NOT the MLIR arm: `--e2e "...|VLLM_ENABLE_V1_MULTIPROCESSING=0,THUNDER_DIAG=1"`
reports `{'launch_reqmajor': 1368, 'attn_calls_eager': 1368,
'attn_calls_eager_prefill': 144, 'fast_armed': 7, 'fast_hit': 1360}` -- the fast
path HITS on 99.4% of launches (7 distinct keys). So the 26 ms is inside the
invocation, i.e. GPU work far larger than the live problem.

**SUSPECT TESTED AND REFUTED: the reservation size is NOT the cause.** The gather
reservation is sized to the engine capacity (`page_rows=270336` = max_num_seqs 1024
x 264 blocks/req) and only 4096 page rows are live at batch 1 / 4k, so it looked
like a 66x over-reservation. But an A/B with `E2E_MAX_SEQS=16` (harness now honours
it, shrinking the capacity 64x) left ITL unchanged (649 vs 537-600 ms) and the
counters byte-identical (`fast_hit=1360/1368`). So the reservation size does not
drive the 26 ms.

Do not repeat the arithmetic error that made this look decisive: `peak=165.6 GiB`
in the e2e log is NOT the gather buffers. `gpu_memory_utilization=0.85` on a 180 GB
B200 is a ~153 GB KV cache, so the peak is weights + cache, and 144.5 GB of gathers
could not coexist with it. The per-impl managers must be shared across layers (or
the reservation much smaller) -- check `_ensure_paged`'s cache key before quoting
per-layer buffer sizes again.

Two harness lessons, both paid for: the counters must be read with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` (otherwise the atexit dump fires in a process
that launched nothing and prints `n=0`), and `THUNDER_TIME_LAUNCH=1` roughly
doubles wall time because it creates and records two CUDA events per launch.

Still unexplained: `jf(...)` -- the fast-path HIT -- costs ~13 ms of host wall time
per decode launch, when a hit is supposed to be ~0.35 ms. `fast_fallback` is absent
from the counters, so `jf` succeeds; it is simply expensive.

The warmup confounder is REMOVED: `E2E_GEN=256` still reports "2 decode" and
ITL 474 ms (vs 537-649 ms at GEN=32), so ~474 ms/step = ~13 ms/layer is
steady-state, not a two-sample artifact.

Ranked next tests, cheapest first:
1. `--e2e "4096|ours|3|4"` (CUDA graphs) vs `ours-eager`: a graph replays all 36
   layers with no per-layer host launch, so if the served rate jumps by the ~28x
   the recorded baseline text implies (58 vs 2 tok/s), the 13 ms/layer is HOST
   launch cost and not GPU work. This is the single most informative comparison
   and needs no new instrumentation.
2. Fix the GPU-event path (`gpu=-1.00`): without GPU time the host/GPU split inside
   `jf(...)` cannot be resolved. `THUNDER_TIME_LAUNCH=1` already creates the event
   pairs; the read-back at exit is what fails.
3. `THUNDER_FASTLAUNCH=0` A/B: if ITL is similar, the fast path is not the lever and
   the cost is inside the kernel/GPU; if much worse, the hit path is already saving
   us and the target is elsewhere.

### THE critical path: graphs are unusable, and graphs are the 13 ms/layer fix

Measured head-to-head on the same cell:

| cfg | result |
|---|---|
| `ours-eager` | serves: ITL 474 ms (steady state), 2.1 tok/s |
| `ours` (CUDA graphs) | **FAILS at engine init**: `cudaErrorIllegalAddress`, engine core init failed |

A captured graph replays all 36 layers from one graph with NO per-layer host launch,
so the ~13 ms/layer measured inside `jf(...)` is precisely what graphs remove -- which
is why the recorded baseline text lists `ours` at 58 tok/s against the 2.1 tok/s
`ours-eager` delivers. Therefore:

**Priority 1 is the post-capture IMA, because it gates the only configuration that is
fast, and it is also the same fault that blocks vLLM's DEFAULT (graphs on).**

What is established about it: both captures now COMPLETE (piecewise 13.5 s, FULL
0.75 s after the fast-path guard fix); the fault lands AFTER them, in the KV-cache-init
phase; the XID is an MMU read fault; and with `THUNDER_FAST_DEBUG=1` no
`MISS ... capturing=1` line appears, so that phase does NOT reach
`launch_thunder_attention` -- the fault is in the gather / CSR build / store that runs
before the launch. Bisection is blocked for the store A/B: `THUNDER_STORE_TORCH=1`
makes the store sync, so the capture fails earlier with
`cudaErrorStreamCaptureUnsupported` and never reaches the phase in question.
### Literature that applies (searched, not assumed)

- **BitDecoding** (Du et al., HPCA 2026, arXiv 2503.18773): the closest work --
  low-bit KV decode at long context. Its claims: existing systems "decode
  inefficiently by relying solely on CUDA cores, underutilizing Tensor Cores",
  which is exactly us (LUT dequant + `mma.sync`). Techniques to take: induce
  tensor-core-friendly layouts, warp-level dequantization parallelism, a
  software-pipelined dequantization kernel for mixed-precision execution, query
  transformation, and **Blackwell NVFP4/MXFP4 formats**. Reported 7.5x average /
  8.6x on Blackwell over FP16 FlashDecoding-v2, 3x single-batch latency at 128K.
  Open source: github.com/OpenBitSys/BitDecoding.
- **TurboQuant** (Zandieh et al., ICLR 2026, arXiv 2504.19874): our own quantizer's
  paper. Includes **QJL bias correction** (random-rotation + Lloyd-Max + a
  sign-based correction term that keeps the attention estimate unbiased), which is
  a MATH lever we do not currently use and which could relax how much accuracy
  headroom 3-bit K needs.
- **turboquant_cutile** (a B200 cuTile implementation of the same algorithm,
  devtechjr.github.io/turboquant_cutile): a working blueprint for our target
  hardware. Five kernel types, and a FUSED attention kernel (score + QJL
  correction + online softmax + V accumulation in one pass) that decompresses V
  **on-chip** instead of round-tripping it through HBM -- the same round-trip our
  gather performs. Its Blackwell list: pipelined TMA loads with `latency=2`
  prefetch hints, TMEM -> tensor-core single hop, `exp2(flush_to_zero=True)` with
  base-2 softmax, approximate division, block swizzling for L2 (954 -> 899 us at
  16k), `occupancy=2`.

### Order of work for the 1.4x target

1. **Attribute the 9.3 ms/layer** (`THUNDER_TIME_LAUNCH=1` splits plumbing/kernel/
   merge; `THUNDER_STAGE_TIMING=1` splits gather/qrot/launch/inverse) and delete
   the largest term. Candidate: the 1.7 GB reservation's reshape/`contiguous()`
   and the gather round-trip.
2. **Right-size the reservation** to the live batch rather than the engine
   capacity (it is what makes the above term huge), keeping pointer stability for
   graphs.
3. Then the kernel rungs above (async multi-stage pipeline first).
4. Optional math lever: QJL bias correction to buy accuracy headroom, and/or
   NVFP4 for native tensor-core execution of the low-bit path.

## Diagnosis-driven next steps (external review, verified against the code)

Ranked by expected value, after checking each claim against our own measurements.

**Arithmetic correction accepted.** The shipped 3-bit prefill run is 137 GFLOP /
5.212 ms = **26.3 TFLOP/s**, not 29.5 (that figure was the 4-bit run at 4.644 ms).
FA4 is **1212 TFLOP/s**, so the gap is **46x** either way.

### P1. Prefill: dequantize once, then a dense attention (DIAGNOSTIC, not product)

Our prefill re-dequantizes the whole KV sequence per q-block (64x redundant at 4k).
Sizing the prize: one dequant pass is ~1/64 of the current dequant work, so
~0.065 ms + FA4's 0.113 ms ~= **0.18 ms against 5.212 ms, ~29x**. If it holds, the
two prefill cells go 0.0217 and 0.0230 -> ~0.6 and ~0.6, and the geomean goes
**0.114 -> ~0.42** (3.7x).

Run it as a *measurement* to size the structural prize, not as the shipped path:
calling FA4 from our plugin would make us an FA4 wrapper and the "our kernel beats
the kernel upstream runs on" claim vacuous. The product-shaped version of the same
idea is to dequantize once into an fp16 scratch and run **our** kernel on it (no LUT
in the inner loop). Keep the compressed cache as the persistent representation so
the batch-1 decode win is untouched.

### P2. Aligned 3-bit packing (best product-shaped idea; CONFIRMED by our packer)

`quant/packing.py::pack_indices` uses a `_contributions(bits, head_dim)` table whose
per-code work is `((idx[c] >> src) & mask) << dst` -- i.e. variable shifts with codes
crossing byte boundaries, exactly the cost identified. A 10-codes-per-32-bit-word
format makes extraction a single aligned word load plus one shift/mask, at 3.2 bits
per value: K grows 48 -> 52 bytes per head (**+8.3%**) instead of 4-bit's +33%.

Expected magnitude, bounded by measurements we already have: 4-bit bought **+12% on
prefill** (instruction-bound) and cost **-5% on decode b16** (traffic-sensitive).
+8.3% bytes is a quarter of 4-bit's penalty, so the trade is strictly better than
the nibble padding already rejected -- predict **~+8-12% prefill, ~-1-2% decode**,
i.e. a small but real geomean gain. Multi-file and correctness-sensitive (packer,
layout, Triton store, kernel `_unpack`), so it needs a dedicated session with the
parity tests as the gate.

### P3. Split-KV for batched decode -- ALREADY IMPLEMENTED

`num_splits` up to 64 exists, the policy takes it (`max_splits_batched=16` above
`BATCHED_DECODE_FROM`), and the launcher only forbids split-mode when
`max_query_len > 1` (prefill). Measurements: splitting buys ~25% going 1 -> 4 at
batch 16 (3.21 -> 2.39 ms at 16k, frozen 4-bit matrix) and the curve is then **flat
from 8 to 64 at 32k** (splits.py documents the same knee). So the remaining lever
here is P4 (overlap), not adding split-KV.

### P4. Async double-buffering (the real structural fix)

Agreed, including the caveat: "importing cpasync alone does not make a kernel
asynchronous". `cpasync` is imported at cute_kernel.py:49 and never called; PASS 2
is a ~6-barrier serial chain per tile. Feasibility already checked: packed rows are
48/64 B (16-byte aligned), so 16-byte-granular cp.async works, and doubling the four
buffers costs ~+10 KB SMEM per CTA -- affordable since occupancy is proven NOT to be
the limit. **Pipelining hides latency; it does not remove unpack instructions**, so
it must not be started before the profile says which one dominates.

### The vLLM attribution needs a timeline (partly conceded)

Attention is 2.5 ms of a ~474 ms step, so "attention is the cause" is not supported,
and 360 launches would each need ~1.31 ms of overhead to explain it by launch gaps
alone -- a claim that needs a timeline, not an inference.

What our own instrumentation *does* show: `THUNDER_SYNC_PROBE` reported ~0.38 ms of
GPU work per launch averaged over 1368 launches, i.e. the GPU is idle for most of the
run. But two caveats weaken that as attribution: the mean includes the (much larger)
prefill launches, and the profiling runs forced `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
which our own notes record as inflating ITL (16.6 -> 78 ms at 4k). So the honest
statement today is **"the GPU is idle most of the step; the cause of that idle time
is not yet attributed"** -- fix with `ncu` for the kernel and Nsight Systems (or
vLLM's own trace) for the orchestration.

### Order of work

1. **Two `ncu` captures first** (prefill 4k; decode b16 32k): issue-active %, stall
   reasons (barrier / wait / short-scoreboard / long-scoreboard / MIO throttle),
   DRAM and SMEM traffic. Interpret as: low issue + stalls => P4 (overlap); high
   issue with integer/SMEM work => P2 (representation); low DRAM but high
   memory-instruction pressure => narrow loads / LUT traffic.
2. **P1 as a diagnostic** -- it decides whether the fused unpack belongs in prefill
   at all, and it is the fastest way to size the whole structural prize.
3. **P2** (aligned 3-bit) against the current 3-bit, 4-bit, and the P1 result.
4. **P4** only if the profile shows actionable stalls; then re-measure.
5. **vLLM end-to-end separately**: fix FM14 capture, then a timeline before any
   attribution.

## ncu profiles: the kernel is starved, not instruction-saturated (MEASURED)

Method: `ncu --profile-from-start off --clock-control none --target-processes all`
around a single launch via `ci/ncu_one.py`, so only our attention kernel is replayed
(not the quantizer, store, gather or FA4). **`--clock-control none` is required
here** -- the default (`base`) tries to lock GPU clocks and aborts with "Failed to
lock GPU clock frequencies". Percentages are therefore relative to unlocked clocks:
indicative within a run, not reproducible across runs.

| metric | prefill 4k | decode b16 32k |
|---|---|---|
| Duration | 5.22 ms | 2.58 ms |
| **DRAM Throughput** | **0.09%** | **2.34%** |
| Memory Throughput / Mem Busy | 67.8% | 71.0% |
| Compute (SM) / Mem Pipes Busy | 20.7% | 24.4% |
| Eligible Warps Per Scheduler | 0.22 | 0.23 |
| **One Or More Eligible** | **20.2%** | -- |
| **No Eligible** | **79.8%** | -- |
| Active Warps Per Scheduler | 1.93 | -- |
| Warp Cycles Per Issued Instruction | 9.52 | 8.69 |
| Avg Active Threads Per Warp | 31.93 (no divergence) | -- |
| **Executed Instructions** | **1.0002e9** | **5.46e8** |
| ncu's top finding | "stalled waiting for a scoreboard dependency", **Est. speedup 41.7%** | (same shape) |

### What this establishes, and what it refutes

1. **NOT issue-saturated** -- refutes the "prefill is instruction-bound" claim made
   from the 3-bit/4-bit delta. The instructions exist (1.0002e9, which matches the
   independent ~1.07e9 LUT-code estimate) but they are not queued: **no eligible
   warp for 79.8% of cycles**.
2. **NOT DRAM-bound** -- 0.09% of DRAM peak in the very kernel that reads the whole
   KV, so the gathered KV is L2/SMEM-resident. **This also puts the earlier
   "decode is traffic-sensitive" reading of the 3-bit/4-bit delta in doubt**: with
   DRAM idle, a 5% decode win from fewer bytes must come from L2/SMEM traffic, not
   bandwidth. Recorded as UNRESOLVED rather than explained away.
3. **Absolute occupancy is the starvation lever** (Active Warps Per Scheduler 1.93,
   ~7.7 warps/SM). The tile sweeps could not move it because they trade CTAs for
   warps 1:1 -- every variant sat at 13.8 warps/SM by construction -- so the fix is
   **per-CTA cost (SMEM and registers)**, not the tile shape.
4. **The top stall is a scoreboard dependency** (ncu's own OPT block, 4.0 cycles per
   warp, est. 41.7% speedup). Short vs long scoreboard is NOT yet distinguished --
   the line was truncated. short = SMEM => cut SMEM round-trips (the P2 direction)
   and add ILP; long = global/L2 => prefetch (the P4 direction).
5. Avg Active Threads 31.93 with 31.51 not-predicated-off => no warp divergence, so
   this is not a mask/predication problem.

### Consequence for the order of work

P4's async prefetch addresses *long*-scoreboard waits, which this profile makes
unlikely (DRAM idle, data L2-resident). The profile points first at **raising
resident warps per SM by cutting per-CTA cost** and at **cutting SMEM dependency
chains** -- the second of which is exactly what P2's aligned packing does (one
aligned word load plus shift/mask instead of byte-spanning reads). P4 stays right for
prefill's barrier-heavy PASS 2 once the stall line is resolved, so: **get the full
stall line before choosing between P2 and P4.**

### Stall resolution: SHARED MEMORY is the bottleneck (Case C). Order changes.

Explicit stall counters, cycles per issued instruction (`smsp__average_warps_issue_
stalled_*_per_issue_active.ratio`) plus pipe throughputs:

| stall reason | prefill 4k | decode b16 32k |
|---|---|---|
| **short_scoreboard (SMEM)** | **3.99** | **4.69** |
| barrier | 1.97 | **0.02** |
| wait | 1.54 | 1.97 |
| long_scoreboard (L2/global) | 0.80 | 0.77 |
| mio_throttle / lg_throttle | 0.01 / 0.00 | 0.08 / 0.01 |
| not_selected / math_pipe_throttle | 0.08 / 0.03 | 0.09 / 0.04 |

| throughput | prefill 4k | decode b16 32k |
|---|---|---|
| **l1tex (SMEM/L1TEX pipe)** | **68.5%** | **71.0%** |
| lts (L2) | 2.55% | 2.22% |
| DRAM | 0.09% | 2.34% |
| sm (compute) | 20.9% | 24.4% |

**Verdict: the L1TEX/shared-memory pipe is the bottleneck and its latency is what
the warps stall on.** This is the external review's Case C ("low DRAM throughput but
high memory-instruction pressure => excessive narrow loads, shared-memory lookup
traffic, serialized memory operations"), not Case A. Every latency-hiding signal is
absent: long_scoreboard 0.8, mio/lg throttle ~0, L2 2.5%, DRAM <3%.

Two predictions resolved against the record:
- CONFIRMED: advance predicted the decode CTA (single warp) would show no barrier
  stall -- measured **0.02**. The barrier cost (1.97) is prefill-only.
- REFUTED: the "prefill is instruction-bound" claim drawn from the 3-bit/4-bit
  delta. Compute is 20.9%; SMEM is 68.5%.

### Consequence: cut SMEM accesses per KV element FIRST

Our dequant performs **three SMEM accesses per code element**: read the packed
bytes (byte-spanning), read the LUT, write the dequantized code. Then the MMA does a
*fourth* (a `cute.copy` from `sK_code`/`sV_code` into the register fragment). That is
what saturates L1TEX and what the short-scoreboard stalls are waiting on. In reach
order:

1. **Dequantize straight into the MMA register fragment** instead of via `sCode` in
   SMEM -- removes the code-buffer store *and* the subsequent SMEM->register copy,
   i.e. 2 of the 4 SMEM accesses per element. The `mma.sync` path already consumes
   register fragments (`make_fragment_A/B`), so this is a data-path change, not a
   new schedule.
2. **P2's word-aligned packing** so one SMEM load feeds several codes instead of one
   byte-spanning load per code (aligned 3.2-bit words), which also removes the
   variable-shift sequence.
3. **P4 (async prefetch / double buffering) only after that**, and only for prefill's
   remaining barrier component (1.97). Prefetching cannot help a SMEM-latency stall.

Caveat, stated plainly: the profile says *where* the cycles go, not how much a given
fix recovers. Each of 1-2 must be measured after implementation, and the batch-1 32k
win (1.397x) must be re-measured to confirm it survives.

### Measured: two-phase dequant, and the metric's noise floor

ncu said the hotspot is shared memory (short-scoreboard 3.99/4.69, L1TEX 68-71%,
no eligible warp 79.8%). The dequant's own load->store chain is part of that, so
both layouts now stage every LUT load into registers before any store (the pattern
the KV tile load already uses), **bounded to a staging window of <=16 values per
thread**. The bound is not optional: unbounded staging is 64 values per thread at
the decode-b1 tile on top of the KV load's 112, which hung the benchmark (30 min
against 3-4).

| cell | ours before -> after | delta |
|---|---|---|
| prefill 4k | 5.126 -> **4.940 ms** | **-3.6%** |
| prefill 16k | 70.429 -> **68.785 ms** | **-2.3%** |
| decode b16 32k | 2.5891 -> 2.5927 ms | flat (path unchanged by design) |
| decode b16 4k | 0.33539 -> 0.33514 ms | flat |
| decode b1 32k | 0.21962 -> 0.21933 ms | flat |

Correctness: the GPU suite passes with the change (102 passed; the only failure is
the pre-existing `test_cuda_graph_replay_parity` stub, already in
`.auto/known_failures.txt`), including `test_packed_prefill_matches_dequant_reference`
and `test_split_k_decode_matches_dequant_reference`, which exercise exactly the two
edited functions. This matters because the harness's `speedup` column compares
*times*, not values, so it cannot catch a wrong dequant -- a correctness run must
precede any timing claim.

**Noise floor (use this when reading the primary metric).** The aggregate
`speedup_vs_fa4` moved 0.113811 -> 0.113442 (-0.3%) while our kernel got 2-4% faster,
because FA4's own number on the decode-b16-4k cell moved **15%** (0.0715 -> 0.0610)
with our side identical to four digits. Per-cell FA4 variance is therefore ~2%
typically and ~15% worst observed, which puts the aggregate's noise floor around
+-10%. **Decide on `ours_ms`, treat the ratio as indicative**, and re-run a cell
before believing a small move in it.
