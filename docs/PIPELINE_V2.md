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

### Where the served step actually goes (measured)

`--e2e "4096|ours-eager|3|4"` -> ITL 336.46 ms. Over 36 layers that is 9.3 ms per
layer, of which the attention kernel is 0.215 ms: **97.7% of the served step is
NOT the attention kernel.** So the 1.4x target is primarily a HOST-PATH and
memory-traffic problem, and the kernel work (above) is the secondary lever.

One structural suspect, visible in the code: the gather reservation is sized for
the ENGINE capacity (1024 reqs x 264 blocks x 16 = 4,325,376 token rows = **1.7 GB
per tensor**) while batch 1 / 4k has only 25 MB live -- a 66x over-reservation that
the launcher then reshapes and `.contiguous()`s every layer every step.

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
