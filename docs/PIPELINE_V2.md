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
