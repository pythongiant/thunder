# Ablation plan

Updated replacement for the original 7-row table. Two corrections vs the
original:

- **No K-only build ever existed.** K and V were fused from the start, so
  "Fused K only" vs "Fused K+V" cannot run as separate rows. They collapse
  into one "current fused kernel" row.
- **Dequant-to-FP16 is not the baseline.** Per `docs/BENCHMARKS.md` it is a
  kernel-level sanity lower bound only. The true baseline column is upstream
  `TURBOQUANT` (e2e, currently blocked at the pinned vLLM commit).
- **"V multi-head GEMM" merged into GQA reuse.** One CTA per KV head serving
  the whole query group covers both; track one row, benchmark shapes under it.

## Correctness gate (runs before every row)

`ci_probe/correctness_check.py` — store, gather, kernel-vs-oracle, baseline
and flags-on, plus `+gqa` decode rows. Never report speed on a failing run.

## Kernel-level rows (B200, `kernel_bench.py` / `kernel_stage_probe.py`)

| Variant | Purpose | How to run | Status |
|---|---|---|---|
| Dequant-to-FP16 ref | sanity lower bound only | `bench_common.attention_ref` | exists |
| Fused kernel, current defaults | reference point for all deltas | `kernel_bench.py` | measured (B200) |
| FA4 vs ours matrix | apples-to-apples gate: where we lose to FA4 | `benchmarks/fa4_matrix.py` (`--quick` first) | **frozen, GPU-gated** |
| Q×K isolation (fork) | dequantize-then-MMA vs native score path | `docs/QK_ISOLATION.md` spec; build variant A on GPU day | **spec only, GPU-gated** |
| −onepass / −reg_rescale / −causal_bound | per-flag wins | `kernel_stage_probe.py` rows | measured (B200: onepass x1.63 prefill, reg_rescale x1.28) |
| +GQA reuse | remove 4× redundant KV reconstruction (M1) | `THUNDER_BENCH_HQ=32 THUNDER_GQA_PACK=1 kernel_bench.py` vs off; `gqa_pack` stage-probe row | wired, **GPU-gated** |
| +split-K on GQA | occupancy gain, attributed separately | `THUNDER_SPLITS=N` on top of GQA row | wired, **GPU-gated** |
| 4/4 vs 3/4 packing | compression tradeoff | `THUNDER_K_BITS/V_BITS`, `THUNDER_STORE3=1` | measured (store parity) |
| Indirect vs request-major gather | gather-path cost | `THUNDER_8B_INDIRECT=0/1` | wired, **GPU-gated** |
| HQ=8 vs HQ=32 bench shape | microbench vs engine-like | `THUNDER_BENCH_HQ=8/32` | wired, **GPU-gated** |

## Required run order (jointly optimized scheduling)

```text
correctness gate
→ fused kernel (defaults)
→ 2×2: baseline | GQA-only | SplitKV-only | GQA+SplitKV
→ tune split counts only after the architecture is chosen
→ packing / gather-path variants
```

The combined cell is interpretable only alongside its single-variable
parents — never from baseline in one jump.

## E2E column (blocked, tracked for later)

Upstream `TURBOQUANT` vs ours per `bench_vs_thunder_vllm.py` sweep.
Baseline unselectable at pinned vLLM `0.29.1rc1` — needs a separate
vLLM 0.19–0.25 image. Ours column needs a clean GPU (see
`docs/RUNBOOK.md` hygiene) and M1 landed first.
