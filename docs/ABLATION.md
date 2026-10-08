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
| FA4 vs ours matrix | apples-to-apples gate: where we lose to FA4 | `benchmarks/fa4_matrix.py` (`--quick` first) | **frozen, GPU-gated** (result: `benchmarks/results/fa4_matrix_b200_full.md`) |
| Q×K isolation (fork) | dequantize-then-MMA vs native score path | `docs/QK_ISOLATION.md` spec; build variant A on GPU day | **spec only, GPU-gated** |
| −onepass / −reg_rescale / −causal_bound | per-flag wins | `kernel_stage_probe.py` rows | measured (B200: onepass x1.63 prefill, reg_rescale x1.28) |
| +GQA reuse | remove 4× redundant KV reconstruction (M1) | `THUNDER_GQA_PACK=0/1` (now the default); `--mode grid` cell | **measured, shipped**: -76.6% at batch 16/32k and -31% at batch 1, at the tile that was current then. Prefill packs too (the M axis carries the group's heads over `tile_m // qhead_per_kvhead` tokens): 4.77 -> 4.45 ms at the batch-1 4k shape (a repeat pair measured 4.75 -> 4.66, so 2-7%) — the causal staircase of the narrower q-block absorbs most of the redundancy saving, so this is a small win, not the 4× the decode schedule bought |
| +split-K on GQA | occupancy gain, attributed separately | `THUNDER_SPLITS=N` on top of GQA row | **measured, shipped**: the knee moved 16 -> 64 once packing took the head axis (8 KV heads, not 32); batch-1 32k decode 0.537 -> 0.215 ms |
| 4/4 vs 3/4 packing | compression tradeoff | `THUNDER_K_BITS/V_BITS`, `THUNDER_STORE3=1` | measured (store parity) |
| Indirect vs request-major gather | gather-path cost | `THUNDER_8B_INDIRECT=0/1` | measured; request-major is the default |
| Tile shape per schedule | decode vs prefill tiling | `--mode grid` (`shape\|splits\|gqa\|m\|t\|n`) | **measured, shipped**: decode 32x64x32 (batch 1) and 16x32x16 (batch 16), prefill 64x128x16 - `tile_shape.py` carries the table |
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

## E2E column

Upstream `TURBOQUANT` cannot be selected in-process at pinned vLLM
`0.29.1rc1` (see `docs/BENCHMARKS.md`), so the comparison runs as two
stacks on two pins: `ci_probe/modal_upstream_baseline.py` (vLLM 0.25.1,
stock `turboquant_3bit_nc`) against `ci/modal_app.py --mode e2e` (pinned
vLLM, `CUSTOM` backend). `benchmarks/results/upstream_vs_ours.md` holds
both tables plus the fp16 cross-stack control that makes them readable
side by side; read its provenance notes before quoting a ratio.
