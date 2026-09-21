# Q×K isolation experiment (the critical fork)

Question: does TurboQuant's published attention-logit advantage come only
from moving fewer bytes, or from exploiting quantized structure in the
computation itself? The answer decides whether the rebuild target is
"dequantize-then-MMA inside an FA4 pipeline" or a TurboQuant-native score
path. Settle it by measurement before rebuilding anything.

## Experiment

Standalone QK-only microbench on decode-representative shapes (Q = 1–4 rows,
S = 4k/16k, Hk = 8, D = 128, GQA = 4) plus one prefill tile shape. Same
`do_bench_stats` methodology as `benchmarks/fa4_matrix.py`. Report µs,
GB/s per side (own bytes), and tensor-active %.

## Variant A: packed → dequant → MMA (buildable from existing pieces)

New `@cute.jit` micro-kernel reusing the real kernel's helpers — no new
math, no new layouts:

- `_load_kv_packed()` for the packed K tile + norms,
- `_dequantize_strided()` into the fp16 code tile,
- the QK MMA slice + rowmax writeout (skip softmax, PV, epilogue).

Compare A against two references on identical shapes:

1. **Dense-fp16 QK MMA** (Q @ Kᵀ, no softmax): if packed-A ≈ dense-QK,
   dequantization is already hidden and the fork resolves toward
   dequantize-then-MMA inside the FA4 pipeline. If packed-A ≫ dense-QK,
   dequantization dominates and variant B is warranted.
2. **Current full kernel** on the same shapes: A vs full isolates the
   PV + softmax + epilogue share.

## Variant B: TurboQuant-native score path (design AFTER A is measured)

Only build if A loses to dense-QK by a margin that matters. B means
computing scores without materializing an fp16 K tile at all. Candidate
shapes to evaluate then — not now — include accumulating in code-index
space or per-codebook-entry partials. Do not design B until A is measured;
any B sketched today would be optimizing an unmeasured bottleneck.

## Decision rule

- A ≈ dense-QK → rebuild = FA4 pipeline + dequant-then-MMA + GQA reuse.
- A ≫ dense-QK → design B, prototype it standalone, A/B it against A.
- Either way, keep only measured wins; record the numbers in the matrix
  JSON, not in prose.
