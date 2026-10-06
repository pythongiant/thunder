# Roadmap

Driving principle:

> **Fix the largest structural source of wasted GPU work first; only then
> optimize overhead around it.**

The optimization thesis:

> **Ensure every compressed KV element is loaded/dequantized once, consumed
> by every Q head that needs it, and kept entirely on-device.**

Strategy in one sentence: first determine whether TurboQuant's advantage
is algorithmic or merely IO reduction; then build the winning path using
FA4's Blackwell execution architecture, with GQA/SplitKV as jointly
optimized scheduling dimensions, and only afterward optimize serving
integration.

## Pivot: v1 measures, v2 wins

Empirical conclusion: tuning the v1 kernel into FA4 is implausible —
v1 is `mma.sync`, Q-head-owned, serial-softmax, with 4× redundant KV
reconstruction. v1's role is now measurement baseline and e2e vehicle;
the target is the v2 pipeline in `docs/PIPELINE_V2.md` (KV-head
ownership, async/TMA movement, UMMA/TMEM, overlapped softmax, SplitKV
scheduling, device-side reduction, direct paged KV). Milestones M1–M5
below still order the *measurements*; M6–M7 now execute against v2, not
as patches to v1.

Acceptance per workload: kernel throughput ≥ FA4 on the same B200, same
shapes, same causal/GQA conditions (`benchmarks/fa4_matrix.py` is the
frozen gate). Execution template: FA4-class Blackwell pipeline carrying
the TurboQuant data path — not the old schedule incrementally patched.

## Milestone order

### M1. Eliminate redundant KV work — DONE (shipped)

Decode in a 4:1 GQA group reconstructed each KV tile 4× (once per Q-head
CTA). The QK/PV math legitimately differs per head; the KV
reconstruction did not.

- `THUNDER_GQA_PACK` is now the decode default (opt out with `0`), with
  parity cases in `tests/test_correctness.py` and the GPU gate.
- Measured: -76.6% at batch 16/32k (11.62 → 2.72 ms) and -31% at batch 1
  at the tile current then. The prediction held: the win is the KV
  load/unpack fraction, not 4×, and it grew once split-K was retuned for
  the narrower grid (-60% more at batch 1, 0.537 → 0.215 ms).
- The "8 KV-head CTAs trade redundancy for head-axis parallelism" caveat
  was real and is handled by the split policy, not by the packing shape:
  the grid lost 4× of its head-axis CTAs, so the split count had to
  follow.
- Do not assume the current packed shape is optimal: 8 KV-head CTAs trade
  redundancy for head-axis parallelism. Follow-ups, in order:
  - A: keep Q0..Q3 live together (current shape).
  - B: one KV load, sequential per-head consume (lower registers).
  Optimize for **KV reuse**, not for an implementation shape.
- "GQA K reuse" and "V multi-head GEMM" are one item: **GQA KV reuse**.

### M2. GQA × SplitKV, jointly optimized — DONE (shipped)

GQA attacks wasted work; SplitKV attacks insufficient parallelism. The
order held: packing first, splits after.

```text
baseline | GQA-only | SplitKV-only | GQA + SplitKV
```

- The split knee moved with the grid: 8 at the old 64-row tile, 16 at the
  32-row tile, 64 once packing took the head axis (S=16 0.537 ms, S=32
  0.318, S=64 0.215, S=128 0.218 flat at batch 1/32k).
- Batched decode caps lower (16 from batch 16 up): at batch 16/4k the grid
  already has its CTAs, so extra splits only add merge work (0.350 vs
  0.366 ms). Both switches now live in `thunder_vllm/attention/splits.py`
  with the measurements beside them.
- **SMEM constraint:** the decode tile's smem is ~89 KB, which fits two
  CTAs/SM so three need ≤75 KB (drop `sOf` via a register-direct epilogue,
  `sS` to fp16). That is what bounds the tile, and therefore the split
  count, more than the arithmetic does.

### M3. Eliminate KV intermediate movement

Objective: **eliminate the intermediate KV gather buffer** — not
"implement TMA gather" (TMA is one implementation option). If the gather
measures immaterial, skip; the copy stays until a profile says otherwise.

### M4. Device-resident execution

Split merge, graph compatibility, and persistent/device scheduling where
beneficial — once per-step kernel time makes framework orchestration a
material fraction.

### M5. Residual fusion

Measure rotations and framework plumbing separately; fuse only material
costs, and only once attention itself is structurally efficient.

### M6. FA4 execution parity

Close the execution gap item by item, each by measurement:

- async MMA / UMMA
- Blackwell tile/schedule
- software exp
- conditional rescale
- TMA / paged KV
- scheduler / persistence

### M7. Terminal acceptance

- **Kernel:** ours ≥ FA4 on target quantized workloads.
- **E2E:** native integration in vLLM / SGLang / TensorRT-LLM with clean
  apples-to-apples comparison.
- **Quality:** PPL / task quality parity.

## Separate track: prefill

Prefill already has a coherent direction (one-pass, register rescale,
causal bound, tile sizing). Keep it out of the decode milestones:

```text
PREFILL → tensor-core efficiency → tiling → metadata reuse
DECODE  → GQA reuse → occupancy → split-K → gather removal → runtime
```

## Entry/exit gates

- No milestone starts until the previous one's A/B is measured on a
  clean GPU (see `docs/RUNBOOK.md` hygiene).- No e2e claim until the kernel-level A/B for that milestone exists
  (`kernel_bench.py`, `kernel_stage_probe.py`).
- The FA4 matrix (`benchmarks/fa4_matrix.py --quick` first, full matrix
  before any rebuild decision) locates where we lose to FA4.
- The Q×K fork (`docs/QK_ISOLATION.md`) resolves before any pipeline
  rebuild: dequantize-then-MMA vs TurboQuant-native score path.
- Whole-model ITL is never compared against the 8-Q-head microbench;
  use `THUNDER_BENCH_HQ=32` engine-like shapes.
- Every gate above can lie: how each one fails, and how to detect it,
  is tracked in `docs/FAILURE_MODES.md`. Read it before trusting a win.
