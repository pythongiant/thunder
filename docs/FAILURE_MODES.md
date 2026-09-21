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
