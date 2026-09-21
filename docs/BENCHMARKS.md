# Benchmarks

## Baseline rule

Always compare against upstream `TURBOQUANT`, never against a hand-rolled dequant-to-fp16 reference. `benchmarks/bench_common.py::attention_ref` is only a kernel-level sanity lower bound.

Primary model is `Qwen/Qwen3-8B`. Canonical e2e entry is the batched harness reused by `ci_probe/modal_probe_batched.py` and `ci_probe/studio_batched.py`.

## What the existing B200 e2e numbers mean

The B200 fp16-vs-ours matrix was collected, but later runs exposed zombie-`EngineCore` GPU contention. Therefore:

- Treat superlinear cells as invalid measurement artifacts.
- Do not quote the worst ITL cells as kernel scaling.
- Even the cleaner cells compare whole-model ITL against a one-layer microbench; see below.

## Microbench caveat

`ci_probe/kernel_bench.py` and `ci_probe/kernel_stage_probe.py` measure one attention invocation with 8 Q heads and `qhead_per_kvhead=1`. The engine runs 32 Q heads with GQA 4:1 across 36 layers, plus store/gather/rotations per layer. Do not compare microbench milliseconds directly to whole-model ITL.

Needed before the next performance claim:

1. GQA-aware kernel bench with 32 Q heads.
2. Clean-GPU skip-flag A/B: `THUNDER_SKIP_KERNEL=1`, `THUNDER_SKIP_GATHER=1`, `THUNDER_SKIP_ROT=1`.
3. `THUNDER_8B_INDIRECT=0/1` comparison.
4. Fast-launch hit diagnostics.

## Upstream baseline blocker

Upstream `TURBOQUANT` e2e is not selectable at the pinned vLLM commit for dense Qwen3-8B. Do not patch vLLM in the main benchmark environment. For an external baseline, use a separate vLLM release supported by the upstream plugin.
