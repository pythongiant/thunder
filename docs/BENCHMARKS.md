# Benchmarks

## Baseline rule

Always compare against upstream `TURBOQUANT`, never against a hand-rolled dequant-to-fp16 reference. `benchmarks/bench_common.py::attention_ref` is only a kernel-level sanity lower bound.

Primary model is `Qwen/Qwen3-8B`. Canonical e2e entry is the batched harness reused by `ci_probe/modal_probe_batched.py` and `ci_probe/studio_batched.py`.

## Canonical kernel numbers (B200)

`./autoresearch.sh` is the reproducible entry: it runs the engine's own
config (`onepass`/`reg_rescale`/`causal_bound` on, the tile and split
policies from `thunder_vllm/attention/`, GQA packing on) through
`ci/modal_app.py --mode grid`, so a measured number is the shipped
configuration. Current medians:

| shape | ours (attention launch) | dequant-fp16 ref |
|---|---|---|
| decode 4k, batch 16 | 0.353 ms | — |
| decode 32k, batch 16 | 2.73 ms | — |
| decode 32k, batch 1 | 0.215 ms | 5.60 ms |
| prefill 4k | 4.45 ms (4.77 with GQA packing off) | 15.08 ms |

The prefill cell is `prefill|policy|1|64|128|16`: the packed schedule now covers
prefill too, so the shipped configuration is the one that packs. Measured
tiles for it: 64x128x16 4.45 ms, 64x128x32 5.20 ms, 32x64x16 7.17 ms (a
narrower M tile shrinks the packed q-block, which costs the causal bound more
than the packing saves).

## What the B200 e2e numbers mean

The B200 fp16-vs-ours matrix was collected, but later runs exposed zombie-`EngineCore` GPU contention. Therefore:

- Treat superlinear cells as invalid measurement artifacts.
- Do not quote the worst ITL cells as kernel scaling.
- Even the cleaner cells compare whole-model ITL against a one-layer microbench; see below.

## Microbench caveat

`ci_probe/kernel_bench.py` and `ci_probe/kernel_stage_probe.py` measure one attention invocation with 8 Q heads and `qhead_per_kvhead=1`. The engine runs 32 Q heads with GQA 4:1 across 36 layers, plus store/gather/rotations per layer. Do not compare microbench milliseconds directly to whole-model ITL.

The earlier list of work needed before a performance claim (GQA-aware
shapes, skip-flag A/B, `THUNDER_8B_INDIRECT` comparison, fast-launch
diagnostics) is done and its outcomes are shipped: see `docs/FLAGS.md`
for the defaults and `thunder_vllm/attention/tile_shape.py` /
`splits.py` for the policies they produced. Use `THUNDER_BENCH_HQ=32`
for engine-like shapes.

## Two-stack upstream comparison

Upstream `TURBOQUANT` e2e is not selectable at the pinned vLLM commit for
dense Qwen3-8B (two independent failures: forced-backend rejection on the
boundary layers, and no common KV-cache layout). Do not patch vLLM in the
main benchmark environment. The comparison therefore runs as two stacks:

- upstream: vLLM 0.25.1 stock + `--kv-cache-dtype turboquant_3bit_nc`,
  via `ci_probe/modal_upstream_baseline.py`;
- ours: pinned vLLM + `attention_config={"backend": "CUSTOM"}`,
  via `ci/modal_app.py --mode e2e`.

Both tables, the fp16 cross-stack control that makes them comparable, and
the caveats live in `benchmarks/results/upstream_vs_ours.md`. Quote a
ratio only with that file's provenance notes attached.
