# Environment flags

Runtime flags change behavior. Diagnostic flags are for measurement/debug only and must stay off for shipping runs.

## Runtime

- `THUNDER_K_BITS`, `THUNDER_V_BITS`: packed KV widths.
- `THUNDER_ONEPASS=0`, `THUNDER_REG_RESCALE=0`, `THUNDER_CAUSAL_BOUND=0`: opt out of default-on kernel fast paths.
- `THUNDER_8B_INDIRECT=0/1`: request-major gather vs CSR/indirect path.
- `THUNDER_GQA_PACK=0/1`: GQA-packed decode (one CTA per KV head scores the whole query group; plan steps 6+8). Decode-only, eager-only, default off until validated.
- `THUNDER_STORE3=0/1`: allow 3-bit Triton store packing.
- `THUNDER_FASTLAUNCH=0/1`: reuse cached compiled launch instead of retracing MLIR host path.
- `THUNDER_SPLITS`: force decode split-K count.
- `THUNDER_ALLOW_ARCH`: sandbox arch override; not for B200 performance claims.
- `THUNDER_SCHEDULE`: default `mma`; `tcgen05` is refused until complete.

## Diagnostic-only

- `THUNDER_STAGE_TIMING=1`: host-dispatch buckets only; no CUDA sync around gather/launch/inverse.
- `THUNDER_TIME_LAUNCH=1`: launch plumbing vs kernel vs merge host timing.
- `THUNDER_DIAG=1`, `THUNDER_FAST_DEBUG=1`, `THUNDER_COUNT=1`: counters and fast-path traces.
- `THUNDER_DEBUG_LAUNCH=1`, `THUNDER_DEBUG_GATHER=1`, `THUNDER_DEBUG_LAYOUT=1`, `THUNDER_DEBUG_BLOCK=1`, `THUNDER_DEBUG_KV=1`.
- `THUNDER_CSR_TRACE=1`, `THUNDER_CSR_DUMP=1`, `THUNDER_CSR_DUMP_PATH`.
- `THUNDER_ENGINE_HOOK=1`, `THUNDER_HOOK_PATH`: first single-request decode capture; may write tensors.
- `THUNDER_VMATRIX=1`, `THUNDER_STORE_AB=1`, `THUNDER_VOL_DIR`: store diagnosis dumps.
- `THUNDER_SKIP_BACKEND=1`, `THUNDER_SKIP_GATHER=1`, `THUNDER_SKIP_KERNEL=1`, `THUNDER_SKIP_ROT=1`: ablation switches; outputs may be wrong.

## Telemetry

- Every profiling/test entry point prints one `[TQ-SYS]` system fingerprint line (machine, torch/CUDA, versions, allowlisted env, git rev).
- Diagnostic dumps additionally emit one `[TQ-TELEMETRY]` JSON record each (`stage`, `count`, `diag`, `launch`, `csr` tags) carrying the same fingerprint plus counters.
- The harness `[PB-JSON]` rows embed the full `sys` fingerprint per result.
- Telemetry never reads non-allowlisted env vars, so tokens/keys cannot leak into logs. See `thunder_vllm/utils/telemetry.py` and `tests/test_telemetry.py`.
