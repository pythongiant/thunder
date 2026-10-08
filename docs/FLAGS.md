# Environment flags

Runtime flags change behavior. Diagnostic flags are for measurement/debug only and must stay off for shipping runs.

## Runtime

- `THUNDER_K_BITS`, `THUNDER_V_BITS`: packed KV widths.
- `THUNDER_ONEPASS=0`, `THUNDER_REG_RESCALE=0`, `THUNDER_CAUSAL_BOUND=0`: opt out of default-on kernel fast paths.
- `THUNDER_8B_INDIRECT=0/1`: force the CSR/indirect gather or the request-major one.
  Unset, the path is a policy (`backend._use_indirect_gather`): the dense gather is
  taken when the request-major reservation would exceed the 24 GiB memory budget or
  `ADDRESSABLE_GATHER_BYTES` (4 GiB -- past that a 32-bit CuTeDSL tensor index
  wraps, `docs/FAILURE_MODES.md` 15), and never while a CUDA graph is capturing
  (`FAILURE_MODES.md` 14). At Qwen3-8B slots that means 4k request-major, 8k/16k/
  32k dense. Setting it explicitly bypasses the policy, which is how the OOM and
  the wrap are reproduced on demand.
- `THUNDER_GQA_PACK=0/1`: GQA-packed schedule (one CTA per KV head scores the whole query group; plan steps 6+8). Default **on**; `0` opts out. Worth -76.6% at batch 16/32k and -31% at batch 1 against the pre-packing schedule (and -60% more at batch 1 once the split count followed the narrower grid), so the old "off until validated" note is spent. Prefill packs too: the M axis carries the group's heads over `tile_m // qhead_per_kvhead` query tokens, so a CTA reconstructs the KV tile once for all of them.
- `THUNDER_STORE3=0/1`: allow 3-bit Triton store packing.
- `THUNDER_FASTLAUNCH=0/1`: reuse cached compiled launch instead of retracing MLIR host path.
- `THUNDER_SPLITS`: force decode split-K count. Otherwise the policy picks it (`thunder_vllm/attention/splits.py`): as fine as the tile budget allows, at most 64, and at most 16 from batch 16 up. Decode-only.
- `THUNDER_ALLOW_ARCH`: sandbox arch override; not for B200 performance claims.
- `THUNDER_SCHEDULE`: default `mma`; `tcgen05` is refused until complete.

## Inert (do not sweep)

- `THUNDER_M_BLOCK`, `THUNDER_N_BLOCK`, `THUNDER_NUM_THREADS`: the engine's tile comes
  from `thunder_vllm/attention/tile_shape.py`, not from these fields, so setting them
  in an engine run changes nothing. The grid harness takes the tile as explicit cell
  fields instead (`<shape>|<splits>|<gqa>|<m_block>|<threads>|<n_block>`).
- `THUNDER_NUM_STAGES`, `THUNDER_NUM_DEQUANT_STAGES`: stored on the kernel and never
  read by it -- the v0 schedule has no circular pipeline, so a depth sweep measures
  noise. (`ci_probe/results/tracka_notes.md` recorded this; the lever table in
  `docs/FAILURE_MODES.md` 18 no longer quotes it as a result.)

## Diagnostic-only

- `THUNDER_STAGE_TIMING=1`: host-dispatch buckets only; no CUDA sync around gather/launch/inverse.
- `THUNDER_TIME_LAUNCH=1`: launch plumbing vs kernel vs merge host timing.
- `THUNDER_DIAG=1`, `THUNDER_FAST_DEBUG=1`, `THUNDER_COUNT=1`: counters and fast-path traces.
- `THUNDER_DEBUG_LAUNCH=1`, `THUNDER_DEBUG_GATHER=1`, `THUNDER_DEBUG_LAYOUT=1`, `THUNDER_DEBUG_BLOCK=1`, `THUNDER_DEBUG_KV=1`.
- `THUNDER_CSR_TRACE=1`, `THUNDER_CSR_DUMP=1`, `THUNDER_CSR_DUMP_PATH`.
- `THUNDER_ENGINE_HOOK=1`, `THUNDER_HOOK_PATH`: first single-request decode capture; may write tensors.
- `THUNDER_VMATRIX=1`, `THUNDER_STORE_AB=1`, `THUNDER_VOL_DIR`: store diagnosis dumps.
- `THUNDER_SKIP_BACKEND=1`, `THUNDER_SKIP_GATHER=1`, `THUNDER_SKIP_KERNEL=1`, `THUNDER_SKIP_ROT=1`: ablation switches; outputs may be wrong.
- `THUNDER_ALLOW_UNADDRESSABLE=1`: skip the gathered-buffer addressability check so
  the 32-bit wrap (`FAILURE_MODES.md` 15) can be reproduced; results are garbage.
- `THUNDER_STORE_TORCH=1`: force the pure-torch KV scatter instead of the Triton one
  (bit-identical, but it syncs on `slot_mapping`). A/B switch for the CUDA-graph
  capture question, `FAILURE_MODES.md` 14.

## Telemetry

- Every profiling/test entry point prints one `[TQ-SYS]` system fingerprint line (machine, torch/CUDA, versions, allowlisted env, git rev).
- Diagnostic dumps additionally emit one `[TQ-TELEMETRY]` JSON record each (`stage`, `count`, `diag`, `launch`, `csr` tags) carrying the same fingerprint plus counters.
- The harness `[PB-JSON]` rows embed the full `sys` fingerprint per result.
- Telemetry never reads non-allowlisted env vars, so tokens/keys cannot leak into logs. See `thunder_vllm/utils/telemetry.py` and `tests/test_telemetry.py`.
