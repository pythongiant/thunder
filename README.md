# turboquant-vllm

A vLLM plugin that fuses TurboQuant KV-cache dequantization into a
FlashAttention-4-style SM100/SM110 attention kernel, written in CuTeDSL. The
K/V cache stays packed (2/3/4-bit Lloyd-Max codes plus fp16 norms) all the way
to the tensor cores; the only fp16 KV that exists is the transient per-stage
code ring in shared memory.

> **Status: working v0.** The kernel compiles for `sm_100a` (verified on a cheap
> L4 with `CUTE_DSL_ARCH=sm_100a`) and **passes end-to-end correctness on B200**:
> 8/8 shapes within `atol=rtol=1e-2` (max abs error ~1.5e-3 at 4-bit), covering
> 2/3/4-bit K, head_dim 64/128, causal and non-causal, and non-tile-multiple KV
> lengths. See `ci_probe/results/probe_summary.md`. The v0 schedule is a
> **cooperative, two-pass, `mma.sync`** design; the tcgen05/TMEM warp-specialized
> single-pass schedule is the next step, so the benchmark tables are still
> placeholders (the kernel has no split-K or GQA K reuse yet). Everything else --
> cache layout, quantizer, paged-KV gather, scratch management, metadata builder,
> backend registration, tests, benchmarks and Modal CI -- is implemented and the
> CPU test-suite is green.

---

## Install

Requirements:

* **GPU**: SM100 (B200) or SM110. The kernel asserts the arch in `__init__` and
  will refuse to construct elsewhere.
* **CUDA**: 12.8+ toolkit (13.0 works). `nvidia-cutlass-dsl` ships CUDA 12 libs
  by default; add the `[cu13]` extra if you are on CUDA 13.
* **vLLM**: `main`, pinned. The backend API this plugin implements
  (`AttentionMetadataBuilder`, `AttentionBackendEnum.CUSTOM`) and the upstream
  `turboquant_attn.py` reference only exist on `main`; the latest tag does not
  carry them. See `ci/modal_image.py` for the validated commit.
* **flash-attention**: `main`, pinned, for the CuTeDSL host utilities and
  `pack_gqa`.

```bash
pip install -e .
pip install -e ".[test,bench]"
```

Registering the backend happens through the entry point declared in
`pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
turboquant_cute = "turboquant_vllm.model.registry:register"
```

Select it at runtime:

```bash
vllm serve <model> --attention-backend CUSTOM \
  --kv-cache-dtype turboquant_cute
# optional knobs (defaults shown)
export TURBOQUANT_K_BITS=4
export TURBOQUANT_V_BITS=4
```

`CUSTOM` is used rather than `TURBOQUANT` because upstream vLLM already
registers a backend under `TURBOQUANT`, and we keep it intact to benchmark
against (see the parity caveat in the benchmark section).

---

## Why it wins

Decode is bandwidth-bound: at batch 1 the attention kernel reads the entire KV
cache for a handful of FLOPs, so the only thing that matters is bytes moved per
token. TurboQuant cuts those bytes ~4x (4-bit K and V plus two fp16 norms per
head, instead of two fp16 vectors). The conventional way to use that
compression is a dequant-to-fp16 kernel followed by a normal attention kernel,
which pays for the extra pass and materializes fp16 KV in HBM. This kernel
instead dequantizes **inside** the attention pipeline: a dedicated dequant warp
unpacks the bit stream and gathers from a per-layer Lloyd-Max LUT directly into
an SMEM fp16 code ring that the tcgen05 MMA consumes, overlapping with the MMA
of the previous tile. Q is pre-rotated by the same Hadamard matrix at the model
level, so the kernel never sees an unrotated Q: the QK MMA runs on rotated Q and
rotated K codes, with `k_norm` applied after the MMA and `v_norm` folded into
the softmax probabilities before the PV MMA. The result is the same attention
output as an fp16 pipeline, at a fraction of the KV bytes, and therefore the
same accuracy with materially more tokens resident in HBM.

---

## Cache layout contract

```
packed K  : uint8, (num_blocks, block_size, num_kv_heads, k_packed_bytes)
packed V  : uint8, (num_blocks, block_size, num_kv_heads, v_packed_bytes)
k_norm    : fp16,  (num_blocks, block_size, num_kv_heads)
v_norm    : fp16,  (num_blocks, block_size, num_kv_heads)
k_lut     : fp16,  (2**K_BITS, head_dim)
v_lut     : fp16,  (2**V_BITS, head_dim)
```

where

```
k_packed_bytes = ceil(head_dim * K_BITS / 8)
v_packed_bytes = ceil(head_dim * V_BITS / 8)
```

Q is pre-rotated at the model level by the same Hadamard/orthogonal matrix
applied to K/V before quantization.

**Allocation.** `allocate_kv_cache(...)` returns `(kv_cache, kv_scales)`:

```
kv_cache  : uint8, (num_blocks, block_size, kv_slot_bytes)
kv_scales : fp16,  (num_blocks, block_size, num_kv_heads, 2)

kv_slot_bytes = num_kv_heads * (k_packed_bytes + v_packed_bytes)
              = [ K codes: Hk * k_packed_bytes | V codes: Hk * v_packed_bytes ]
```

K and V codes share one allocation so the gather reads one strided region per
slot; `TurboQuantCacheLayout.k_codes/v_codes/views` produce zero-copy views in
exactly the four-tensor shapes above, and `kv_scales[..., 0/1]` are `k_norm` /
`v_norm`. The slot resolves as `slot = block_id * block_size + offset` with no
extra indirection, so the scheduler's `slot_mapping` addresses a byte row
directly. `head_dim_padded` rounds `head_dim` up to a multiple of 16: packed
bytes are computed against the true `head_dim` (compression is unaffected), while
code tiles and LUT rows are padded to `head_dim_padded`.

---

## Ablation table

Measured at `batch=1, seqlen_k=16384, nheads=32, num_kv_heads=8, head_dim=128,
causal=True`. Each row is a distinct compilation (the kernel takes a
`VARIANT_TAG` constexpr), so the ablation cannot be confounded by cache reuse.
**Placeholder values:** the kernel schedule is incomplete, so these rows are not
yet populated; the table records what each row is designed to isolate.

| Variant | Build | What to report | Latency (ms) | TFLOPs/s | Bandwidth (GB/s) |
|---|---|---|---|---|---|
| TurboQuant + FP16 dequant | reference path (dequant to fp16 in a separate kernel, then FA4-style attention on fp16) | latency, TFLOPs, bandwidth | _pending_ | _pending_ | _pending_ |
| Fused K only | `V_BITS=16` (identity LUT), reference V outside the kernel | K-side bytes | _pending_ | _pending_ | _pending_ |
| Fused K + fused V | the kernel as written | full KV bytes | _pending_ | _pending_ | _pending_ |
| + K reuse | `HEADS_PER_GROUP = N_REP` vs `= 1` | SMEM K traffic | _pending_ | _pending_ | _pending_ |
| + V multi-head GEMM | single PV vs `N_REP` separate PVs | SMEM V traffic | _pending_ | _pending_ | _pending_ |
| + split-K | `num_splits = 1` vs `_choose_num_splits` | SM occupancy at batch 1 | _pending_ | _pending_ | _pending_ |
| + 3/2/4-bit packing | `(K_BITS, V_BITS) ∈ {(2,2), (3,3), (4,4), (3,4)}` | bytes/token | _pending_ | _pending_ | _pending_ |

Generated by `tests/test_ablation.py` into `benchmarks/results/ablation.md`.

---

## vs FA4: expected performance

Kernel-level comparison against `flash_attn.cute.flash_fwd_sm100.FlashAttentionForwardSm100`
with a dense fp16 KV cache built alongside the packed cache. Shape:
`batch ∈ {1,8,32}, nheads=32, num_kv_heads=8, head_dim=128, causal=True`.

Expectations, stated plainly:

* **Effective bandwidth**: we win at every sequence length. FA4 reads
  `2 * seqlen * Hk * D * 2` bytes; we read `seqlen * Hk * (k_packed_bytes +
  v_packed_bytes + 4)`. At 4/4-bit that is roughly a 3.9x reduction.
* **Memory capacity**: we win at every sequence length. At 4/4-bit and
  `Hk=8, D=128`, one 40 GB budget holds ~4x more tokens than fp16 KV.
* **Latency**: we win at ≥16k context, where the KV read dominates. We are
  expected to **lose** on latency at short context (1k–8k), because the dequant
  stage and the LUT gather are pure overhead when there are few bytes to read.
  This is not a bug and should not be hidden.
* **TFLOPs**: we lose at short context and approach parity at long context. The
  kernel is bandwidth-bound by design; TFLOPs is the wrong headline metric for
  decode and is reported only for completeness.

| batch | seqlen | FA4 latency | ours latency | TFLOPs ours | TFLOPs FA4 | eff BW ours | BW FA4 | lat ours/fa4 | tokens/GB ours | tokens/GB fp16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 1 | 16384 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 1 | 131072 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 8 | 32768 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 32 | 8192 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

FA4 is skipped at 65536/131072 for the larger batches when it OOMs; the OOM is
recorded in `benchmarks/results/vs_fa4.md` rather than silently dropped. Our
kernel is expected to run at 131072 with headroom.

Generated by `benchmarks/bench_vs_fa4.py` into `benchmarks/results/vs_fa4.md`.

---

## vs TurboQuant-vLLM: expected performance

End-to-end against vLLM's upstream `TURBOQUANT` backend, same model loaded twice.
Targets from the acceptance criteria: **≥1.5x** on decode-heavy workloads and
**≥1.15x** on mixed and long-prefill. Sweep:
short-decode, mixed, decode-heavy, high-load, long-prefill, very-long-prefill.

**Parity caveat.** The two backends use different KV layouts -- upstream stores
packed K with **fp16 V**, centroids and no per-head norms; this plugin stores
packed K **and packed V** plus fp16 norms. End-to-end logit parity is therefore
not a meaningful assertion. The harness instead feeds both paths the *same*
quantized tensors and compares attention outputs with `atol=rtol=1e-2`, which
isolates kernel correctness from quantizer semantics. Any end-to-end quality
comparison must first agree on the quantizer.

| Workload | batch | prompt_len | gen_len | baseline tok/s | ours tok/s | ours % | target |
|---|---|---|---|---|---|---|---|
| short-decode | 1 | 128 | 512 | _pending_ | _pending_ | _pending_ | – |
| mixed | 1 | 512 | 512 | _pending_ | _pending_ | _pending_ | ≥115% |
| decode-heavy | 1 | 64 | 1024 | _pending_ | _pending_ | _pending_ | ≥150% |
| high-load | 500 | 512 | 128 | _pending_ | _pending_ | _pending_ | – |
| long-prefill | 1 | 4096 | 128 | _pending_ | _pending_ | _pending_ | ≥115% |
| very-long-prefill | 1 | 8192 | 64 | _pending_ | _pending_ | _pending_ | ≥115% |

Generated by `benchmarks/bench_vs_turboquant_vllm.py` into
`benchmarks/results/vs_turboquant_vllm.md`.

---

## Known limitations

1. **v0 schedule is cooperative, two-pass, `mma.sync`.** The kernel is correct
   (8/8 B200 shapes within 1e-2; see `ci_probe/results/probe_summary.md`) but not
   yet fast. It does not use tcgen05/TMEM, is not warp-specialized, and makes
   **two passes** over the KV tiles (pass 1 row max; pass 2 `exp2` + PV) so the
   fp32 accumulator never needs an online rescale. That costs one extra K read
   and one extra QK per tile.
2. **tcgen05/TMEM: sync fixed, operand K-order mismatch remains (path A).**
   `attention/cute_kernel_tcgen05.py` (opt-in via `TURBOQUANT_SCHEDULE`)
   follows the canonical NVIDIA Blackwell GEMM and now **runs to completion on
   B200** — the completion deadlock was fixed by using
   `PipelineUmmaAsync.create(...).make_participants()`. The deposit is proven
   self-consistent (see `ci_probe/modal_probe_deposit.py`), but the **MMA reads
   the operands from different addresses than their layout describes**: sparse
   identity/one-hot patterns look correct, while dense A/B returns wrong values
   (and with a single k-block, all zeros). Tried and failed: native
   nested-coordinate writes, rank-4 staging + `make_tiled_copy_A/B` blit, plain
   unswizzled operands, `byte_alignment` 128, FA's vendored PTX descriptor path,
   `make_trivial_tiled_mma`, `M=64`. Fixing it needs FA's `mma()` replicated in
   full, or TMA-filled operands. The backend refuses to select the schedule until
   then; details and the reproducing harness are in
   `ci_probe/results/probe_summary.md`.
3. **`ldmatrix` is not used.** v0 does plain 16-bit universal SMEM-to-register
   copies: `ldmatrix` needs 128-bit-aligned sources and a swizzle-compatible
   SMEM layout, which the row-major v0 tiles do not provide. Swizzled layouts
   plus `ldmatrix` is a pure performance win.
4. **`head_dim` must be a multiple of 16** (64/128/256), so the padded and true
   head dims coincide and no Q zero-padding is on the hot path.
5. **`PagedKVManager` gather is a first-version simplification.** The gather is a
   fixed-shape, pre-allocated, graph-recordable copy into contiguous buffers,
   not an in-kernel block-table walk. FA4's
   `flash_attn/cute/paged_kv.py::PagedKVManager` shows the intended end state:
   TMA atoms that walk the block table in-kernel, eliminating a full extra pass
   over the KV cache. This is the primary follow-up, and
   `PagedKVManager.block_table_row_map` exists so the rewrite consumes exactly
   the mapping it needs.
6. **Correction warp / split-K / GQA K reuse / V multi-head GEMM not wired.**
   The `VARIANT_TAG` constexprs exist for the ablation, but only the fused-KV
   variant is implemented. `num_splits` is always 1, K is re-dequantized per Q
   tile, and PV is one GEMM per tile rather than `N_REP` separate PVs. These are
   the four optimisations the ablation table is designed to measure.
7. **SM100/SM110 only.** The kernel asserts the arch family in `__init__`.
   H100/SM90 CI exists only for the quantizer and cache-layout tests.
8. **No ALiBi, no sliding window, no logits soft cap.** The impl raises
   `NotImplementedError` for these at construction rather than silently
   producing wrong output.
9. **2-CTA MMAs are rejected.** The coop schedule raises `NotImplementedError`
   for `use_2cta_instrs`; there is no validated 2-CTA configuration yet.
10. **`kv_cache_dtype` is not a real vLLM preset.** vLLM's `CacheDType` is a
   `Literal` a plugin cannot extend, so bit widths come from
   `TURBOQUANT_K_BITS`/`TURBOQUANT_V_BITS` (or `TurboQuantCuteConfig`), while the
   cache dtype string is accepted for compatibility only.

---

## How to benchmark

```bash
# 1. correctness gate -- never report speed on a failing run
python -m pytest tests/ -q

# 2. kernel vs FA4 (writes benchmarks/results/vs_fa4.md)
python -m benchmarks.bench_vs_fa4 --out benchmarks/results/vs_fa4.md

# 3. plugin vs upstream vLLM TurboQuant (writes .../vs_turboquant_vllm.md)
python -m benchmarks.bench_vs_turboquant_vllm \
  --model Qwen/Qwen3-4B --out benchmarks/results/vs_turboquant_vllm.md

# 4. ablation (writes .../ablation.md)
python -m pytest tests/test_ablation.py -s

# 5. everything, with optional ncu profiling
TURBOQUANT_KERNEL_ENABLE=1 bash benchmarks/run_all.sh
PROFILE=1 TURBOQUANT_KERNEL_ENABLE=1 bash benchmarks/run_all.sh
```

`--profile` / `PROFILE=1` runs Nsight Compute with this metric set and writes
`benchmarks/results/ncu_<variant>_<shape>.csv`:

| metric | what it tells you |
|---|---|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | tensor-core utilisation: is the MMA the bottleneck? |
| `dram__bytes_read.sum` | total HBM read -- the number TurboQuant exists to minimise |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | SMEM bank conflicts: is the LUT gather thrashing? |
| `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | how much the dequant warp blocks |
| `sm__tmem_alloc_cols` | TMEM column pressure |

High dequant stall ⇒ the dequant pipeline is the bottleneck. Low tensor-core
utilisation **with** high DRAM read ⇒ bandwidth-bound, which is the desired state
for long-context decode. `benchmarks/ncu_summary.py` writes this verdict up.

---

## How to run CI locally

CI runs on Modal because GitHub-hosted runners cannot run SM100 kernels.

```bash
# One-time
pip install modal
modal token set --token-id <id> --token-secret <secret>

# Cheap compile probe (L4, sm_100a cross-compile) -- no B200 spend
modal run ci_probe/modal_probe_kernel.py --head-dim 128 --k-bits 4 --v-bits 4 --gqa 1

# Correctness on B200
modal run ci/modal_app.py --mode correctness

# Smoke benchmark on B200 (gates PRs)
modal run ci/modal_app.py --mode smoke

# Pull results
modal volume get turboquant-ci-results / ./benchmarks/results/
```

The GitHub workflow is `.github/workflows/modal_ci.yml`: a CPU test job on the
runner, then `correctness` on B200, then `smoke-bench`, plus the cheap
`compile-probe`. Add `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as repository
secrets.

### CI cost

| job | tier | GPU time per PR |
|---|---|---|
| CPU tests | GitHub runner | 0 GPU-s |
| compile probe | L4 | ~2 min |
| correctness | B200 | ~15 min |
| smoke bench | B200 | ~10 min |
| nightly full sweep | B200 | ~2 h |

The CuTeDSL cache volume (`turboquant-cute-cache`, mounted at
`~/.cache/cutlass` and `~/.cache/cute`) cuts per-PR compile time from ~5 min to
~30 s after the first run against a given commit. Modal bills per second, so a
cancelled PR job stops charging within a few seconds. The image is pinned by
CUDA/torch/CUTLASS/FA/vLLM versions; rebuild only when those pins change.

### What runs where

| Test | Local (CPU) | Modal B200 | Modal H100 |
|---|---|---|---|
| `test_cache_layout` (layout, packing, codebook, quantizer round-trip) | ✓ | ✓ | ✓ |
| `test_paged_kv` (gather vs. block-table walk) | ✓ | ✓ | ✓ |
| `test_attention_correctness` | ✗ | ✓ | ✗ (SM100 assert) |
| `test_cuda_graph_replay` | ✗ | ✓ | ✗ |
| `test_vs_turboquant_vllm_parity` | ✗ | ✓ | ✗ |
| Smoke bench | ✗ | ✓ | ✗ |
| Compile probe (`ci_probe/`) | ✗ | ✓ | ✓ (cross-compiles on L4) |
| Full bench sweep (nightly) | ✗ | ✓ | ✗ |

Local runs are limited to CPU-side correctness (cache layout math, block-table
construction, LUT generation, bit packing). Anything that touches the kernel or
the tensor cores runs on Modal.

---

## Package layout

```
turboquant_vllm/
├── pyproject.toml            # entry point: vllm.general_plugins
├── setup.py
├── README.md
├── turboquant_vllm/
│   ├── attention/
│   │   ├── backend.py        # TurboQuantAttentionBackend + Impl
│   │   ├── metadata.py       # AttentionMetadataBuilder subclass
│   │   ├── cache_layout.py   # layout, allocation, reshape_and_cache
│   │   ├── paged_kv.py       # PagedKVManager gather layer
│   │   ├── scratch.py        # graph-safe reserve/alloc scratch pools
│   │   └── cute_kernel.py    # the CuTeDSL kernel
│   ├── quant/
│   │   ├── lloyd_max.py      # codebook + LUT builder
│   │   ├── hadamard.py       # Hadamard/orthogonal rotation
│   │   ├── packing.py        # little-endian bit packing
│   │   └── quantizer.py      # per-layer online quantizer
│   ├── model/registry.py     # vLLM registration
│   └── utils/logging.py
├── tests/
├── benchmarks/
├── ci/                       # Modal app/image/correctness/smoke
└── ci_probe/                 # cheap compile probes + results
```

## License

Apache-2.0.
