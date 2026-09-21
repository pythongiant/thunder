# Track A — vLLM integration notes (2026-09-17)

## Image: now builds (was never buildable)

`ci/modal_image.py` fixes, in the order each one bit:
1. `wheel` missing -> FA editable install died in
   `prepare_metadata_for_build_editable` ("No module named 'wheel'").
2. `setuptools` 68.1.2 too old -> vLLM's PEP-639 `license` string rejected
   ("'project.license' must be valid exactly by one definition"). Pinned
   `setuptools>=77`.
3. `setuptools_rust` and `setuptools_scm` missing -> vLLM metadata generation
   died. Added both, plus `cmake`/`ninja`, and `python -m pip install -U pip`.
4. FA source build takes ~1h and is NOT needed for vLLM. Replaced with a
   prebuilt `flash-attn` wheel; the pinned FA source tree is still cloned to
   /opt/flash-attention for the vs-FA4 benchmark but is not installed.
5. vLLM installed with `VLLM_USE_PRECOMPILED=1` (no 20-min kernel build).

Result: `vllm 0.29.1rc1.dev159+gdffbb714e` (the pin) + `thunder_vllm` install
cleanly. Image is cached; rebuild is ~6 s.

## Registry / API facts (measured in-container)

* `AttentionBackendEnum` **already contains `TURBOQUANT`** at this pin, and also
  `CUSTOM`. So upstream ships a TurboQuant backend and the vs-upstream comparison
  is directly available.
* `register_backend(CUSTOM, "thunder_vllm.attention.backend.ThunderAttentionBackend")`
  **succeeds**.
* BUT the registry keys on the *enum* member: resolving `"THUNDER_CUTE"` fails
  ("Unknown attention backend"), because there is no such enum member. The
  selectable name after registering on CUSTOM is **`CUSTOM`**. `get_name()`
  returning "THUNDER_CUTE" is therefore not the selection key.
  (The local probe also tried `_ATTN_BACKEND_REGISTRY`, which does not exist in
  this version -- that was my error, not vLLM's.)
* `VLLM_ATTENTION_BACKEND` is **no longer recognised** ("Unknown vLLM environment
  variable detected"). This vLLM selects attention via `attention_config` /
  `--attention-backend`. Any integration doc/env in the repo referencing
  VLLM_ATTENTION_BACKEND is stale.
* Backend surface is healthy:
  name THUNDER_CUTE, head sizes [64, 128, 256], impl + builder resolve,
  kv_cache_dtypes ['thunder_cute', 'thunder_k8v4', 'thunder_k3v4_nc',
  'thunder_4bit_nc', 'thunder_3bit_nc'].

## Engine run: fails on a torch/flash-attn ABI mismatch

The container ends up with **torch 2.13.0+cu130 / CUDA 13.0**, not the pinned
torch 2.8.0, because vLLM 0.29.1rc1's dependency resolution upgrades torch. The
prebuilt flash-attn wheel is built against torch 2.8, so vLLM's default backend
path dies on import:

  ImportError: flash_attn_2_cuda...so: undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

That is the *only* remaining blocker for the first engine run, and it is a
dependency choice, not our code.

## Next for Track A

1. Pin torch to whatever this vLLM commit requires and install the matching FA
   wheel (or select a non-FA backend), so the engine core starts.
2. Select our backend as `CUSTOM` via `attention_config`/`--attention-backend`
   (not VLLM_ATTENTION_BACKEND, not "THUNDER_CUTE").
3. Then the correctness matrix (single / short decode / long decode / prefill),
   decode concurrency scaling, memory / max supported context.
4. Compare against upstream `TURBOQUANT` (now known to exist at this pin).


## Engine integration chain (each link found by a B200 run, all in OUR code)

With torch 2.13 + vLLM bundled FA (FA2/FA3/FA4 all AVAILABLE) and the external
flash_attn wheel removed, `--attention-backend CUSTOM` selection now works and the
engine walks into our backend. The sequence of rejections, in order:

1. `platforms/cuda.py: ValueError: Selected backend AttentionBackendEnum.CUSTOM is
   not valid ... Reason: ['block_size not supported']`
   -> `supports_block_size(None)` returned False because vLLM asks with None to
   mean "backend decides". FIXED (accept None) + instrumented via
   THUNDER_DEBUG_BLOCK.
2. `registry.py: ValueError: Unknown attention backend: 'THUNDER_CUTE'`
   -> vLLM resolves the backend name through AttentionBackendEnum, whose key is
   the ENUM MEMBER, so `get_name()` must return the registry key. FIXED:
   `get_name()` returns REGISTRY_NAME = "CUSTOM"; "THUNDER_CUTE" stays the
   display name only.
3. `attention.py:728 assert hasattr(impl, "do_kv_cache_update")` ->
   `AssertionError: ThunderAttentionImpl does not support kv cache update`
   FIXED: implemented `do_kv_cache_update` on the Impl (same reshape_and_cache the
   forward path used to do inline), with argument discovery by role
   (cache = largest tensor, slot mapping = 1-D int, then key/value) and
   THUNDER_DEBUG_KV logging. The call DID arrive and the discovery worked.
4. `backend.py:650 RuntimeError: TurboQuant norms buffer not bound; call
   bind_scales()` -> nothing calls bind_scales because vLLM allocates the cache
   itself. FIXED: `_scales_for` now lazily allocates the norms buffer from
   `layout.get_scales_shape(num_blocks)`. NOTE: that side buffer is not counted by
   vLLM's cache accounting; the tidy fix is a view into the norms region of the
   packed slot.
5. CURRENT BLOCKER: `cache_layout.py:173 k_codes`: `nb, bs, _ = kv_cache.shape`
   -> `ValueError: too many values to unpack (expected 3)`. vLLM hands the backend
   a cache tensor whose RANK differs from the 3-D `(num_blocks, block_size,
   kv_slot_bytes)` our `get_kv_cache_shape` declares (vLLM conventionally
   allocates with a leading k/v dim and passes slices). NEXT FIX: align
   `get_kv_cache_shape` with vLLM's convention, or make k_codes/v_codes tolerant
   of the extra leading dim.

Harness bugs I introduced and fixed along the way (not plugin bugs): the smoke ran
the engine at module level, which breaks vLLM's spawn-based worker ("An attempt has
been made to start a new process before the current process has finished its
bootstrapping phase"); it is now under `if __name__ == "__main__":`.

Also note: `LLM(..., attention_config={"backend": "CUSTOM"})` is the working
selection API (`AttentionConfig(backend=<AttentionBackendEnum.CUSTOM: None>)` is
accepted). `attention_config` is the only attention kwarg in this version's
`LLM.__init__` signature; `VLLM_ATTENTION_BACKEND` is dead.


## Link 5 DIAGNOSED (exact call chain)

The rank-4 cache is vLLM's own canonical KV-cache layout, and it arrives during the
KV-cache WARM-UP, not during a normal step:

  core.py:145   EngineCore.__init__
  core.py:356   _initialize_kv_caches(vllm_config)
                  -> model_executor.compile_or_warm_up_model()
  executor/abstract.py:126  collective_rpc(...)          # warm-up forward
  backend.py:544            ThunderAttentionImpl.forward
  paged_kv.py:162/205       gather_packed_tiles -> gather_packed_tiles_ref
  cache_layout.py:173       k_codes: nb, bs, _ = kv_cache.shape
  -> ValueError: too many values to unpack (expected 3)

So the failing object is the cache vLLM allocated for the warm-up, and it has rank 4.
Our `get_kv_cache_shape` declares 3-D `(num_blocks, block_size, kv_slot_bytes)` and
`k_codes`/`v_codes` assume that same 3-D form.

Given the correction that V1 views the raw allocation into exactly the shape the
backend declares, the resolution is NOT a tolerant `nb, bs, *_ =` parse (that would
silently reinterpret the head/position dims). It is to stop pretending the cache is
a 3-D byte-slot tensor and use the convention the framework and upstream TurboQuant
both use:

    (num_blocks, num_kv_heads, block_size, slot_bytes)      # combined K+V, no leading 2

Then transpose ONCE at the boundary for kernels that want `(nb, bs, Hk, packed)`,
exactly as upstream TurboQuant transposes `(B,H,N,C) -> (B,N,H,C)` before its
kernels.

### Bounded refactor this implies
  * `ThunderCacheLayout.get_kv_cache_shape` -> 4-D, and
    `get_scales_shape` re-derived to match (norms ride in the same slot region).
  * `k_codes` / `v_codes` -> consume the 4-D cache, return the kernels' layout
    (transpose at this single boundary).
  * `paged_kv.gather_packed_tiles_ref` and `reshape_and_cache` -> same boundary.
  * `ThunderAttentionBackend.get_kv_cache_shape` -> return the 4-D shape.
  * CPU tests in tests/test_cache_layout.py and test_paged_kv.py must be updated to
    the new convention (they currently encode the 3-D one).

### Harness note
My layout DEBUG logging did not appear because the spawned EngineCore process does
not inherit the driver process's `logging.basicConfig`; to see those lines the
logging config has to be applied inside the engine process (e.g. via an env-driven
setup at plugin import time in `thunder_vllm.model.registry`).


## Link 5 FIXED — 4-D canonical cache layout

`ThunderCacheLayout` now uses vLLM's canonical combined-K+V layout:

    (num_blocks, num_kv_heads, block_size, head_slot_bytes)
    head_slot_bytes = k_packed_bytes + v_packed_bytes      # per head, K then V
    kv_scales: (num_blocks, num_kv_heads, block_size, 2)

Changes:
  * `get_kv_cache_shape` / `get_scales_shape` -> 4-D; `ThunderAttentionBackend
    .get_kv_cache_shape` delegates, so it follows.
  * `k_codes` / `v_codes` -> slice the head slot and `permute(0, 2, 1, 3)` to the
    kernels' `(nb, bs, Hk, packed)` order. This is THE single transpose boundary.
  * new `k_norm` / `v_norm` accessors -> `(nb, bs, Hk)` via `permute(0, 2, 1)`.
  * `views()` returns those.
  * `reshape_and_cache_ref` writes per-head slots: `k_view[block_id, offset]`,
    `kv_scales[block_id, :, offset, 0/1]`.
  * Triton kernel: added `stride_cache_head` and uses it in both address
    computations; V is now at `+ k_packed_bytes` inside the head slot instead of
    `+ k_region_bytes`; the call site passes `kv_cache.stride(1)/(2)`.
  * `paged_kv.gather_packed_tiles_ref` uses the norm accessors.
  * `tests/test_cache_layout.py` updated to the 4-D convention.

Verification: full CPU suite `37 passed, 8 skipped` (same as baseline).

## Link 6 (current): gather reshape during vLLM's KV-cache warm-up

The rank error is gone; the engine now reaches our gather and fails there:

  backend.py:544 forward
  paged_kv.py:162 gather_packed_tiles
  paged_kv.py:222 gather_packed_tiles_ref -> v_packed = v.reshape(...)
  RuntimeError: shape '[32768, 16, 8, 64]' is invalid for input of size 1879048192

`max_page_rows` = 32768 (= max_num_reqs * max_blocks_per_req as configured by the
Impl from attn_metadata) does not match the actual `v` element count under vLLM's
warm-up, i.e. R*B*bs != max_page_rows*bs, so the fixed-target reshape is wrong. The
previous 3-D path never got this far. NEXT: derive the gathered shapes from the
actual block table (R, B) instead of a fixed max_page_rows, or make the Impl
construct the paged manager with sizes that match the metadata it is handed.


## Link 6 FIXED — and it was never a reshape bug

Two real bugs, both in `customize_spec`:

1. It DEFERRED when vLLM had already set `state_content_bytes`:
       if getattr(spec, "state_content_bytes", None) is not None: return spec
   vLLM pre-populates that field with the standard fp16 K+V size, so our packed
   sizing never applied and the engine allocated a STANDARD cache. Measured:
       cache=(30582, 8, 16, 512)     # 512 B per (block, head, position) = 2*128*2
   v_codes then read 512-64 = 448 bytes instead of 64 -- a 7x over-read, which is
   exactly the factor the gather reshape complained about. The reshape was innocent.

2. The value convention. It must OVERRIDE, and the semantics are: vLLM consumes
   `state_content_bytes` as the per-position COMBINED K+V byte count and halves it
   for the cache's per-(block, head) inner dim. Measured directly: passing 1056
   gave inner dim 528 = 1056/2. Our head slot is `head_slot_bytes` = k_packed +
   v_packed, so the correct value is `2 * head_slot_bytes` = 256 (the fp16 norms
   live in the plugin's own paired buffer, not in this slot).

RESULT:

    cache=(122330, 8, 16, 128)   cache_stride=(16384, 128, 1024, 1)
    k_codes=(122330, 16, 8, 64)
    v_codes=(122330, 16, 8, 64)

Inner dim exactly 128 = k_packed + v_packed, views correct, 7x over-read gone.
BONUS, from the same line: block count 29,655 -> 122,330, i.e. ~4x more cached
tokens in the same memory. The capacity claim is now observable rather than
asserted.

## Link 7 (current): CUDA_ERROR_ILLEGAL_ADDRESS at kernel launch

The framework ABI is now fully cleared and the engine REACHES KERNEL EXECUTION:

    run_compiled_program -> cutlass.base_dsl.common.DSLCudaRuntimeError:
    error: CUDA_ERROR_ILLEGAL_ADDRESS (error code: 700)

during vLLM's warm-up forward. Next: validate the launch path under vLLM's
metadata -- the launcher dereferences seq_lens / query_start_loc / block table, and
the warm-up metadata does not necessarily carry the same fields or dtypes our
ThunderMetadata provides (e.g. absent q_start -> garbage pointer). Add shape and
dtype assertions in ThunderAttentionImpl.forward before the launch so the next
failure is a clear message rather than an illegal address.

CPU suite after all of the above: 38 passed, 8 skipped.


## Link 7 DIAGNOSED — metadata is valid; the kernel has no per-request KV row offset

Instrumented launch (all fields present, correct dtype, CUDA):

    [TQ-LAUNCH] q=(16384, 16, 128)/float16 n=16384 kv=(122330, 8, 16, 128)
                bt=(1024, 32) sl=(1024,) qsl=(1025,) slot=(16384,)
                num_reqs=1024 max_blocks_per_req=32 is_prefill=True
                max_query_len=16 num_heads=16 Hk=8 head_size=128

So the illegal address is not bad metadata and not an ABI problem. It is inside our
own kernel, on vLLM's warm-up batch: **1024 requests**, 16 q tokens each, 32 blocks
per request.

And here is the structural bug it exposes:

  `_load_kv_packed` reads rows `nt * tile_n + row` of `mK`/`mV` -- i.e. it assumes
  the gathered KV for the request starts at row 0. The `req` grid index (block_idx
  [2]) is only used to fetch seq_lens/query_start_loc; NOTHING adds a per-request
  KV row base. The gather, meanwhile, emits rows in `(req, block)` order, i.e.
  row = req * max_blocks_per_req + block.

  With more than one request the kernel therefore reads request 0's KV rows for
  every request. This has never been exercised, because EVERY test in this project
  used exactly one request:
    * ci_probe/modal_probe_exec.py: block_table built as `.reshape(1, nb)` and
      `make_paged_kv_manager(..., max_num_reqs=1, ...)` -- so the "8/8 shapes
      verified on B200" result is a single-request result.
    * Phase-1 smoke (decode-short/long, prefill): batch = 1.
  vLLM's warm-up is the first genuinely multi-request invocation the kernel has
  ever had.

That also retro-explains the Phase-1 table: those numbers (1.70x / 0.52x / 0.02x)
are batch-1 numbers.

NEXT (bounded, and reproducible without vLLM):
  1. Drive ci_probe/modal_probe_exec.py (or a variant) with >1 request --
     block_table (R, B), max_num_reqs = R -- and confirm the same failure / wrong
     results in a controlled harness.
  2. Fix the row base: pass a per-request KV row base into `_load_kv_packed`
     (`req * max_blocks_per_req`) for both K and V and for the norms, and make the
     launcher/Impl agree on where that base lives (metadata field or a kernel
     argument derived from the gathered tensor's dimensions).
  3. Re-verify the multi-request case in the exec probe BEFORE returning to vLLM,
     then rerun the engine smoke.


## Multi-request reproduction: case A CONFIRMED, no OOB (answers A/B/C)

`ci_probe/modal_probe_multireq.py`, framework-independent, R requests x B blocks,
unique per-(req, block) V sentinels, identity LUTs, seq_len=1 so out == the V value
the kernel actually read.

BEFORE the fix:
  [A]      R=2  B=2  seq=1     [1.0, 1.0]                     expected [1.0, 3.0]
  [R4B4]   R=4  B=4  seq=32    [1.5, 1.5, 1.5, 1.5]
  [R8B8]   R=8  B=8  seq=64    [2.5 x 8]
  [R16B16] R=16 B=16 seq=128   [4.5 x 16]
  control  R=1  B=4  seq=32    [1.5]   <- single request is correct
1.5/2.5/4.5 are exactly the means of request 0's own block ranges, i.e. every
request read request 0's gathered rows. And NO illegal address at any size (up to
R=16/B=16/seq=128).

=> A (wrong request's KV), NOT B (out-of-bounds), NOT C. The illegal address vLLM
   hits is therefore a SECOND, separate bug.

## Fix applied (layout contract, request-major base)

  gather (unchanged): row = req * max_blocks_per_req + block
  kernel (new):       base = req * kv_row_stride; row = base + nt*tile_n + row

  * `_load_kv_packed` takes `req` and `kv_row_stride` and applies `base` to the K
    read, the V read, and BOTH norm reads.
  * `ThunderAttentionForward.kernel` takes a runtime `kv_row_stride: Int32`
    (runtime, not constexpr, so it does not add a compile per stride).
  * `launch_thunder_attention` derives it from `metadata.max_blocks_per_req`,
    falling back to `page_rows // num_reqs` (the fallback is only right when the
    gather did not pad up to max_num_reqs).

AFTER the fix (partial):
  [R4B4]   [1.5, 1.75, 2.0, 2.25]   <- now request-dependent, so the base reaches
                                        the kernel
  [A]      [1.0, 1.0]               <- STILL aliased for seq_len=1

So one more indexing discrepancy remains. Note the probe's own expectations for
seq_len>1 are wrong: out is an attention-weighted MEAN over the valid rows, so only
the seq_len=1 case is a clean assertion (out == V[base + 0]). That case still reads
request 0's row, so either the base is not what the kernel receives, or a second
place indexes the gathered buffer without it.

NEXT: instrument the kernel's `base` value (and the launcher's `kv_row_stride`) for
the seq_len=1 case, then check every remaining consumer of the gathered buffer
(the softmax mask's `kv` index is relative and fine; the norms and the V read share
`base`; look for any other absolute row use).


## PROOF for the failing case, and a correction to the attribution

Added a debug tensor (`mDbg`) the kernel writes: `req`, `kv_row_stride`, `req_base`.
The launcher passes a small persistent buffer; the probe reads it back.

R=2, B=2, seq_len=1 -- the failing case:

    req    = [0, 1]
    stride = [2, 2]
    base   = [0, 2]        <-- EXACTLY as intended
    out    = [1.0, 1.0]    expected [1.0, 3.0]

So `req`, `kv_row_stride` and `req_base` are all correct, and the kernel STILL reads
request 0's V for request 1. **The request row base was NOT the (only) cause of the
aliasing.** Fixing it demonstrably changed the larger cases (R4B4 went from
[1.5,1.5,1.5,1.5] to [1.5,1.75,2.0,2.25], i.e. request-dependent), so the base
does take effect on some path -- but the seq_len=1 path is still wrong with a
provably correct base. That means there is a SECOND aliasing mechanism in the load
path, most likely in the V read / PV staging (the K read and the mask use the same
`base`, so the suspect is whatever the PV MMA actually consumes).

Harness caveat, so this is not misread: for R>4 the printed `stride`/`base` columns
are garbage because my print offsets overlap (`req` at [0,R), `stride` at [4,4+R)).
That is my reporting bug, not a kernel result. R=2 has no overlap and is therefore
the only trustworthy readout -- and it is the case that matters.

NEXT: the base is proven correct, so stop touching it. Audit what the PV MMA
consumes:
  * PASS 2's `_load_kv_packed(..., want_v=True)` row computation (is `req_base`
    actually threaded into the V branch, not just the K branch?)
  * the `sV_packed` -> PV B-descriptor path (is the staged tile the one that was
    loaded, or a stale tile from a previous request?)


## Second audit: the staged V value is NOT the value at `base`

Published `sV_code[0,0]` (the value the PV actually multiplies) per request, with
16-wide debug regions so the offsets cannot overlap (my earlier print collided for
R>4; fixed).

  [A] R=2  B=2   base=[0, 2]            staged=[1, 1]                 expected [1, 3]
  [R4B4]         base=[0, 4, 8, 12]     staged=[1, 1, 1, 1]           expected [1, 5, 9, 13]
  [R8B8]         base=[0, 8, ..., 56]   staged=[1,1,2,2,3,3,4,4]     expected [1, 9, 2, 10, 3, 11, 4, 12]
  [R16B16]       base=[0, 16, ..., 240] staged=[5,6,7,...,15,1,2,3,4,5] expected [1,2,...,15,1]
  [control] R=1  base=[0]               staged=[1]                     expected [1]  OK

So:
  * `req`, `kv_row_stride` and `req_base` are CORRECT at every size (proven).
  * The K side clearly uses the base (the R4B4 outputs became request-dependent).
  * The V side does NOT land on the value at `base`.

Quantified: for R=16,B=16 the staged sequence is the expected sequence shifted by a
CONSTANT 4 blocks -- staged[req] == code((req+4)*B). So the V row is being taken
from somewhere other than `base + 0`.

Both branches apply `base` in source, so the divergence is in the V OPERAND's
indexing/tensor, not the row arithmetic:
  * `mVh = mV[None, None, kv_head]` -- check mV's layout/permute against mK's in
    `__call__` (q_perm/kv_perm handling); a different effective stride between the K
    and V tensors would make the same row index land on different data.
  * `_dequantize_transposed(sV_packed, sVLut, sV_code, ...)` writes `sV_code[col,row]`
    -- verify the transpose maps the kv-row axis to the axis the PV reads as its K
    (column), i.e. that `sV_code[0, 0]` really is (hdim 0, kv 0) for THIS request.

The illegal address remains unexplained by any of this and is still a separate fault.


## Three-point tap: CASE 1, then suspect #1 refuted -> stale staged V tile

P0/P1/P2 for one element (R=2, B=2, request 1, expected sentinel 3):

  [A]  P0 raw V byte = [17, 17]   (low nibble = [1, 1])
       P1 sV_code[0,0] = [1, 1]
       P2 out = [1.0, 1.0]        expected [1, 3]

P0 is already wrong: the RAW GLOBAL READ returns request 0's byte. So it is not the
dequant (case 2) and not the PV contraction (case 3) -- case 1.

Then the layout comparison for the same logical row:

  [A]      K offset = [0, 256]              V offset = [0, 256]
  [R4B4]   K = [0,512,1024,1536]           V = same
  [R8B8]   K = [0,1024,...,7168]           V = same
  [R16B16] K = [0,2048,...,30720]          V = same
  [control] K = [0]                        V = [0]

K and V map the same logical row to the SAME physical offset at every size. So the
suspect #1 chain (`mV`, `kv_perm`, `mVh`, layout, stride) is REFUTED: there is no
K/V layout divergence.

That leaves a contradiction to explain: `req_base` is published correctly (0, 2 for
R=2), the source applies `base + nt*tile_n + row` in BOTH the K and V branches, the
K/V layouts agree -- and yet P0 for request 1 is request 0's byte, i.e. the V read
behaves as if base were 0.

Leading mechanism, and it fits the data exactly: a STALE STAGED V TILE. P0 is
`const_byte(1)` for every request, i.e. the value of the very first thing staged.
PASS 1 stages with `want_v=False`; PASS 2 stages with `want_v=True`; if the PASS 2
V staging does not actually re-run per request (or the tile the PV reads is not the
tile just loaded), the PV consumes whatever the previous request left behind. That
also explains the seq_len=1 symptom precisely: correct scores (K uses base) with
request 0's values (V is stale).

NEXT TAP (cheap, same harness): prove the PASS 2 V staging executes per request --
publish a per-request counter or a per-request tag from inside the PASS 2 V load
(e.g. write `nt` and `req` into the debug buffer from the V-branch), and separately
confirm which smem tile the PV consumes (`sV_code` after dequant vs a stale copy).

Bug decomposition stands:
  Bug 1  request row base        fixed + proven correct
  Bug 2  V read ignores base     NOT a layout fault; leading candidate = stale
                                 staged V tile (staging not re-run per request)
  Bug 3  illegal address (vLLM)  independent, still open


## v0 has NO circular pipeline; and my debug slots are torn across blocks

1. `num_stages` / `num_dequant_stages` are DEAD CONFIG in the v0 kernel: declared
   (default 2) but never read, and the staging buffers are single-stage
   (`tn * k_packed_bytes`, no stage dimension). SMEM is per-block anyway, so there
   is no circular pipeline stage to be out of sync. The pipeline-stage hypothesis
   has no mechanism in this kernel.

2. Host-side check: `gathered.v_packed[r,0,0,0] & 0xF = [1,2,3,4]` for r=0..3 --
   the gathered buffer is CORRECT, row r holds block r. Together with correct
   `base`, correct layout offsets (K == V, 256 for row 2) and correct source
   expressions, every inspectable quantity says the V load should read row 2.

3. The in-load tap published `base=240 nt=1 kv_len=128 first V row index=366`,
   i.e. `base` provably REACHES the load function (240 = 15*16 for the last
   request), yet the row index in the same snapshot is inconsistent with it.

4. WHY: `mDbg[112..115]` are written by EVERY block in the grid (one slot each), so
   the four numbers are a TORN snapshot from different (x,y,z) blocks. The same
   applies to the earlier per-request slots when the grid has more than one y
   (head) block per request. So the last two taps' *interpretation* is unsound --
   the values are individually written but not a coherent per-block snapshot.

=> Instrumentation rule for this kernel, to stop repeating this: any debug write
   must be unique per (blockIdx.x, blockIdx.y, blockIdx.z), e.g.
   `mDbg[base_off + (z * Hq + y) * N ...]`, or single-threaded with a barrier.
   Four measurement-layer mistakes this session (two grep filters, one overlapping
   region, this torn snapshot) vs zero kernel-side surprises.

NEXT: re-run the V tap with block-unique slots and one barrier before reading, so
the snapshot is coherent. Only then interpret where the wrong V byte enters.


## V-tap rerun attempt: instrumentation error #5, halted

Block-unique slots were wired, but the kernel fails to stage with:

  NameError in `__call__`: cannot access free variable 'dbg_slot' where it is not
  associated with a value in enclosing scope
  suggestion: Variables used inside staged control flow must be defined before the
  control flow region.

So `dbg_slot` is being read from a staged scope that does not see it. The fix is
mechanical: hoist `dbg_slot` (and `n_q_heads`) to the very top of the kernel body,
before every `if`/`for` region, and make sure each publish site lives in the same
staged scope as its definition -- CuTeDSL needs loop/branch-referenced locals defined
before the region, and a constexpr subscript (`mDbg[dbg_slot + k]`) inside a staged
loop is exactly the case that trips it.

I stopped here rather than spend another GPU run on instrumentation. The accounting
at that point, unchanged:

  Bug 1  request row base        fixed + proven correct (0/2 for R=2; 240 for R16)
  Bug 2  V raw read wrong        located at P0 (the global read, before dequant and
                                 before the PV); mechanism NOT proven, because the
                                 two taps that would have proven it were unsound
                                 (torn slots). Repro is deterministic and clean.
  Bug 3  illegal address (vLLM)  independent, still open

Evidence that IS sound and should not be re-derived:
  * gathered.v_packed[r,0,0,0] & 0xF = [1,2,3,4] for r=0..3 -- the GATHER IS RIGHT.
  * crd2idx((base,0), layout) for K and V are IDENTICAL at every size (256 at row 2)
    -- so there is NO K/V layout divergence.
  * the V branch source applies `base + nt*tile_n + row` exactly like the K branch.
  * `base` provably reaches the load function (published 240 for R16's last req).
  * num_stages/num_dequant_stages are dead config; v0 has no circular pipeline.
  * the illegal address is NOT explained by any of the above.

Measurement-layer failures this session (5): two grep filters that hid results, one
overlapping debug region, one torn cross-block snapshot, one staged-scope
reference. Zero kernel-side surprises. Fix the measuring before the next conclusion.


## 2026-09-17 (later) — Bug 3 CLOSED: it was TWO bugs, and it was never the V path

Both illegal-address bugs are now identified and fixed; neither was the stale-V
mechanism tracked above, and Bug 2 (the aliasing) is also resolved as a side
effect of the base fix. Every earlier "V is stale/wrong" conclusion is
superseded.

### Bug 2 resolution — the V tap was measured with a BLOCK-unit row base

The tracker's contradictory taps (`base=[0,2]`, "base proven correct", yet V
aliased) are explained by the base being applied in BLOCKS at the time of those
runs. The gather emits `row = req*max_blocks_per_req + block` in BLOCKS; the
kernel reshapes the gathered buffer to `(page_rows*block_size, Hk, pb)`, so the
row base must be in TOKENS:

    kv_row_stride = max_blocks_per_req * block_size      # tokens per request

`launch_thunder_attention` now multiplies by `block_size`
(`cute_kernel.py:~745`). With that, `R=2,B=2` gives `base=[0,32]`, and the
multi-request probe is fully correct:

    [A]      out = [1.0, 3.0]                 expected [1.0, 3.0]        OK
    [R4B4-s1]    [1.0, 5.0, 9.0, 13.0]                                        OK
    [R16B16]     [1,2,...,15,1]                                               OK
    [R4B4]       [1.5, 5.5, 9.5, 13.5]                                        OK
    [control]    [1.5]                                                        OK

So there was NO second aliasing mechanism and NO stale staged V tile: the PV
path was always correct. (`STAGED V WRONG` still prints for R16; that is the
probe's own torn debug tap, output is correct.)

### Bug 3a — Q-block grid was global, q offsets are per-request

`__call__` sized the grid as `total_q / tile_m` while the kernel addresses Q as
`q_start[req] + q_block*tile_m + row` and only guarded `row < q_len`. Warm-up:
`total_q=16384` -> 256 q-blocks for EVERY request, `q_start=req*16` -> Q read
reached row `32703` of a 16384-row tensor -> CUDA_ERROR_ILLEGAL_ADDRESS.

Never seen before because:
  * single request -> `q_start == 0`, indices stay in range;
  * multi-req probe -> `total_q <= tile_m`, so only `q_block 0` existed.

Fix (`cute_kernel.py`):
  * grid.x = `ceil(max_query_len / tile_m)`, `max_query_len` threaded from
    `metadata.max_query_len` (host fallback = token count for single request);
  * Q load guard is now `q_block*tile_m + row < q_len`;
  * `_valid` takes `q_off` and tests `q_off + row < q_len`.

### Bug 3b — the fault that actually stopped the engine: PAD slots in the cache write

`CUDA_LAUNCH_BLOCKING=1` moved the error off the CuTe kernel entirely:

    do_kv_cache_update -> reshape_and_cache -> _reshape_and_cache_kernel
    Triton Error [CUDA]: an illegal memory access

Instrumented host probe (THUNDER_DEBUG_KV=1) at the warm-up:

    [TQ-CACHE] n=16384 ... slots_min=-1 slots_max=-1

vLLM's KV-cache warm-up supplies an all-PAD `slot_mapping` (PAD_SLOT_ID == -1).
The Triton writer computed `blk = -1 // 16 = -1`, `pos = 15` and stored to a
negative address. The CuTe kernel crash we were chasing was the SECOND symptom of
a poisoned context; the write path faulted first.

Fix (`cache_layout.py`):
  * Triton: `slot_ok = row_mask & (slot >= 0)` gates every K/V/norm store.
  * Reference path: index only `slot >= 0` (torch would have wrapped -1 to the
    last block).
  * Also fixed a real (non-crash) bug found on the way: the scales strides were
    bound head/pos swapped (`stride(1)` is Hk, `stride(2)` is block_size).

### Bug 4 — causal mask used the GLOBAL flat token index

Found while building the warm-up-shape repro. `_valid` masked `kv <=
q_start + q_off + row`, i.e. the global position in the packed query tensor.
vLLM appends query tokens to the request's existing context, so the correct
comparison is the position WITHIN the request:

    kv <= (kv_len - q_len) + q_off + row

For decode (`q_len == 1`) the old form masked a 512-token context down to token
`req`, so a decode returned only block 0's value. Fixed; single-request prefill
(`q_len == kv_len`, `q_start == 0`) is unchanged by construction.

### Verification

New standalone repro `ci_probe/modal_probe_warmup_shape.py` uses the exact
production tuple (`R=1024, B=32, T=16, Hq=16, Hk=8, D=128`, GQA 2, causal) and a
decode-shaped `T=1` case. B200:

    [R64B32T16]    OK   (finite, request-dependent, causal mean exact)
    [R1024B32T16]  OK   <- the previously-crashing tuple
    [R1024B32T1]   OK
    WARMUP SHAPE: ALL PASS

vLLM engine smoke (`modal_tracka_smoke.py`): warm-up now runs to completion
across profile shapes (`n=16384 max_query_len=16`, then `n=2048
max_query_len=2`), `slots_min=16 slots_max=16385` (real slots), no fault. The
run was only ever stopped by the local client disconnecting before the
generation line printed, so "engine generates text" is NOT yet observed.

CPU suite: 38 passed, 8 skipped.

### Open items

  * Engine generation still unobserved end-to-end (runs aborted at the client).
    The smoke now streams the child's output and defaults CUDA_LAUNCH_BLOCKING=0;
    use `modal run --detach` so a client disconnect does not kill the app.
  * `ci_probe/modal_probe_exec.py` (single-request, random Q/K/V) now reports
    `max_abs ~0.35` and many violations against its fp16-SDPA reference; the
    stored green result (`probe_exec_b200.txt`, max_abs 1.46e-3) predates the
    Track-A launcher rewrite. The masking edits above are PROVABLY no-ops for
    that probe (q_len is a multiple of tile_m and q_start == 0, so grid, Q guard
    and `_valid` are all unchanged), so this regression is pre-existing and
    needs its own controlled baseline before chasing.
  * The norms are still written through the lazy `_scales_for` side buffer
    (not counted by vLLM's cache accounting); the tidy fix is a view into the
    packed slot.

## 2026-09-17 (even later) — trimmed gather, kernel bench, and what "e2e" costs

### Engine now generates (observed)

With 3a/3b/4 fixed the engine gets all the way to decode:
prefill forwards (`is_prefill=True, max_query_len=16`, then `2`), then decode
(`is_prefill=False, max_query_len=1, n=1`) with the KV slot index advancing
(19 -> 20 -> ... -> 25). No fault. The generated string itself was still not
printed at the time of writing because the run is extremely slow (below).

### Trimmed gather — the per-token cost was the gather

`PagedKVManager.gather_packed_tiles` used to rebuild the ENTIRE reservation
(`max_num_reqs * max_blocks_per_req` page rows) every call, with padding and an
out-of-place advanced index + copy:

    warm-up: r=1024 b=32  -> 32768 page rows -> ~536 MB per layer per forward

It now gathers only the live region and writes it straight into the reserved
buffer (pointer-stable, request-major stride preserved so the launcher contract
is unchanged):

    b_live = ceil(max(seq_len) / block_size)      # capped by the table width
    kv_view[:r, :b_live].copy_(k_codes[bt[:r, :b_live]])

Measured in the engine: `live_page_rows=1 reserved_page_rows=32768` for decode
and `1024` for the warm-up batch (32x/512x less gather traffic). `reserve()` is
unchanged; rows past the live region stay stale and are never read (seq_lens
masks them, kernel row bound is `kv_len`). CPU suite still 38 passed, 8 skipped
(the paged-KV test now compares the live region rather than the padded rows).

### Kernel benchmark (B200, v0 mma_sync, `ci/modal_bench_smoke_run.py`)

Reference = dequant-to-fp16 + fp16 SDPA on the same quantized cache.
Synthetic shapes: nheads=32, nkv=8, hd=128 (GQA 4:1); no HF model.

    decode-short (1,1,4096)      ours= 1.315 ms  ref= 2.518 ms  speedup=1.915x  PASS
    decode-long  (1,1,32768)     ours=10.423 ms  ref= 6.210 ms  speedup=0.596x  BELOW
    prefill      (1,4096,4096)   ours=958.15 ms  ref=15.18 ms   speedup=0.016x  BELOW

(Slightly better than Phase-1: short decode 1.70 -> 1.92x.) The two BELOW rows
are structural performance, not correctness: no pipelining, no split-K, and the
prefill path re-reads/re-dequantizes K per KV tile.

### The remaining e2e cost is host launch overhead, not the kernel

Even with the gather trimmed to 1 page row, decode is ~1s per forward. The
smoke runs `enforce_eager=True`, so every one of the ~28 layers calls the
CuTeDSL entry point per token: host-side `from_dlpack` conversions, argument
marshalling and type checking, plus one kernel launch. That host path, not the
GPU work, dominates an eager e2e number. An end-to-end THROUGHPUT number only
means something once the path is captured in a CUDA graph (the builder already
declares `UNIFORM_SINGLE_TOKEN_DECODE`), otherwise it measures the launcher.

### Model / shapes used

  * engine smoke: `Qwen/Qwen3-0.6B` (THUNDER_SCHEDULE=mma), max_model_len=512,
    block_size=16, fp16, gpu_memory_utilization=0.6, enforce_eager=True.
  * kernel smoke: synthetic, nheads=32/nkv=8/hd=128, decode-short 4096,
    decode-long 32768, prefill 4096.
  * upstream comparison: vLLM ships `AttentionBackendEnum.TURBOQUANT` at this
    pin, so a vs-upstream run is available once the engine loop is acceptable.
