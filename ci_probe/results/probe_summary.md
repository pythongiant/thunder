# Kernel probe results

Cheap-tier probing: CuTeDSL cross-compiles `sm_100a` PTX (and runs the tracer)
on an sm_89 **L4** when `CUTE_DSL_ARCH=sm_100a`, so "does it compile" costs ~2
minutes of L4 time instead of a B200. Execution needs a real SM100 part (B200).

## 1. Original `attention.py` (prototype)

`ci_probe/modal_probe.py`

| step | result |
|---|---|
| import | OK |
| construct (L4, no arch override) | ERR AssertionError: targets SM100/SM110 |
| construct (L4, `CUTE_DSL_ARCH=sm_100a`) | OK |
| compile | ERR ValueError: invalid mode element for input of rank 3, got mode=[1, 3, 2, 0] at attention.py:429 |
| execute | same error |

## 2. First packaged port (schedule stubbed)

`ci_probe/modal_probe_kernel.py`

| step | result |
|---|---|
| construct | OK |
| compile | OK |

Fixes that moved it: rank-aware permutations (Q/O rank 3); SMEM layouts built
inside each CuTeDSL region (region isolation forbids sharing a layout object
between host and kernel bodies); `SmemAllocator().allocate(SharedStorage)` rather
than `SharedStorage()`; `.launch()` must not pass `smem=`; LUT staging iterates
by the real thread count.

## 3. Cooperative two-pass mainloop (current)

`ci_probe/modal_probe_kernel.py` compiles the full mainloop;
`ci_probe/modal_probe_exec.py` executes it on **B200** (the kernel is
self-contained, so this needs only torch + nvidia-cutlass-dsl -- no flash-attn,
no vLLM).

Compile: construct OK, compile OK.

Execute, gate `|a-b| <= atol + rtol*|b|` with atol=rtol=1e-2. `violations`
counts elements outside the gate:

| k | v | head_dim | seqlen_k | causal | max_abs | violations |
|---|---|---|---|---|---|---|
| 4 | 4 | 128 | 128 | True | 1.464e-03 | 0 |
| 4 | 4 | 128 | 256 | True | 1.464e-03 | 0 |
| 4 | 4 | 128 | 512 | True | 1.464e-03 | 0 |
| 4 | 4 | 128 | 128 | False | 4.302e-04 | 0 |
| 4 | 4 | 128 | 192 (not a tile multiple) | True | 1.464e-03 | 0 |
| 2 | 2 | 128 | 128 | True | 1.021e-03 | 0 |
| 3 | 4 | 128 | 128 | True | 1.464e-03 | 0 |
| 4 | 4 | 64 | 128 | True | 1.380e-03 | 0 |

The residual is quantization distortion, not kernel arithmetic: the reference is
an fp32 SDPA over the *dequantized* K/V, and the kernel matches it to ~1e-3.

## Bugs this loop found and fixed

1. **Pass 1 used `rK` before it was ever loaded** from SMEM, producing NaN.
   Fragments must be re-read every tile because the code buffer is reused.
2. **Probe, not kernel: wrong basis.** The kernel scores in the rotated basis
   and accumulates `P @ (R V)`, so the reference had to be built in the rotated
   basis. This also pinned down a contract requirement: the caller must apply
   `R^T` to the output, which `TurboQuantAttentionImpl.forward` does as its
   output-projection GEMM.
3. `ldmatrix` requires 128-bit-aligned SMEM sources and a swizzle-compatible
   layout. v0 uses plain 16-bit universal copies; 32-bit copies cannot vectorize
   over the non-unit fragment strides.

## P3b path A: tcgen05 / TMEM (sync fixed; operand K-order mismatch remains)

`turboquant_vllm/attention/cute_kernel_tcgen05.py` is the attempted tcgen05
conversion. It does **not** compile, and the blocker is in CUTLASS-DSL 4.7.1,
not in the schedule:

```
'cute.gemm' op invalid layout of A/B/D.
  A: (128,128):(128,1)  B: (64,128):(128,1)  D: ((128,64),1,1):((65536,1),0,0)
```

`cute.gemm` is the only generic GEMM entry in this build, and it rejects both
operand forms a tcgen05 SMEMxSMEM MMA needs:

| operand layout | result |
|---|---|
| composed / swizzled (`make_composed_layout(make_swizzle(...))`) | `doesn't support composed layout for A/B/D` |
| plain row-major | `invalid layout of A/B/D` |

and `make_fragment_C` yields a register-fragment layout for the accumulator
(row stride 65536), not a TMEM layout, which the op also rejects.
`sm100_utils.make_trivial_tiled_mma` does not change this: it still emits
`atom_layout_MNK=(1,1,1)` / `permutation_MNK=[_;_;_]`.

FA4 avoids this by never using `cute.gemm` for tcgen05 — it calls its own
PTX-level helpers (`gemm_ptx_precomputed_varname`, `gemm_ptx_partial`) with
precomputed SMEM descriptors, an instruction descriptor and an explicit TMEM
column offset. `cutlass.utils.blackwell_helpers` 4.7.1 has no `gemm_ptx`
helpers, so completing this needs the descriptor-level path
(`tcgen05.make_umma_smem_desc`, `tcgen05.get_s2t_smem_desc_tensor`,
`tcgen05.mma` with explicit TMEM offsets) or vendoring FA's helpers.

API facts established against 4.7.1 while attempting this (all used correctly in
the blocked module, and verified to instantiate/trace):

* `cutlass.utils.TmemAllocator(alloc_result_dst_smem_ptr, barrier_for_retrieve=`
  `NamedBarrier(id, num_threads), allocator_warp_id=0)` with `.allocate(cols)`,
  `.wait_for_alloc()`, `.retrieve_ptr(Float32)`, `.relinquish_alloc_permit()`,
  `.free(ptr, cols)`.
* `cute.arch.mbarrier_init(ptr, count)`, `cute.arch.mbarrier_wait(ptr, phase)`,
  `tcgen05.commit(mbar_ptr=..., cta_group=CtaGroup.ONE)`.
* `tcgen05.make_tmem_copy(atom, tmem_tensor).get_slice(tidx)` with
  `tcgen05.copy.Ld32x32bOp(Repetition(32))` for TMEM->register and
  `St32x32bOp` for register->TMEM.
* There is **no** `zero_init` or `accumulate` kwarg on `cute.gemm` in this
  build; the accumulator must be zeroed explicitly through a TMEM store.
* `cute.make_tmem_ptr` does not exist; TMEM tensors come from
  `thr_mma.make_fragment_C(cute.append(partition_shape_C((M,N)), stage))`.

Path A next steps (measured, not guessed)
-----------------------------------------
FA's tcgen05 helpers were vendored into `attention/_vendor/` (they have **no
quack dependency**; only `cutlass` + `mma_sm100_desc`). The non-MMA plumbing is
API-correct and reaches the compiler: `TmemAllocator` alloc/wait/retrieve/free,
`tcgen05.commit` + `mbarrier_wait`, explicit TMEM accumulator zeroing (no
`zero_init`/`accumulate` kwarg exists), and `Ld32x32b` TMEM readback.

The remaining gap is the SMEM operand plumbing, and the conventions are now
measured rather than guessed (`ci_probe/modal_layoutprobe.py`):

```
make_smem_layout_a/b(tiled_mma, (M,N,K), dtype, stages)
  -> _ComposedLayout, rank-4 outer, shape ((128,16),1,(4,2),1), inner S<3,4,3>
partition_shape_C((M,N))        -> ((128,64),1,1)
make_fragment_C(append(psc,1))  -> rank-4 shape ((128,64),1,1,1)
make_fragment_A(smem)           -> ERROR: "use recast_ptr(ptr, S<3,4,3>,
                                   element_type) to move swizzle to the ptr"
```

`gemm_ptx` / `gemm_ptx_partial` index the operand as `sX[None, None, 0]`, so it
must be a rank-4 swizzle-in-pointer tensor. FA never calls them directly in its
SM100 forward: the live call sites use `declare_ptx_smem_desc` to precompute
`smem_desc_start` and then `gemm_ptx_precomputed_varname`; the direct calls are
commented out. So the two things left are (1) precompute the SMEM descriptors and
per-K-tile offsets, and (2) write the dequantized tile into the swizzled layout
through a matching tiled copy (element indexing cannot address a rank-4 swizzled
layout; a plain staging buffer + SMEM-to-SMEM copy is the cheapest route).

Execution on B200 (`ci_probe/modal_probe_exec.py --kernel-module
...cute_kernel_tcgen05 --tile-m 128 --tile-n 64`) -> `probe_exec_tcgen05_b200.txt`:

```
k=4 v=4 d=128 nk=128 causal=True max_abs=inf violations=32317
k=2 v=2 d=128 nk=128 causal=True max_abs=2.2124e+03 violations=32327
```

So the tcgen05 schedule **compiles, launches and executes on B200** with the
right grid/TMEM/MMA plumbing, but the output is wrong.

**Write path validated -- the bug is elsewhere.** `ci_probe/modal_probe_deposit.py`
deposits `B[n,k] = n*1000+k` into a tcgen05 B operand via
`cute.composition(operand, row_major)` and reads it back through the operand's
*own* MMA partitioning (`make_tiled_copy_B(...).partition_S(operand)` ->
`partition_D(plain)`). Result: **`exact match: True`**.

So the composed row-major write agrees with the operand's coordinate system; the
earlier "composition mangles the swizzle" hypothesis was wrong, and the five
deposit rewrites chased the wrong layer. The kernel keeps the validated write
path (`smem.allocate_tensor(layout=outer, swizzle=inner)` + composition writes).

Remaining suspects are all in the TMEM readback / MMA path:

1. missing `cute.arch.fence_view_async_tmem_load()` before the `Ld32x32b` read
   after `tcgen05.commit` + `mbarrier_wait`;
2. the S epi-split readback (`zipped_divide` + `partition_D`) column mapping;
3. `tmem_ptr + tmem_o_offset` column arithmetic for O.

Next: extend the deposit harness with a real QK MMA (deposited identity x
deposited pattern) and check `S[m,n] == B[n,m]`, which bisects "MMA writes S" vs
"readback of S" and picks one of the three.

One compiler caveat found on the way: replacing the epilogue sub-tiled readback
(`zipped_divide` + `partition_D`) with a whole-tile
`cute.make_fragment_like(tDtS, Float32)` **segfaults the CuTeDSL compiler**
(SIGSEGV, returncode -11) on this build, so the sub-tiled readback is retained.

The backend refuses `TURBOQUANT_SCHEDULE=tcgen05` with a message pointing at
this module rather than silently falling back.

## P3b follow-up

* tcgen05/TMEM via the descriptor-level path (see above).
* Warp specialization (load / dequant / MMA / softmax / correction / epilogue)
  with `PipelineTmaAsync` + `PipelineUmmaAsync`.
* Single-pass online softmax plus a correction warp (v0 is known-max two-pass).
* split-K, GQA K reuse, V multi-head GEMM (each needs a `pack_gqa`-style M
  restructure; `VARIANT_TAG` scaffolding is in place).
* `ldmatrix` with swizzled SMEM tiles.


## 2026-09-16 — tcgen05 K-traversal SOLVED in isolation; kernel still failing

### Proven (ci_probe/modal_probe_kblock.py, M=128 N=64)
Descriptor path replaces `cute.gemm` entirely:
  * `declare_ptx_smem_desc(start, base, tCrX[None,None,None,0].layout, prefix)`
  * `declare_ptx_idesc(op, var_name)`
  * `gemm_ptx_precomputed_varname(acc.toint(), b_start, smem_desc_base_b=..., tCrB_layout=..., ...)`
  * base from `make_smem_desc_base(recast_layout(128,16,layout.outer[0]), layout.inner, Major.K)`
    computed on the HOST (kernel-side `smem_desc_base_from_tensor` is wrong: `allocate_tensor` rewrites the tensor layout).
  * K block offsets come from `crd2idx((0,0,k), tCrX_layout)` = 16-byte descriptor units.
RESULTS: K=32  A-blocks {0,1} -> 3.0;  K=128 A-blocks {0,1} and {0..7} offsets
  A: 0,2,4,6,1024,1026,1028,1030  B: 0,2,4,6,512,514,516,518  -> 3.0 for both variants. CORRECT.
RESULTS (PV side, A=(128,64) B=(128,64), flat mode2=4): blocks 0/1/2/3 all -> 1+val. CORRECT.

### Deposit
`cute.composition` with the shape's coordinate strides is WRONG — measured
`ci_probe/modal_probe_semview.py`: composition gives 176/192 where the explicit
nested coordinate gives 1/3. Use the explicit coordinate:
  hdim>64 (two swizzle atoms): ((r, c%16), 0, ((c//16)%e1, c//(16*e1)), 0)  e1 = hdim//32
  hdim<=64 (one atom):         ((r, c%16), 0, c//16, 0)
PASSING A SCALAR to a nested mode makes CuTe do the row-major decomposition itself,
so the probe's flat `k//16` is equivalent to the nested tuple form.

### Kernel state after applying both fixes
`cute_kernel_tcgen05.py` now uses the descriptor path + `_nested_store` deposits.
B200 sweep: max_abs 28.6 -> 10.6 (d=64: 21.1 -> 11.0), all 8 cases still fail.
So the K traversal is fixed in isolation but NOT yet in the kernel — remaining
difference between probe and kernel is un-isolated (candidates: softmax/P path,
TMEM acc live-range between the QK and PV accs, descriptor-vs-`cute.gemm`
accumulation/synchronisation ordering).

## 2026-09-16 — single-KV diagnostic: QK FAILS on dense data (decision rule -> addressing)

`ci_probe/modal_probe_singlekv.py` (DBG=1/2/3) drives the PRODUCTION kernel via
constexpr hooks (`dbg_mode`), single KV tile, identity LUTs so dequant == code.
Host references are exact.

  dbg_mode 1  S = raw QK accumulator      ref_absmax 175.7   max_abs 497.7
              got[0,:6] = [216.6, 196.0, 296.3, 220.5, 187.7, 277.6]
              ref[0,:6] = [ 76.8,  47.0,  66.3,  53.3,  57.0,  84.6]
  dbg_mode 2  O = (P_fixed * v_norm) @ V  ref_absmax 366.3   max_abs 844.8
              got ~3.1x ref, but rows sharing the same P row agree exactly
              (got[1] == got[5], ref[1] == ref[5]) -> P->V row mapping is consistent
  dbg_mode 3  O = (softmax@V)             ref_absmax  93.8   max_abs 106.3
              same ~2-4x inflation, inherits mode 2's V-side error

CONCLUSION: mode 1 already fails => the QK itself is wrong with a DENSE operand,
so this is addressing, not TMEM/softmax. The earlier "3.0" probes were too weak:
  * `modal_probe_kblock.py` used one-hot A (only row 0, only 1-2 K blocks) and
    B = ones, so B's per-K-block origins were NEVER validated;
  * the PV probe likewise used B = ones;
  * the one clean counter-signal was dismissed: PV block 3 gave 13 instead of 4
    (`A[m,0]=1, A[m,48]=3, B=1`) == 1 + 3*4, i.e. the k=48 descriptor origin
    read the same element from 4 K atoms. That was written off as cross-variant
    contamination; it was real.
NEXT: re-run the descriptor-origin probe with DISTINCT per-K-block values on BOTH
operands (e.g. A[m, 16j] = j+2, B ones -> expect sum j+2; and a B-side variant),
for all 8 blocks at hdim=128 and all 4 at tn=64. Suspect: block>=2 origins inside
a SWIZZLE_128B atom (32 B / 96 B in-unit offsets) are not valid K-block boundaries.

Incidental real bug found and fixed: CuTeDSL traps "tensor memory not completely
freed" if the kernel returns with a live TMEM allocation -- both diagnostic exits
must `relinquish_alloc_permit()` + `free()`. Anything that adds an early return to
the tcgen05 kernel must do the same.

## 2026-09-16 — all-blocks probe: deposit<->descriptor K-block correspondence BROKEN past block 1

`ci_probe/modal_probe_blocks.py`: M=128 N=64 K=128, exactly ONE K block active on
ONE operand, value = blk+2, B/A = ones elsewhere. Correct => out == blk+2
everywhere (all 128 rows x 64 cols). side/blk/val are runtime kernel args, so all
16 variants share one compile.

  RESULT: 3/16 variants correct.

  side=A (activate A[:, 16*blk])      side=B (activate B[:, 16*blk])
    blk=0 val=2 -> 2.0    OK            blk=0 val=2 ->  2.0    OK
    blk=1 val=3 -> 3.0    OK            blk=1 val=3 ->  6.0 = 2x  BAD
    blk=2 val=4 -> 16.0 = 4x BAD        blk=2 val=4 -> 12.0 = 3x  BAD
    blk=3 val=5 -> 20.0 = 4x BAD        blk=3 val=5 -> 20.0 = 4x  BAD
    blk=4 val=6 -> 24.0 = 4x BAD        ...
    blk=5 val=7 -> 28.0 = 4x BAD
    blk=6 val=8 -> 32.0 = 4x BAD

  Multiplicity is uniform across ALL rows (0,1,4,7,8,15,63) and all columns, so
  this is not a row/swizzle-phase effect.

  Interpreted: the value deposited for block j is read (j+1) times (A side saturates
  at 4 = the mode2[0] extent). Reading multiple origins is impossible -- 1+2+3+4 > 8
  sub-MMAs -- so the *deposit* is smearing the value across blocks 0..j, while the
  descriptor origins themselves are internally consistent with the layout:
    A deltas (16B units) = [0, 2, 4, 6, 1024, 1026, 1028, 1030]
    B deltas (16B units) = [0, 2, 4, 6,  512,  514,  516,  518]
  (physical bytes 0/32/64/96 then 16384+0/32/64/96 for A, 8192+... for B), matching
  layout ((128,16),1,(4,2),1) strides ((64,1),0,(16,8192),0) exactly.

  Why the earlier probes missed it: they used constant data along K (B == ones) or a
  single active column, both of which are invariant to K-block smearing. Only a
  per-block-distinct value exposes it.

ROOT CAUSE (to fix): the deposit coordinate `((r, k%16), 0, k//16, 0)` and the
descriptor origins (from `crd2idx` on the *fragment* layout) are two independent
guesses at the same K-block map, and they only agree for blocks 0-1. Not swizzle
arithmetic, not TMEM, not softmax.

FIX DIRECTION: derive both from ONE source of truth. Either
  (a) compute the deposit flat index with `crd2idx` on the operand's own outer
      layout (invert the layout we hand to the descriptor), or
  (b) pin the SMEM layout explicitly so the K decomposition is unambiguous
      (flat K mode, no nested (4,2)), then re-check descriptor deltas.
Regression gate: this probe must go 16/16 before touching the production kernel.

## 2026-09-16 — per-k multiplicity sweep: previous "deposit smears" conclusion was WRONG

Option (a) is formally impossible, proven by the compiler:
  fragA[None,None,None,0].layout = (1,1,(4,2)) : (0,0,(2,1024))
  fragB[None,None,None,0].layout = (1,1,(4,2)) : (0,0,(2,512))
M and within-block-K are collapsed (shape 1, stride 0). `crd2idx` with any
within-block coordinate hard-fails:
  MLIRError: unable to compute crd2idx with '!cute.layout<"(1,1,(4,2)):(0,0,(2,1024))">'
             and '!cute.coord<"(?,?,?,0)">'
So the object that produces the descriptor origins cannot express a deposit. The
K-block deltas it does encode, [0,2,4,6,1024,1026,1028,1030] u128 (A) and
[0,2,4,6,512,514,516,518] (B), are byte-identical to my nested-store deposit map
(layout mode2 (4,2) stride (16,8192) f16 = (2,1024) u128). Deposit and descriptor
already share one map -- they were never two competing guesses.

`major` is not free either: Major.MN is rejected outright by
  make_smem_desc_base -> ValueError: Not a canonical UMMA_MN Layout
Major.K gives base=0x4000404000010000 ltype=2 LBO=1 SBO=64.

FULL per-k multiplicity (single k activated, value 1.0, expect 1.0 everywhere).
IDENTICAL for the A and B operands:
  k   0.. 15: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1          correct
  k  16.. 31: 2 2 2 4 2 2 2 2 2 2 2 4 2 2 2 2
  k  32.. 47: 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  k  48.. 63: 4 ... (all 4)
  k  64..127: 4 ... (all 4)
  correct k positions: 16/128

CORRECTED READING: multiplicity is a function of k ALONE and is identical on both
operands. A smeared deposit would have to differ between the (128,128) A tensor and
the (64,128) B tensor; it does not. So the value is written once and READ multiple
times -- the 8 per-K-block sub-MMA invocations are reading OVERLAPPING K ranges,
doubling with each K-block bit (1,2,4 then saturated at 4). The earlier claim that
"1+2+3+4 > 8 sub-MMAs so the reads must be fine" was the error: it assumed the 8
sub-MMAs partition K, which is exactly what is broken.

=> The defect is in the per-K-block descriptor ORIGIN semantics for a SWIZZLE_128B
operand, i.e. in how the K-block advance must be encoded (start-address bump vs
LBO/SBO stepping), not in the deposit and not in TMEM.

NEXT: stop hand-rolling 8 sub-MMAs with address bumps. Issue ONE tcgen05 MMA per
tile and let the hardware walk K via instr_k using the descriptor's LBO/SBO -- the
mechanism CUTLASS actually designs for. Re-run this exact per-k sweep as the gate:
it must read 1.0 for all 128 k.

## 2026-09-16 — single-MMA/instr_k gate: K-per-instruction hypothesis REFUTED; anomaly moves to accumulator readback

Parameterized the instruction K-step (KSTEP, in 16-f16 blocks) by handing the helper
synthetic instruction-offset layouts:
  lay_a_off = ((1,1,(4//KSTEP, K//64)), (0,0,(2*KSTEP, 1024)))   # B: 512 instead of 1024
KSTEP=1 -> 8 instructions (one per 16-block, the previous form)
KSTEP=4 -> 2 instructions (blocks 0 and 4)

KSTEP=1 (8 instr): unchanged, 16/128 correct
  k   0.. 15 -> 1        k  16.. 31 -> 2 (4 at k=19,27)
  k  32..127 -> 4

KSTEP=4 (2 instr), side=A:
  k   0.. 15 -> 1
  k  16.. 63 -> 0
  k  64.. 79 -> 3   (4 at k=67,75)
  k  80..127 -> 0

Reading of this:
  * Each tcgen05.mma consumes EXACTLY K=16: with 2 instructions only the two
    activated regions respond, and the untouched ones read exactly 0. The
    "instruction covers 64 K, I advance by 16" hypothesis is REFUTED. Descriptor
    origins are fine.
  * The residual anomaly is *within* a single instruction's 16-wide K span: the
    value placed at k=64 is seen 3 times by the block-4 instruction (and the
    k=64..79 region is seen 4 times in the 8-instruction form).

That within-instruction multiplicity cannot come from the deposit: the deposit is
`t[((m,k%16),0,((k//16)%4,k//64),0)] = v`, a single assignment through a layout
index. A layout index is injective by construction (it is exactly what crd2idx
computes), so one write reaches one slot. A single 16-wide K step therefore cannot
legitimately observe one slot three times. The redundancy is being introduced
downstream of the MMA -- in the TMEM accumulator readback
(Ld32x32bOp + Repetition(32) via tcgen05.make_tmem_copy, with a destination
partition built from tiled_mma.get_slice(0) mixed with the copy's per-thread
slice). That path is now the leading suspect and it also explains why the
per-k multiplicity is identical for A and B: it is a function of the readout
geometry and k, not of either operand.

NEXT: read the accumulator through an independent, trivially-correct path (plain
1:1 thread->(row,col) mapping over the whole (M,N) tile) and re-run the per-k
sweep. If the multiplicity collapses to all-1.0, the whole K-traversal was correct
and the bug is purely in the accumulator readback partition; fix that in the
production kernel and re-gate.

## 2026-09-16 — deposit PROVEN correct, readback EXONERATED; fault is inside the tcgen05.mma

Two independent experiments, both decisive:

### 1. Physical SMEM dump of the operand (no MMA, no partition involved)
`sA_flat = make_tensor(sA.iterator, make_layout(M*K))` indexes the allocation's raw
address space. Physical structure is address = 64*m + c0 + 16*c1 + 8192*c2 (f16),
i.e. two 8192-element atoms, each readable as [m, w] with m = a//64, w = a%64.

Single k activated, value 1.0, count 1.0-slots per LOGICAL row:
  k=  0: atom0=128 atom1=0  total=128 (expect 128)  rows with !=1 logical slot: 0
  k= 16: atom0=128 atom1=0  total=128               rows with !=1 logical slot: 0
  k= 32: atom0=128 atom1=0  total=128               rows with !=1 logical slot: 0
  k= 48: atom0=128 atom1=0  total=128               rows with !=1 logical slot: 0
  k= 64: atom0=0   atom1=128 total=128              rows with !=1 logical slot: 0
  k= 96: atom0=0   atom1=128 total=128              rows with !=1 logical slot: 0
  k=127: atom0=0   atom1=128 total=128              rows with !=1 logical slot: 0
DEPOSIT IS EXACTLY CORRECT: one slot per logical row, correct atom, injective.
(This also retro-corrects the mis-designed earlier check, which reshaped the flat
physical dump as (M,K) -- meaningless, since physical rows are 64 f16 wide.)

### 2. Independent accumulator readback
Replaced the destination-partition algebra with: Repetition(N) load (each thread
reads its row's whole N extent), flatten, and place with an explicit (row, col)
index. Identical result:
  k   0.. 15 -> 1        k  16.. 31 -> 2 (4 at k=19,27)
  k  32..127 -> 4        correct k positions: 16/128
READBACK IS EXONERATED.

### Conclusion
With the deposit proven correct and the readback exonerated, the same SMEM element
is being read multiple times *inside* a single 16-wide tcgen05.mma K step. The
multiplicity depends only on k and is identical for A and B, and doubles at exactly
the 32-byte and 64-byte within-row K boundaries. That is a property of the K walk
the hardware performs from the descriptor, i.e. of the descriptor's swizzle/LBO/SBO
encoding -- not of the deposit, the readback, TMEM lifetime, or softmax.

LIKELY DEFECT: the descriptor base is built from mode0 alone --
  make_smem_desc_base(recast_layout(128, 16, la_o[0]), la_i, Major.K)
which discards mode2, the very mode that encodes the K-atom stride (8192 f16). An
earlier note in this file recorded that kernel-side
`smem_desc_base_from_tensor(tensor)` DISAGREED with this host computation, and I
resolved the disagreement in favour of the host value because the then-sparse probe
"worked". That resolution now looks wrong: sparse data was invariant to exactly this
defect, the same blind spot that hid everything else.

NEXT: obtain the descriptor from CUTLASS's canonical constructor over the FULL
operand layout (sm100_utils/blackwell_helpers make_umma_desc /
make_smem_desc_base_from_tensor on the real tensor) and re-run this per-k sweep.
Gate: 128/128. This is a one-constant change plus one run.

## 2026-09-16 — canonical descriptor == host descriptor; last hypothesis refuted

Added `desc_src`: 0 = host `make_smem_desc_base(recast_layout(128,16,la_o[0]), la_i,
Major.K)`, 1 = CUTLASS's canonical `smem_desc_base_from_tensor(sA, Major.K)` on the
REAL post-allocation tensor.

  desc_src=1 kernel fields: A LBO=1 SBO=64 ltype=2 | B LBO=1 SBO=64 ltype=2
  => IDENTICAL to the host values.

Inspection of the vendored source explains why: `smem_desc_base_from_tensor` itself
does `make_smem_desc_base(recast_layout(128, width, sA.layout[0]), swizzle, major)` --
it only ever looks at mode0. That is by design: the descriptor never encodes mode2;
the K blocks are addressed by the per-block start-address bumps. So the K-atom stride
(8192 f16) being absent from the descriptor is correct, and the descriptor base is
NOT the defect.

Per-k multiplicity with desc_src=1 is the SAME map (1 / 2 / 4, 16/128). One detail
changed: k=19 and k=27 read 3.0 instead of 4.0, while every other k is bit-identical
between runs. CORRECTION: desc_src=0 and desc_src=1 give BIT-IDENTICAL results including the 3.0
at k=19/27, so this is deterministic, not a race. The earlier 4.0 at those k came
from the pre-plumbing kernel, i.e. it is sensitive to the compiled shape of the
kernel (a real clue, not noise).

STATUS: deposit correct (physical dump), readback exonerated (independent path),
descriptor base canonical and identical both ways. The same SMEM element is still
read 2x/4x inside one 16-wide K step, doubling at the 32-byte and 64-byte within-row
K boundaries. Not yet explained.

REMAINING CANDIDATES
  (a) the vendored helper's usage/assumptions -- validate it by running it on a
      known-good case (e.g. FA's own published example geometry) in this harness;
      if that also fails, the harness usage is wrong, not the kernel.
  (b) a race: the 4.0 -> 3.0 change at k=19/27 wants a fence/barrier audit between
      the deposit, `tcgen05.commit`, and the operand read.

## 2026-09-16 — reframing as a physical 16B-cell read map + API findings for the next two experiments

Key arithmetic observation. The four per-atom descriptor origins are at u128
offsets 0,2,4,6 = bytes 0,32,64,96 = f16 offsets 0,16,32,48. Those coincide exactly
with the c1 cell boundaries (c1 = k//16). So the measured staircase is not indexed by
logical k at all -- it IS the MMA's per-physical-16B-cell read count, and the deposit's
k -> cell map is a fixed bijection that I have already verified by physical dump.

    cell 0 (f16  0..15): read 1x
    cell 1 (f16 16..31): read 2x   (3x at f16 offsets 19 and 27)
    cell 2 (f16 32..47): read 4x
    cell 3 (f16 48..63): read 4x

Sanity check that fails: with 4 per-atom instructions and K=16 per instruction, a
correct walk reads exactly K=64 f16 per row = 4 cells, i.e. every cell exactly once
(1,1,1,1). The observed counts sum to 11, not 4. More K-reads happen than K exists.
So the MMA is issuing more K-reads than the idesc should allow for these operands.

API findings for the two experiments requested (both are real work, not flag flips):

* SWIZZLE_NONE IS expressible -- `cute.make_swizzle(0, 4, 3)` maps to
  LayoutType.SWIZZLE_NONE via `_layout_type`. BUT it is not a one-line A/B:
  the SWIZZLE_NONE branch of `make_smem_desc_base` for Major.K requires
  `stride_00 == swizzle_atom_mn_size == 1`, i.e. the innermost 8 MN elements must be
  contiguous. The swizzled arm has K contiguous instead. So the no-swizzle arm needs
  its own canonical (MN-contiguous-at-uint128) operand layout, its own deposit map, and
  its own descriptor -- it is a separate operand encoding, not a switch.

* Base-alignment sweep (base + 0/16/32/64/128) needs an SMEM operand whose *iterator*
  is offset while its swizzle type is preserved. `allocate_tensor(byte_alignment=1024)`
  forces 1024B alignment, so the shift must come from a raw Uint8 allocation plus an
  explicitly constructed swizzled pointer. That API needs verification before the sweep
  can be trusted (a wrong pointer/swizzle pairing would fabricate a result).

Cheapest high-information variant that avoids both API risks: keep the layout and
descriptor fixed and PERMUTE only the deposit's cell assignment, i.e. place logical k
in cell pc1 = (k//16 + s) % 4 for s = 0..3. If the per-cell read counts (1,2,4,4) follow
pc1, the map is physical-cell-scoped; if they follow k, it is logical. That is a pure
deposit change, already inside proven-correct code.

## 2026-09-16 — static analysis: the instruction descriptor carries NO K field

Decoded `make_instr_desc` (vendor/mma_sm100_desc.py) bit layout:
  sparse_id2[0:2] sparse[2] saturate[3] c_format[4:6] a_format[7:10] b_format[10:13]
  a_negate[13] b_negate[14] a_major[15] b_major[16] n_dim[17:23]=N>>3 m_dim[24:29]=M>>4
  max_shift[30:32]
=> There is NO k_dim field. The K consumed per tcgen05.mma is implied by
`kind::f16` and by the operand encoding, NOT by anything the instruction
descriptor states. So "does the idesc agree on what one K=16 atom means" has a
precise answer: the idesc does not express it at all. All of the K semantics live
in the operand descriptor plus the physical operand encoding + the format code
(a_format/b_format, F16F32Format.F16 for fp16).

Coverage arithmetic. 4 per-atom issues at origins u128 0,2,4,6 = f16 0,16,32,48.
Correct coverage = 64 K f16 read once each = (1,1,1,1) per cell. Observed
(1,2,4,4) = 16*1 + 16*2 + 32*4 = 176 read-events for 64 positions, i.e. 2.75x
over-consumption.

No uniform per-issue K span reproduces the measurement:
  span 16B -> (1,1,1,1)
  span 32B -> (1,2,2,2)
  span 64B -> (1,2,3,4)
  observed -> (1,2,4,4)
The observed pattern requires *growing* coverage per issue: issue0 covers {0},
issue1 {0,1}, issue2 {0,1,2,3}, issue3 {0,1,2,3} -- i.e. issue i's effective read
region roughly doubles. That is not a wrong stride; it is a descriptor
start-address decode that partially ignores the advance, which is exactly the
"per-K descriptor start-address semantics" boundary.

Descriptor arithmetic is internally coherent, for the record: LBO=1 (u128) = 16 B
= 8 f16, and K=16 fp16 per instruction = 32 B = 2 LBO steps, consistent; the four
origins are start+0,+2,+4,+6 u128 with no carry into the LBO bits at bits[16:30).

Still not run (deliberately, pending API verification): the SWIZZLE_NONE arm and
the physical base-alignment sweep -- see the preceding entry for why neither is a
flag flip.

## 2026-09-16 — symbolic descriptor->cell decoder: model predicts a PERFECT diagonal

`ci_probe/desc_model.py` (local, no GPU). Decodes the exact descriptors the probe
feeds to tcgen05.mma and enumerates the physical 16B cells each issue is entitled
to consume under the documented model (start_address in 16B units; each K=16 fp16
issue consumes 2 u128 per row; canonical Swizzle<3,4,3> cell = c XOR (row%8);
SBO=64 u128 row-group stride).

  descriptor x physical-cell multiplicity (cells 0..7):
    start+0   [1,1,0,0,0,0,0,0]
    start+2   [0,0,1,1,0,0,0,0]
    start+4   [0,0,0,0,1,1,0,0]
    start+6   [0,0,0,0,0,0,1,1]
    column sums [1,1,1,1,1,1,1,1]        <-- PERFECT DIAGONAL
  per-f16-cell-group totals for f16 0..63: [1,1,1,1] (sum 4)
  total predicted cell-reads = 64 for 64 cells  => every cell exactly once

So the descriptor semantics PREDICT (1,1,1,1), i.e. total 64 read-events, and
nothing in the descriptor-level model can produce 176/64 = 2.75x.

This is the "Case 2" outcome: the descriptor is semantically valid under the
documented model, and the observed hardware/compiler path does not execute that
model. Per the agreed plan that is the point to stop changing CuTe layouts and
reduce to a minimal raw-PTX tcgen05.mma reproducer.

Cumulative status of every layout/descriptor-side line of inquiry:
  deposit placement        correct  (physical SMEM dump, one slot per logical row)
  accumulator readback     correct  (independent read path, identical result)
  descriptor origins       correct  (instructions consume exactly K=16; untouched
                                     regions read exactly 0)
  descriptor base encoding canonical (CUTLASS constructor == ours, bit-identical)
  instruction descriptor   silent on K (no k_dim field exists)
  descriptor cell model    predicts a perfect diagonal (this entry)
  => unexplained: hardware consumes 2.75x the K-cells the descriptor model allows.

## 2026-09-16 — RAW PTX REPRODUCER IS CORRECT. Defect is in the CuTe/CUTLASS path.

`ci_probe/modal_probe_rawptx.py` + `ci_probe/rawptx_kernel.py`. Hand-written
everything that matters: SWIZZLE_128B placement computed by hand
(f16_index = row*64 + ((cell ^ (row%8))*8) + (k%8)), descriptor words hand-built
(lo=0x00010000 LBO=1, hi=0x40004040 SBO=64|version=1|SWIZZLE_128B=2), instruction
descriptor hand-built (0x08118010), raw inline
`tcgen05.mma.cta_group::1.kind::f16`. Geometry M=128 N=64 K=64 = one atom, four
K=16 issues at u128 +0,+2,+4,+6. B all ones so acc[r,n] == sum_k A[r,k], i.e. a
single accumulator number exposes the read multiplicity.

RESULT (issue mask = which of the four MMAs are issued):
  issue_mask= 1 bits=[0]        acc[0,0]=16.0000   sum=1024.0  expect 16.0
  issue_mask= 2 bits=[1]        acc[0,0]=16.0000   sum=1024.0  expect 16.0
  issue_mask= 4 bits[2]         acc[0,0]=16.0000   sum=1024.0  expect 16.0
  issue_mask= 8 bits[3]         acc[0,0]=16.0000   sum=1024.0  expect 16.0
  issue_mask=15 bits[0,1,2,3]   acc[0,0]=64.0000   sum=4096.0  expect 64.0
  all distinct=1 (uniform across the N columns)
a_base=0xc0, b_base=0x4c0, desc lo=0x10000, hi=0x40004040, idesc=0x8118010

=> PERFECT (1,1,1,1). Every single issue reads exactly its own 16 K, and four
   issues read exactly K=64. No over-consumption anywhere.

Consequences:
  * The documented model is implementable and behaves correctly on this hardware
    in this configuration. "Hardware defect" is off the table for this shape.
  * The 2.75x over-consumption (176 vs 64) measured through the CuTe path is NOT
    reproduced by raw PTX with the same geometry, same descriptor values, same
    idesc value, same start addresses, same sync, same readback.
  * Therefore the defect lies in the generated CuTe/CUTLASS path -- in what the
    vendored helper emits into PTX/SASS, or in the operands it is handed.

Decisive split from the plan has fired on the second branch:
    raw PTX 1,1,1,1  ->  problem is in the generated CUTLASS/CuTe path
                     ->  diff emitted PTX/SASS against the raw case

BISECTION PLAN (each step cheap, gate = per-k sweep must read 1.0 for all k):
  1. Same geometry (M=128 N=64 K=64, one atom, 4 issues) driven through the CuTe
     path (make_smem_layout_a + make_smem_desc_base + vendored
     gemm_ptx_precomputed_varname). If this ALREADY shows the staircase, the bug
     needs only 4 issues and a single atom -- then diff its PTX against rawptx.
  2. If step 1 is clean, go to K=128 / 8 issues (the original probe config) -- the
     bug then depends on the two-atom / 8-issue configuration.
  3. Dump both PTX streams and compare, in this order:
       - the four/eight emitted tcgen05.mma operand registers
       - the idesc immediate
       - the per-issue descriptor immediates (add.s32 offsets)
       - operand type/major metadata
  4. Regression signature to carry along: the 3.0 at f16 offsets 19/27.

## 2026-09-16 — PTX-diff infrastructure: both paths compile in one L4 run

Per the plan, raw PTX did NOT reproduce the anomaly (single issues all 16.0, four
issues 64.0, exact), so we are on the branch: diff the generated CUTLASS/CuTe PTX
against the raw kernel BEFORE any benchmarking.

New artifacts:
  ci_probe/cutepath_kernel.py   CuTe/CUTLASS-path twin at IDENTICAL geometry
                                (M=128 N=64 K=64, same one-atom SWIZZLE_128B
                                contents, 4 issues, same sync, same readback).
                                Only the operand construction differs:
                                CuTe swizzled layout + make_smem_desc_base +
                                vendored gemm_ptx_precomputed_varname.
  ci_probe/modal_probe_ptxdump.py  L4-only driver; compiles BOTH kernels in one
                                process and looks for dumped PTX.

Status of that run:
  * BOTH kernels compile cleanly in the same L4 process
    (CudaDialectJitCompiledFunction) -- so a like-for-like diff is set up.
  * `CUTE_DSL_DUMP_DIR` is a real hook in the installed package but produced no
    files as invoked; needs the directory to exist / the right form, or the
    artifact located in the compiler cache instead.
  * `CUTE_DSL_COMPILER_OPT=3` makes the compiler ICE outright
    ("failed to add cute-to-nvvm ... opt-level=3"). Directly relevant to
    perturbation B (optimisation level): at least one opt level is unusable, so
    any opt-level A/B must first bound which levels compile at all.

Also fixed along the way, worth keeping:
  * `declare_ptx_smem_desc` requires the FRAGMENT layout
    (tCrA[None,None,None,0].layout, 3 modes), not the composed outer layout --
    `crd2idx((0,0,k), la.outer)` fails on a 4-mode layout.
  * CuTeDSL kernels must be imported from a real source file; `exec()` of the
    source breaks AST parsing ("Failed to parse function").

Next actions for the diff (no B200 needed, all on L4):
  1. make CUTE_DSL_DUMP_DIR work (mkdir first) or locate the generated artifact in
     the compile cache (search for files modified during compile), then
  2. `cuobjdump -ptx` / `nvdisasm` the cubin for both kernels,
  3. compare in this order: the emitted tcgen05.mma operand registers -> the idesc
     immediate -> the per-issue descriptor immediates (add.s32 offsets) -> operand
     type/major metadata.
  4. carry the 3.0-at-f16-19/27 regression fingerprint.

Benchmarking stays gated on this: the CuTe path currently produces wrong answers,
so a vLLM comparison against it would be contaminated by construction.

## 2026-09-16 — raw reproducer CORRECTED and now rigorous (my own idesc bug found)

`Major.K == 0`, `Major.MN == 1` (vendor `Major` IntEnum). My first raw reproducer
hand-built the idesc with bits 15 and 16 SET, i.e. it declared **MN-major operands
while supplying K-major descriptors and K-major physical layout**. Its earlier
"correct" verdict was therefore NOT a valid reference for cell identity: the A and B
operands were all-ones, and a wrong major mode still traverses a constant field
bijectively and yields the same total. Same class of blind spot as the constant-data
probes earlier in this investigation.

Corrected idesc:  IDESC = (1<<4) | ((N>>3)<<17) | ((M>>4)<<24) = 0x08100010
  -> BYTE-IDENTICAL to what the CuTe path emits (checked in
     ci_probe/modal_probe_idesc.py: mma_op_to_idesc(op) == make_instr_desc(..., Major.K,
     Major.K) == 0x08100010; op.a_major_mode really is OperandMajorMode.K).

Also replaced the all-ones operand with the sentinel matrix:
  A[r, k] =  1.0 for k 0..15 | 2.0 for 16..31 | 4.0 for 32..47 | 8.0 for 48..63
so a wrong cell map OR a wrong major mode is visible in the sum.

RESULT (ci_probe/modal_probe_rawptx.py, B200):
  issue_mask= 1  acc= 16.0000  expect  16.0  OK
  issue_mask= 2  acc= 32.0000  expect  32.0  OK
  issue_mask= 4  acc= 64.0000  expect  64.0  OK
  issue_mask= 8  acc=128.0000  expect 128.0  OK
  issue_mask=15  acc=240.0000  expect 240.0  OK
  a_base=0xc0 b_base=0x4c0 desc=0x10000/0x40004040 idesc=0x08100010

=> With the SAME idesc, SAME descriptor words, SAME starts, SAME geometry and a
   data pattern that exposes cell identity, raw PTX reads exactly K elements once
   each. Raw PTX is now a rigorous known-good reference at K=64.

Open items this leaves:
  * The CuTe path at K=64 gave 64.0 -- but with ALL-ONES data, so that run only
    bounds over-consumption, NOT cell identity. It needs the sentinel test too.
  * The 176-vs-64 over-consumption was measured at K=128 (two atoms / 8 issues).
    So the defect needs either the K=128 configuration or a CuTe-path cell map
    difference that only shows with non-constant data.

NEXT (single run each, gate = sentinel sums must match):
  1. CuTe path at K=64 with the sentinel matrix -> expected 240.
  2. CuTe path at K=128 with the sentinel matrix -> expected (per 4 issues) 240
     for the first atom, 3840 for all 8.
  Whichever breaks is then diffed against rawptx at the SAME geometry, now with a
  trustworthy reference on the other side.

## 2026-09-16 — CuTe/CUTLASS helper path VALIDATED at both shapes. The staircase was a probe artifact.

ci_probe/cutepath_kernel.py (K=64) and ci_probe/cutepath_kernel128.py (K=128), run by
modal_probe_cutesentinel.py. Same CuTe path as production: make_smem_layout_a/b +
make_smem_desc_base + vendored declare_ptx_smem_desc / declare_ptx_idesc /
gemm_ptx_precomputed_varname. Sentinels on BOTH operands so cell identity, major
mode and over-consumption are all observable:
  A[r,k] = B[r,k] = 1,2,4,8,16,32,64,128 per 16-element K group
  acc[0,0] must equal sum_k A*B

RESULT (B200):
  K= 64  acc[0,0]=  1360.000  expect   1360.0  OK  uniq=1
         a_base=0xc0 b_base=0x4c0 base=0x10000/0x40004040 idesc=0x08100010
  K=128  acc[0,0]=349520.000  expect 349520.0  OK  uniq=1
         a_base=0xc0 b_base=0x8c0 base=0x10000/0x40004040 idesc=0x08100010

Cross-check: if the blocks-probe staircase (1,2,4,4; 176 read-events per 64) were
real, the K=64 sentinel sum would be
  16*(1*1) + 16*(2*4) + 16*(4*16) + 16*(4*64) = 16 + 128 + 1024 + 4096 = 5264
not 1360. It is 1360. So there is NO over-consumption in this path.

CONSEQUENCE: the CuTe/CUTLASS descriptor+layout+helper stack is correct at both
production shapes (QK K=hdim=128, PV K=tile_n=64). The 1,2,4,4 staircase and the
176-vs-64 result came from ci_probe/modal_probe_blocks.py and do NOT reproduce in
this cleaner twin. That harness is therefore the artifact, not the library -- I
cannot yet name the exact mechanism (its synthetic offset layouts, one-hot deposit,
kstep plumbing and dep_mode experiments are all differences; the one-hot pattern is
the prime suspect because it is the only one that turns "cell read twice" into a
directly visible number).

REVISED TARGET: the defect is in the PRODUCTION kernel's use of the stack
(ci_probe/modal_probe_singlekv.py dbg_mode 1 already showed the production QK wrong
with dense data: max_abs 497 vs ref 176), not in descriptor construction, deposit,
readback, the helper, or the CuTe layout algebra -- all of which are now either
validated or exonerated. The production operands differ from these twins in that Q
and K are staged from global memory and the QK K=hdim=128 uses the hd_e=(16,4,2)
nested store, while the PV uses tn_e. Next bisection: make the production kernel
deposit a known sentinel Q/K tile and check acc directly, mirroring this twin.

Also note: the first raw reproducer's idesc was wrong (bits 15/16 set = Major.MN
while supplying K-major descriptors and a K-major layout). Major.K == 0 and
Major.MN == 1 in the vendor enum. The corrected raw reproducer (idesc 0x08100010,
sentinel data) is exact at K=64.

## 2026-09-16 — PRODUCTION-SHAPED SENTINEL QK: failure REPRODUCED in a controlled harness

Per the requested design: production kernel, normal gathering SKIPPED, deterministic
sentinel Q tile and sentinel K tile written through the production hd_e=(16,4,2)
flat-scalar stores, same descriptor, same idesc, same CuTe partitioning, same
accumulator init, NO softmax, then read the accumulator and compare against the
analytic sentinel GEMM. Sentinel formula identical to the validated twin
(A=B=2**(col//16) per 16-element K group) -> a correct QK must give acc[m,n] =
16*(1+4+16+64+256+1024+4096+16384) = 349520 for EVERY (m,n).

RESULT (dbg_mode 4, ci_probe/modal_probe_singlekv.py, B200):
  acc = 1,398,032 everywhere (uniform across rows AND columns -- so no row/col
  mapping problem), expected 349,520.
  Exact decomposition:  1398032 = 16*(1*1 + 4*(4+16+64+256+1024+4096+16384))
  i.e.  k in [0,16)      read 1x
        k in [16,128)    read 4x
  Deterministic, uniform, and reproducible.

## What was eliminated in this round (all compared against the validated twin)
  * descriptor words : production a_base LBO=1 SBO=64 ltype=2, b_base LBO=1 SBO=64
                       -- byte-identical to the twin.
  * idesc            : production 0x08100010 -- identical to the twin.
  * CuTe layouts     : same make_smem_layout_a/b calls, same (128,64,128) shape args.
  * allocation       : same layout=outer + swizzle=inner + byte_alignment=1024.
  * store form       : flat scalar `col//e0` vs explicit nested tuple -> identical
                       result, so the deposit form is not it.
  * smem addresses   : perturbing the TWIN to production's exact starts
                       (a_start=7360, b_start=9408 u128) still gives 349520. So the
                       base address/phase is not it.
  * readback         : production's ld_s_atom Repetition(32) vs tile_n=64 changed
                       to Repetition(tile_n) -> result unchanged. Not it.

## Where that leaves it
The failure is reproduced in a production-shaped harness with an exact, uniform
numeric signature, yet every descriptor/layout/address/store/readback quantity
compared so far is identical to a twin that is exact. The last quantity NOT yet
compared one-for-one is the per-K-block descriptor offset list the vendored helper
derives inside the production kernel:
    crd2idx((0,0,k), sl_qk_a)  and  crd2idx((0,0,k), sl_qk_b)  for k=0..7
    plus num_k_tile = size(layout, mode=[2]) for each side
Twin values (validated): A [0,2,4,6,1024,1026,1028,1030],
                         B [0,2,4,6,512,514,516,518], num_k_tile 8 both.
Instrumentation for exactly this dump is already installed in the production kernel
(mode 4 writes it to mDbgS row 126, and the probe prints it); the run that would
have produced it stalled, so this comparison is PENDING, not done.

Profile k>=16 -> 4x with k<16 -> 1x is a strong constraint for that comparison: it
is what you would see if the K-block origin sequence covers the first 16-K block
correctly and then re-covers the remainder four times.

## 2026-09-16 — pending run DONE: every descriptor-side quantity identical. Delta is the async-proxy write-visibility contract.

The stalled run was re-run to completion (longer timeout). Results:

  production sl_qk_a crd2idx k=0..7 = [0,2,4,6,1024,1026,1028,1030]
  production sl_qk_b crd2idx k=0..7 = [0,2,4,6,512,514,516,518]
  production num_k_tile a=8 b=8
  twin      sl_qk_a = [0,2,4,6,1024,1026,1028,1030]
  twin      sl_qk_b = [0,2,4,6,512,514,516,518]
  twin      num_k_tile 8/8
  => IDENTICAL. The descriptor path is now fully eliminated, not partially.

Physical dump of production's sQ after the sentinel deposit (raw flat view, first
8192 f16 = logical row 0, k=0..63):
  row 0: [(1.0, 16), (2.0, 16), (4.0, 16), (8.0, 16)]
  row 1: [(1.0, 16), (2.0, 16), (4.0, 16), (8.0, 16)]
i.e. the 16-f16 groups of row 0 hold exactly 1,2,4,8 with 16 elements each, in the
right cells. DEPOSIT IS CORRECT (note: my first histogram check mis-read the dump --
dump rows are 64-wide physical slices, not logical rows; the same class of
instrumentation error as earlier in this file, corrected here).

So: descriptor words, idesc, offset list, num_k_tile, layout objects, allocation,
base addresses, deposit form, deposit contents and readback are ALL identical to a
twin that is exact -- yet production reads k>=16 four times.

REMAINING DELTA, and it is not a layout quantity: the **async-proxy
write-visibility contract**. Regular (generic-proxy) SMEM stores followed by a
tcgen05.mma read of that SMEM require an async-proxy fence, i.e.
`fence.proxy.async.shared::cta` (CUTLASS: `fence_view_async_shared()`), before the
MMA. Neither the production kernel nor the twin has an explicit one; the twin
happens to be correct because its schedule differs (far fewer instructions and no
competing SMEM traffic). Production's frame has many other SMEM writers (LUT copy,
packed KV load, norms, row max/sum) plus four separate descriptors, so its
generic->async ordering is different and can expose stale/partially-written cells.
This also explains the pattern being k-dependent and deterministic rather than
random: the cells that are stale are the ones the later K blocks read.

NEXT (single change, then re-run DBG=4):
  add an async-proxy fence between the SMEM staging and every tcgen05.mma in the
  production kernel -- `cute.arch.fence_view_async_shared()` (or raw
  `fence.proxy.async.shared::cta;`) immediately after the staging barriers and
  before the QK and PV gemm calls.
Gate: DBG=4 acc must become 349520 everywhere.

Also note for the record: in that last run my physical dump overwrote mDbgS rows
126/127 where the descriptor facts are written, so the descriptor numbers printed as
1s. Harmless (the descriptor comparison had already been taken from the previous
run), but the dump loops in this kernel must be range-limited to avoid clobbering
the debug rows.

## 2026-09-16 — async-proxy fence REFUTED; every measurable quantity now matches a correct twin

Three more refutations this round, all with the production sentinel QK (dbg mode 4):

1. ASYNC-PROXY FENCE. Added `fence.proxy.async.shared::cta` (raw inline asm) before
   all three tcgen05.mma sites. Result BIT-IDENTICAL: acc = 1,398,032. Memory
   ordering is not the cause, and the result is deterministic rather than racy.

2. READBACK ROUTE. Replaced mode 4's TMEM->SMEM->out dump with the twin's exact
   path (Repetition(tile_n) TMEM->register copy, explicit (row,col) placement).
   Result unchanged: 1,398,032. So the accumulator itself really is 4x high; it is
   not an artifact of production's SMEM staging.

3. sK_code PHYSICAL PLACEMENT. Dumped the raw B operand after its sentinel write.
   Observed mean = 31.875 over 8192 f16.
   Expected mean for the sentinel = (1+2+4+8+16+32+64+128)/8 = 255/8 = 31.875.
   EXACT. So the B operand's physical contents are correct too.

Complete list of quantities now proven identical between production and the exact
twin: descriptor words, idesc, offset list, num_k_tile, layout objects, allocation,
base addresses (twin perturbed to production's exact starts stays correct), deposit
form, deposit contents for BOTH operands, readback route, and memory ordering
(fenced). Production's accumulator is still 1,398,032 vs 349,520, exactly
1x for k in [0,16) and 4x for k in [16,128), uniform over the tile.

I have no remaining hypothesis that this evidence supports. The difference between
production and the twin is no longer any single measurable quantity; it is the
surrounding kernel context (additional SMEM tensors, the TMEM allocator, the
pipeline objects, occupancy, instruction schedule).

RECOMMENDED NEXT APPROACH (no hypothesis required -- bisect in the other
direction): start from the WORKING twin and add production's context one piece at a
time until it breaks:
  a. the real TmemAllocator with production's column count,
  b. production's extra SMEM tensors (sKLut/sVLut/sK_packed/sV_packed/sKNorm/
     sVNorm/sRowMax/sRowSum/sOf/sS) present and written,
  c. the PipelineUmmaAsync producers/consumers instead of the plain barriers,
  d. the second (PV) tiled_mma and its descriptor declarations in the same kernel.
Each step is a small edit to the twin (which has no defect) and the gate is the
sentinel sum: it must stay 349,520. Whichever step first breaks it localises the
cause by construction, with no guessing.

Instrumentation note: mode 5 now dumps sK_code and needs the sentinel K staging,
which is gated on `dbg_mode == 4 or dbg_mode == 5`. Also, dump loops must be
range-limited -- they previously clobbered mDbgS rows 126/127 where the descriptor
facts are recorded.

## 2026-09-16 — CORRECTION + one-hot per-K-block sweep: 4x is attached to the K OPERAND, not accumulator reuse

CORRECTION first: I previously told the user the base-4 decomposition of the
aggregate sum uniquely determined the per-block multiplicities. THAT WAS WRONG.
Base-4 digits must be < 4 and m1 = 4 is not a legal digit, so
1398032 = 16*sum_j m_j*4^j does NOT uniquely determine {m_j}. The one-hot sweep was
necessary after all.

ONE-HOT SWEEP (production, dbg_mode 4, frozen helper, one K block of data live at a
time, fresh accumulator each launch, carried block index via softmax_scale -- unused
in mode 4, so no new parameter and no new compile):

  K block 0 (k=  0.. 15): acc= 16.000  multiplicity 1.000  OK
  K block 1 (k= 16.. 31): acc= 64.000  multiplicity 4.000  BAD
  K block 2 (k= 32.. 47): acc= 64.000  multiplicity 4.000  BAD
  K block 3 (k= 48.. 63): acc= 64.000  multiplicity 4.000  BAD
  K block 4 (k= 64.. 79): acc= 64.000  multiplicity 4.000  BAD
  K block 5 (k= 80.. 95): acc= 64.000  multiplicity 4.000  BAD
  K block 6 (k= 96..111): acc= 64.000  multiplicity 4.000  BAD
  K block 7 (k=112..127): acc= 64.000  multiplicity 4.000  BAD
  multiplicities: [1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
  (twin and raw: all 1.0)

WHAT THIS SETTLES (the question asked):
  * The excess is NOT accumulator reuse and NOT async dependency semantics. Each
    launch used ONE accumulator holding ONE live K block, so there is no
    accumulation across blocks to confuse; the 4x survives anyway.
  * Nor is it issue ORDER: the sweep moved the DATA across positions 0..7, so the
    effect tracks the K-block POSITION, not which issue is first. Data placed at
    block 1 (necessarily read by issues other than issue 0) still reads 4x.
  * Therefore: 4x is attached to the K operand path, and specifically to every
    K block other than the first. Block 0 is read by exactly 1 of the 8 issues;
    blocks 1..7 are each read by exactly 4 of the 8.

WHAT IS STILL OPEN: the exact window shape. A contiguous 4-block-wide pick per
issue does not reproduce (1,4,4,4,4,4,4,4): a window of [i, i+3] would give block 1
a multiplicity of 2, and every "whole atom" or "whole K" model inflates block 0.
So it is not a simple contiguous window. It is nonetheless a K-operand-side,
position-dependent read multiplicity of exactly 4 for non-first blocks, which is
consistent with per-issue reads spanning a full 128 B SWIZZLE_128B atom (4 K
blocks / 64 f16) instead of the intended 16 elements / 32 B -- but the mismatch at
block 0 means that reading is not yet proven.

## 2026-09-16 — pair-footprint probe: multiplicities are per-block, additive, adjacency-irrelevant

production, dbg_mode 4, two live K blocks (block a at amplitude 1.0, block b at
amplitude 2.0), values used for BOTH operands so acc = 16*(m_a + 4*m_b).
Block pair carried in q_len, amplitude in softmax_scale (both unused by mode 4).

  calib single block 0    acc= 16.000   acc/16= 1.000
  calib single block 1    acc= 64.000   acc/16= 4.000
  {0,1}                   acc=272.000   acc/16=17.000   = 1 + 4*4
  {1,2}                   acc=320.000   acc/16=20.000   = 4 + 4*4
  {2,3}                   acc=320.000   acc/16=20.000   = 4 + 4*4
  {3,4}                   acc=320.000   acc/16=20.000   = 4 + 4*4
  {6,7}                   acc=320.000   acc/16=20.000   = 4 + 4*4
  {1,3} non-adjacent      acc=320.000   acc/16=20.000   = 4 + 4*4
  {0,7} distant           acc=272.000   acc/16=17.000   = 1 + 4*4

EMPIRICAL READ-ADDRESS FUNCTION (production):
      m(0) = 1
      m(j) = 4   for j >= 1
  * strictly ADDITIVE: every two-block case equals m_a + 4*m_b exactly.
  * INDEPENDENT of neighbours: no cross terms, no coupling.
  * ADJACENCY-IRRELEVANT: {0,1} == {0,7} == 272; {1,2} == {1,3} == {6,7} == 320.
  => the footprint is PER BLOCK, not a sliding/contiguous window. A window model
     [i, i+3] would couple neighbours ({0,1} would not equal {0,7}); it does not.
  => no contiguous-window model reproduces the discontinuity at block 0; the
     transformation is index-class based, not address-range based.

IMPORTANT LIMITATION (must not be glossed): an accumulator sum cannot distinguish
"stored once, read 4x" from "stored 4x, read 1x". So this result does NOT yet
localize the 4x to the MMA read vs the SMEM write.
  * sQ arrangement WAS verified earlier for row 0 / k=0..63 (blocks 0..3): exactly
    1,2,4,8 x 16 each -- correct, so blocks 0..3 are stored once in that row.
  * sK_code was checked by MULTISET ONLY (mean 31.875) -- arrangement-blind. That
    is a real gap.

NEXT DISCRIMINATOR: full arrangement dump of sK_code (not just its multiset), and
of sQ across ALL rows (only row 0 was checked). If any block j>=1 appears stored
more than once per logical (row,k), the 4x is a deposit/aliasing effect; if every
(row,k) appears exactly once, the 4x is in the MMA's K addressing and the frozen
helper/descriptor really does behave differently in production.

## 2026-09-16 — ARRANGEMENT TEST: deposit exonerated for good. 4x is in the MMA K addressing.

Mode 6: unique-per-k sentinel (value = k+1, exact in fp16), values used for BOTH
operands, full physical dump of sQ (128x128 -> mDbgO) and sK_code (64x128 -> mDbgS),
formula-free per-row permutation check:
   each logical row must contain every value 1..128 exactly once, across its two
   64-f16 swizzle atoms:
     sQ  atoms at flat 0 and 8192   (128 rows x 64 f16 per atom)
     sK  atoms at flat 0 and 4096   (mode2 stride is 4096 f16, NOT 8192 -- my first
                                     run used 8192 for sK and reported 64/64 rows
                                     bad; that was a host-side formula error, not a
                                     finding)

RESULT:
  sQ  (Q operand) : rows=128  rows_not_a_permutation_of_1..128 = 0
  sK_code (B operand): rows=64   rows_not_a_permutation_of_1..128 = 0

=> Every logical (row, k) occupies EXACTLY ONE physical slot in both operands.
   No duplication, no loss, no aliasing. A unique-value sentinel makes this a real
   placement test, unlike the earlier group-constant sentinels (and unlike the
   multiset-only sK check, which was arrangement-blind).

DECISION TREE RESOLVED (the point of this run):
   deposit-side  -> NO
   MMA K-addressing -> YES
   The 4x is created by the MMA's K read, not by the SMEM write.

State of the whole investigation, all measured:
   operands physically correct (this entry, unique-value, both operands)
   descriptor words/idesc/offset list/num_k_tile identical to a correct twin
   base addresses identical (twin perturbed to production's exact starts stays OK)
   readback route identical (twin path in production gives the same wrong value)
   async-proxy ordering fenced -> no change
   per-block multiplicity m(0)=1, m(j>=1)=4, strictly additive, adjacency-irrelevant
   and yet the same frozen helper + same descriptors + same operand contents are
   exact in the twin (349520; one-hot all 16.0) and 4x wrong in production.

Only contextual differences remain between the two kernels: production's much larger
SMEM frame with other tensors PRESENT AND WRITTEN (LUTs, packed KV, norms, row
max/sum, sOf, sS), the second (PV) tiled_mma and its descriptor declarations, and
the pipeline objects. Address magnitude and frame size are already ruled out (the
twin was perturbed to production's exact addresses and stayed correct).

NEXT: build-up bisection from the correct twin -- add production context one piece at
a time; gate = sentinel sum 349520 (and one-hot all 16.0). Order:
 1. production's extra SMEM tensors present and written,
 2. the second (PV) tiled_mma + its declare_ptx_smem_desc / declare_ptx_idesc,
 3. PipelineUmmaAsync in place of the plain barriers.
First step that breaks the gate localises the cause with no hypothesis.

## 2026-09-16 — footprint pinned at ELEMENT granularity; two bisection steps ruled out

Per-k one-hot sweep in production (single logical k at 1.0 on BOTH operands, so
acc == mult(k)); k carried in q_len, 128 launches, one compile:

  k   0.. 15: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  k  16.. 31: 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  k  32.. 47: 4 ...  (all 4)
  k  48.. 63: 4 ...
  k  64..127: 4 ...
  correct k: 16/128

=> EXACT LAW: m(k) = 1 for k < 16, m(k) = 4 for k >= 16. Uniform within each
   16-element block: no sub-block structure. The footprint is per-K16-block with a
   single discontinuity at the block-0/block-1 boundary.

Model fitting is still open: (1,4,4,4,4,4,4,4) does not fit any contiguous per-issue
window -- [i,i+3] gives (1,2,3,4,4,4,4,4); atom-aligned groups give 4 for block 0 too.
Total read events = 1 + 7*4 = 29 over 8 issues.

BISECTION STEPS RULED OUT this round (production, gate = sentinel, no change):
  1. the other SMEM writers: gating off both _copy_lut_to_smem calls and
     _load_kv_packed for the probe mode -> identical multiplicities.
  2. the PV descriptor/idesc declarations (the only shared namespace with the
     frozen helper's function-scope PTX registers) -> identical multiplicities.

INVALID EXPERIMENT, noted so it is not repeated: CUTE_DSL_COMPILER_OPT is NOT an
optimisation level -- it is injected into the pass-manager pipeline string, so "1"
is parsed as a pass name ("no such option 1") and "3" produced the earlier ICE.
Do not use it as an -O switch.

REGRESSION CHECK (important, and green): the shipped v0 mma.sync kernel was re-run
end-to-end on B200 after all of this session's edits to the package:
  8/8 shapes, 0 violations, max_abs ~1.5e-3 (d=64: 1.38e-3)
Note v0 requires m_block_size == num_threads//32*16 == 64 with 128 threads; tile-m
128 fails with a clear ValueError, i.e. a config constraint, not a regression.


## 2026-09-17 — PHASE 1 NUMBERS ON B200 (v0 mma_sync, default schedule)

Harness fix required first: ci/modal_bench_smoke.py called
launch_turboquant_attention(None, q, gathered, out, None, scale, quantizer=...)
but the launcher now reads kernel.k_packed_bytes and needs metadata
(seq_lens/query_start_loc), so the old call could not type-check. Updated the smoke
to construct TurboQuantAttentionForward + a SimpleNamespace metadata, matching the
exec probe. bench_common itself was already current.

RESULTS (B200, schedule = mma_sync, reference = dequant-to-fp16 + fp16 SDPA on the
same quantized cache; decode timed under a CUDA graph, prefill via do_bench):
  decode-short (1,1,4096)      ours= 1.4976 ms  ref= 2.5485 ms  speedup=1.702x  PASS
  decode-long  (1,1,32768)     ours=11.8935 ms  ref= 6.1414 ms  speedup=0.516x  BELOW
  prefill      (1,4096,4096)   ours=740.96 ms  ref=15.1638 ms  speedup=0.020x  BELOW

Recorded in benchmarks/results/smoke_b200.json.
  * short decode wins 1.70x even on mma_sync -> the packed-KV idea has real value.
  * long decode and prefill lose badly: prefill is ~0.75 TFLOP/s, i.e. the kernel
    re-reads/re-dequantizes per KV tile with no pipelining and no split-K. These are
    exactly the shapes a tcgen05 schedule is meant to fix.
  * Still missing for the acceptance tables: the pinned FA4 and upstream-TurboQuant
    baselines built into the image; these smoke numbers use an fp16-SDPA reference.
