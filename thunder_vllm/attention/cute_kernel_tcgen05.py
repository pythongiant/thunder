"""tcgen05 / TMEM schedule for the TurboQuant forward (opt-in).

Select with ``THUNDER_SCHEDULE=tcgen05``. The default remains the verified
cooperative ``mma.sync`` schedule in :mod:`thunder_vllm.attention.cute_kernel`.

Status
------
Compiles and RUNS on B200. Two things are fixed and verified:

* **Sync.** tcgen05 completion is now driven by
  ``PipelineUmmaAsync.create(...).make_participants()`` with
  acquire/commit on the producer handle and wait/release on the consumer
  handle. A hand-rolled ``mbarrier_init`` + ``tcgen05.commit`` +
  ``mbarrier_wait(phase)`` deadlocked (the async-proxy arrival never became
  visible; ``mbarrier_init_fence`` did not help), and ``elect_one``-wrapping the
  commit was worse. The kernel now runs every probe shape to completion.

* **The deposit is self-consistent.** ``ci_probe/modal_probe_deposit.py`` writes
  a pattern through ``cute.composition(operand, row_major)`` and recovers it
  *exactly* through the operand's own ``partition_A``/``partition_B`` (both
  operands).

Remaining defect (sharply isolated; the "mid-swizzle-unit alias" hypothesis is
REFUTED). Measured with `ci_probe/modal_probe_kblock.py` (M=128, N=64, K=32;
`cute.gemm` removed from the K traversal; FA's `declare_ptx_smem_desc` +
`gemm_ptx_precomputed_varname`).

Baseline (standard layout, `mode0=(128,16):(32,1)`, SWIZZLE_64B, K continued at
32 B), variant 1 = `A[0,0]=1, A[0,16]=2`, B=1:

| K-block descriptor advance | result |
|---|---|
| 0 | 5.0 |
| 2 units (= 32 B, the layout's K stride) | 9.0 |
| 8 units | CUDA illegal address |

Experiment A -- put the K continuation on a full swizzle unit (64 B) by building
`mode0=(128,16):(64,1)` with SWIZZLE_128B (descriptor `layout_type=2, LBO=1,
SBO=64`, `stride00=8`) -- gives **identical** values:

| advance | result |
|---|---|
| 0 | 5.0 |
| 4 units (= 64 B, now unit-aligned) | 9.0 |

So the mid-64B-unit origin is NOT the cause: unit-aligning the K continuation
changes nothing. And the values are *layout-independent*, which means the
per-K-block descriptor advance is not reaching the second K block's data at all
-- not via the layout's own K stride, and not via a unit-aligned stride.

`crd2idx((0,0,k), tCrX_layout)` (the source of the per-block offset) is a
*staged, symbolic* value (`?{div=2}` for the standard layout, `?{div=4}` for
experiment A), so its units cannot be confirmed from the trace, and the values
it produces clearly do not advance the descriptor to the intended K block.

The no-swizzle arm is structurally unreachable: the fragment profile needs a
flat mode0 `(M, K_atom)` while SWIZZLE_NONE needs MN stride = 1 uint128; with
`K_atom = 16` those self-overlap, so no valid SWIZZLE_NONE operand with this
fragment profile exists.

Kept: `smem.allocate_tensor(layout=outer, swizzle=inner)` operands. Note
`smem_desc_base_from_tensor(tensor)` disagrees with
`make_smem_desc_base(layout, swizzle)` for hand-built layouts
(`allocate_tensor` rewrites the tensor layout), so bases must be computed on the
host and passed in.

Recommended next step: replicate FA's ``flash_fwd_sm100.py::mma()`` *in full*
rather than in parts -- it drives the same helpers but with ``q_stage``-staged
operands, ``declare_ptx_smem_desc`` computed once from
``sQ[None,None,None,last_stage]``, ``tmem_s_offset`` from its TMEM layout, and
``partial(...)`` per K-block. Alternatively fill the operands with TMA from a
gmem staging buffer (`make_tiled_tma_atom_A/B`), which is what every canonical
example does and which guarantees the descriptor matches. Then re-run
``ci_probe/modal_probe_exec.py --kernel-module
thunder_vllm.attention.cute_kernel_tcgen05 --tile-m 128 --tile-n 64`` and
expect 0 violations.

Pattern
-------
Follows the canonical NVIDIA CuTeDSL Blackwell GEMM
(``examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py``):

* ``op = tcgen05.MmaF16BF16Op(dtype, acc_dtype, (M, N, 16), CtaGroup.ONE,
  OperandSource.SMEM, K, K)`` then ``tiled_mma = cute.make_tiled_mma(op)``.
* SMEM operands via ``sm100_utils.make_smem_layout_a/b(tiled_mma, (M,N,K),
  dtype, stages)`` and ``get_tensor(outer, swizzle=inner)`` — the swizzle lives
  in the *pointer*, which is what the MMA descriptor path requires.
* Accumulators are TMEM: ``make_fragment_C(partition_shape_C((M,N)))`` gives the
  layout, then the tensor is **rebound** from the retrieved TMEM pointer
  (``cute.make_tensor(tmem_ptr, frag.layout)``). Without the rebind the layout is
  a register-fragment layout and the MMA is rejected.
* ``cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)`` with A/B as the
  fragments returned by ``make_fragment_A/B``; accumulation is toggled on the
  mma with ``tiled_mma.set(tcgen05.Field.ACCUMULATE, ...)``.
* S is read back with ``tcgen05.ld`` (``Ld32x32b``) into registers and staged to
  a plain SMEM tile for the row-wise softmax; O likewise in the epilogue.

K/V stay packed with the dequantizer from :mod:`cute_kernel`; the code tiles are
written through a composed 2-D view of the swizzled SMEM layout.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils as utils_basic
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import pipeline as _pipeline
from thunder_vllm.attention._vendor import blackwell_helpers as _bh
from thunder_vllm.attention._vendor import mma_sm100_desc as _sd
from cutlass._mlir.dialects import llvm


@cute.jit
def _fence_proxy_async_shared():
    """Make generic-proxy SMEM stores visible to the tcgen05 async proxy.

    tcgen05.mma reads SMEM through the async proxy. Stores issued by the generic
    proxy (our staging / sentinel / dequant writes) are NOT guaranteed visible to
    it until `fence.proxy.async.shared::cta`. Without this the MMA can read stale
    or partially written cells.
    """
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;\n\t",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

from thunder_vllm.attention.cute_kernel import (
    _LOG2E,
    _copy_lut_to_smem,
    _dequantize_strided,
    _dequantize_transposed,
    _unpack,
    _load_kv_packed,
)
from thunder_vllm.utils.logging import get_logger

logger = get_logger("attention.cute_kernel_tcgen05")

KERNEL_STATUS = "sync-fixed-operand-k-order-mismatch"

_TMEM_MIN_COLS = 32


def _round_tmem_cols(cols: int) -> int:
    n = _TMEM_MIN_COLS
    while n < cols:
        n *= 2
    return n


# The MMA operands live in *swizzled* SMEM. Element-wise writes cannot address
# those layouts, and `cute.composition(swizzled, row_major)` is wrong because
# `composition` decomposes indices with the shape's row-major order, which does
# not match the swizzle atom's semantic (N,K) decomposition.
#
# Instead the dequant/Q stages write a plain staging tile, then a blit copies
# staging -> operand with `make_tiled_copy_A/B`, which already encode the MMA's
# own logical -> swizzled mapping. (Staging costs one extra SMEM round trip per
# tile; correctness first, fold it away later.)
@cute.jit
def qk_kind_idesc():
    """idesc the production QK actually emits (same call declare_ptx_idesc uses)."""
    op = tcgen05.MmaF16BF16Op(
        cutlass.Float16, Float32, (128, 64, 16), tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    return _bh.sm100_desc.mma_op_to_idesc(op)


@cute.jit
def _sentinel_pk(col, kcol):
    """Single-k one-hot: 1.0 at logical K index `kcol`, else 0.0.

    Finest possible footprint probe: acc == mult(kcol), so a sweep of kcol gives
    the read multiplicity at element granularity.
    """
    v = cutlass.Float16(0.0)
    if col == kcol:
        v = cutlass.Float16(1.0)
    return v


@cute.jit
def _sentinel_unique(col):
    """Unique value per logical K index: col+1 (1..128). Exact in fp16.

    Constant-per-group sentinels cannot detect placement errors; a unique value
    lets the host verify that each logical (row, k) occupies exactly one physical
    slot, with a formula-free per-row permutation test.
    """
    return cutlass.Float16(Float32(col) + Float32(1.0))


@cute.jit
def _sentinel_pair(col, code, amp):
    """Two K blocks live: block a (code//8) at amplitude 1.0, block b (code%8)
    at amplitude `amp`. Values are used for BOTH operands, so
    acc = 16*(mult_a + amp^2 * mult_b). code/amp are carried in q_len and
    softmax_scale, neither of which mode 4 uses.
    """
    v = cutlass.Float16(0.0)
    blk = col // 16
    if blk == code // 8:
        v = cutlass.Float16(1.0)
    if blk == code % 8:
        v = cutlass.Float16(amp)
    return v


@cute.jit
def _sentinel_onehot(col, scale):
    """One-hot K-block sentinel: 1.0 in K block `scale`, 0.0 elsewhere.

    `scale` is a runtime Float32 (unused by mode 4, so it carries the block index
    without adding a kernel parameter or a new compile). The accumulator is then
    16*m(j) where m(j) is that K block's read multiplicity: a correct kernel gives
    16.0 for every j.
    """
    v = cutlass.Float16(0.0)
    if cutlass.Float32(col // 16) == scale:
        v = cutlass.Float16(1.0)
    return v


@cute.jit
def _sentinel16(col):
    """Sentinel operand value: 1,2,4,8,16,32,64,128 per 16-element K group.

    Byte-for-byte the same construction that produced the validated 349520
    control (ci_probe/cutepath_kernel128.py). Depends only on the K index, so a
    correct QK with sentinel Q and sentinel K gives acc[m,n] == 349520 for every
    (m, n) in the tile.
    """
    v = cutlass.Float16(1.0)
    if col >= 16:
        v = cutlass.Float16(2.0)
    if col >= 32:
        v = cutlass.Float16(4.0)
    if col >= 48:
        v = cutlass.Float16(8.0)
    if col >= 64:
        v = cutlass.Float16(16.0)
    if col >= 80:
        v = cutlass.Float16(32.0)
    if col >= 96:
        v = cutlass.Float16(64.0)
    if col >= 112:
        v = cutlass.Float16(128.0)
    return v


@cute.jit
def _nested_store(t: cute.Tensor, row: Int32, col: Int32, val,
                  e0: cutlass.Constexpr[int], e1: cutlass.Constexpr[int],
                  e2: cutlass.Constexpr[int]):
    """Write one element with the operand's *semantic* nested coordinate.

    `cute.composition` cannot be used here: its flat index ``row*cols + col``
    decomposes through the tensor's post-`allocate_tensor` shape, whose K factors
    interleave in the wrong order. Passing the coordinate explicitly is the only
    thing the probe proved correct (see ci_probe/modal_probe_kblock.py).
    """
    # Flat scalar into the K-block mode. This is the ONLY store form validated as
    # correct (ci_probe/cutepath_kernel*.py, exact at K=64 and K=128): CuTe
    # decomposes the scalar over the mode's actual (possibly rewritten) shape,
    # whereas an explicit nested tuple hardcodes an (e1, e2) split that
    # allocate_tensor may not preserve.
    t[((row, col % e0), 0, col // e0, 0)] = val


@cute.jit
def _dequantize_nested(sPacked: cute.Tensor, sLut: cute.Tensor, sCode: cute.Tensor,
                       bits: cutlass.Constexpr[int], head_dim: cutlass.Constexpr[int],
                       rows: cutlass.Constexpr[int], tidx: Int32,
                       num_threads: cutlass.Constexpr[int],
                       e0: cutlass.Constexpr[int], e1: cutlass.Constexpr[int],
                       e2: cutlass.Constexpr[int]):
    """`_dequantize_strided` writing through `_nested_store`."""
    total: cutlass.Constexpr[int] = rows * head_dim
    iters: cutlass.Constexpr[int] = (total + num_threads - 1) // num_threads
    for e in cutlass.range_constexpr(iters):
        idx = tidx + e * num_threads
        if idx < total:
            row = idx // head_dim
            col = idx % head_dim
            _nested_store(sCode, row, col, sLut[_unpack(sPacked, row, col, bits), col],
                          e0, e1, e2)


@cute.jit
def _dequantize_nested_transposed(sPacked: cute.Tensor, sLut: cute.Tensor,
                                  sCode: cute.Tensor, bits: cutlass.Constexpr[int],
                                  head_dim: cutlass.Constexpr[int],
                                  rows: cutlass.Constexpr[int], tidx: Int32,
                                  num_threads: cutlass.Constexpr[int],
                                  e0: cutlass.Constexpr[int],
                                  e1: cutlass.Constexpr[int],
                                  e2: cutlass.Constexpr[int]):
    """`_dequantize_transposed` writing through `_nested_store`."""
    total: cutlass.Constexpr[int] = rows * head_dim
    iters: cutlass.Constexpr[int] = (total + num_threads - 1) // num_threads
    for e in cutlass.range_constexpr(iters):
        idx = tidx + e * num_threads
        if idx < total:
            row = idx // head_dim
            col = idx % head_dim
            _nested_store(sCode, col, row,
                          sLut[_unpack(sPacked, row, col, bits), col], e0, e1, e2)


_BLIT_BITS = 16


@cute.jit
def _flat(x: cute.Tensor):
    """Collapse a (partitioned) tensor to a single mode.

    `make_tiled_copy_A/B` partitions iterate the MMA operand's logical index
    space, so the staging (plain) and operand (swizzled) partitions cover the
    same elements but differ in rank. Flattening both makes `cute.copy` accept
    them and pairs element i in the staging tile with element i of the operand.
    """
    return cute.group_modes(x, 0, cute.rank(x))


class ThunderAttentionForward:
    """tcgen05/TMEM TurboQuant forward (cooperative, two-pass).

    Same ``__call__`` contract as the default schedule, so the launcher in
    :mod:`thunder_vllm.attention.cute_kernel` and the backend are
    schedule-agnostic.
    """

    def __init__(
        self,
        head_dim: int,
        K_BITS: cutlass.Constexpr[int],
        V_BITS: cutlass.Constexpr[int],
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 64,
        num_stages: int = 1,
        num_dequant_stages: int = 1,
        num_threads: int = 128,
        use_2cta_instrs: bool = False,
        q_stage: cutlass.Constexpr[int] = 1,
        variant_tag: cutlass.Constexpr[int] = 0,
    ):
        self.head_dim = int(head_dim)
        if self.head_dim % 16 != 0:
            raise ValueError(f"head_dim must be a multiple of 16, got {self.head_dim}")
        self.head_dim_padded = self.head_dim
        self.K_BITS = K_BITS
        self.V_BITS = V_BITS
        self.k_packed_bytes = (head_dim * K_BITS + 7) // 8
        self.v_packed_bytes = (head_dim * V_BITS + 7) // 8
        self.qhead_per_kvhead = int(qhead_per_kvhead)
        self.is_causal = bool(is_causal)
        self.num_threads = int(num_threads)
        self.num_warps = self.num_threads // 32
        self.variant_tag = variant_tag

        self.tile_m = int(m_block_size)
        self.tile_n = int(n_block_size)
        self.tile_hdim = self.head_dim_padded
        if use_2cta_instrs:
            raise NotImplementedError("2-CTA tcgen05 is not implemented")
        if self.tile_m not in (64, 128):
            raise ValueError(f"tcgen05 tile_m must be 64 or 128, got {self.tile_m}")

        self.tmem_s_cols = self.tile_n
        self.tmem_o_cols = self.tile_hdim
        self.tmem_total_cols = _round_tmem_cols(self.tmem_s_cols + self.tmem_o_cols)
        self.tmem_o_offset = self.tmem_s_cols

    # ------------------------------------------------------------------
    def _mma_op_qk(self):
        return tcgen05.MmaF16BF16Op(
            cutlass.Float16,
            Float32,
            (self.tile_m, self.tile_n, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )

    def _mma_op_pv(self):
        return tcgen05.MmaF16BF16Op(
            cutlass.Float16,
            Float32,
            (self.tile_m, self.tile_hdim, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )

    def _get_tiled_mma(self):
        return (
            cute.make_tiled_mma(self._mma_op_qk()),
            cute.make_tiled_mma(self._mma_op_pv()),
        )

    @cute.jit
    def _smem_layouts(self, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma):
        return {
            "sQ": sm100_utils.make_smem_layout_a(
                tiled_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim),
                cutlass.Float16, 1,
            ),
            "sK_code": sm100_utils.make_smem_layout_b(
                tiled_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim),
                cutlass.Float16, 1,
            ),
            "sP": sm100_utils.make_smem_layout_a(
                tiled_mma_pv, (self.tile_m, self.tile_hdim, self.tile_n),
                cutlass.Float16, 1,
            ),
            "sV_code": sm100_utils.make_smem_layout_b(
                tiled_mma_pv, (self.tile_m, self.tile_hdim, self.tile_n),
                cutlass.Float16, 1,
            ),
        }

    @cute.jit
    def _get_shared_storage_cls(self, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma):
        dt = cutlass.Float16
        u8 = cutlass.Uint8
        f32 = Float32
        tm, tn, hd = self.tile_m, self.tile_n, self.tile_hdim
        L = self._smem_layouts(tiled_mma_qk, tiled_mma_pv)

        @cute.struct
        class SharedStorage:
            sOf: cute.struct.Align[cute.struct.MemRange[f32, tm * hd], 1024]
            sS: cute.struct.Align[cute.struct.MemRange[f32, tm * tn], 1024]
            sKLut: cute.struct.Align[cute.struct.MemRange[dt, (1 << self.K_BITS) * hd], 128]
            sVLut: cute.struct.Align[cute.struct.MemRange[dt, (1 << self.V_BITS) * hd], 128]
            sK_packed: cute.struct.Align[cute.struct.MemRange[u8, tn * self.k_packed_bytes], 128]
            sV_packed: cute.struct.Align[cute.struct.MemRange[u8, tn * self.v_packed_bytes], 128]
            sKNorm: cute.struct.Align[cute.struct.MemRange[dt, tn], 128]
            sVNorm: cute.struct.Align[cute.struct.MemRange[dt, tn], 128]
            sRowMax: cute.struct.Align[cute.struct.MemRange[f32, tm], 128]
            sRowSum: cute.struct.Align[cute.struct.MemRange[f32, tm], 128]
            tmem_holding_buf: Int32
            mbar_s: cute.struct.MemRange[Int64, 2]

        return SharedStorage

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKN: cute.Tensor,
        mVN: cute.Tensor,
        mKLut: cute.Tensor,
        mVLut: cute.Tensor,
        mO: cute.Tensor,
        mSeqLens: cute.Tensor,
        mQStart: cute.Tensor,
        softmax_scale: Float32,
        stream=None,
        mDbgS=None,
        mDbgO=None,
        dbg_mode: cutlass.Constexpr[int] = 0,
    ):
        q_perm = [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=q_perm))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=q_perm))
        kv_perm = [0, 2, 1]
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=kv_perm))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=kv_perm))

        num_reqs = mSeqLens.shape[0]
        num_q_heads = mQ.shape[2]
        num_q_blocks = cute.ceil_div(mQ.shape[0], self.tile_m)

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls(tiled_mma_qk, tiled_mma_pv)

        # tcgen05 SMEM descriptor bases. `smem_desc_base_from_tensor(tensor)`
        # gives the WRONG base here (allocate_tensor rewrites the tensor layout
        # so `layout[0]` is not the intended operand mode), so derive them from
        # the layout we built ourselves -- exactly as the probe proved.
        _L = self._smem_layouts(tiled_mma_qk, tiled_mma_pv)

        def _base(name):
            lay = _L[name]
            return _sd.make_smem_desc_base(
                cute.recast_layout(128, cutlass.Float16.width, lay.outer[0]),
                lay.inner, _sd.Major.K)

        q_a_base = _base("sQ")        # Q -> A of QK
        k_b_base = _base("sK_code")   # K -> B of QK
        v_b_base = _base("sV_code")   # V -> B of PV
        p_a_base = _base("sP")        # P -> A of PV

        self.kernel(
            mQ, mK, mV, mKN, mVN, mKLut, mVLut, mO, mSeqLens, mQStart,
            softmax_scale, tiled_mma_qk, tiled_mma_pv, SharedStorage,
            q_a_base, k_b_base, v_b_base, p_a_base, stream,
            mDbgS, mDbgO, dbg_mode,
        ).launch(
            grid=(num_q_blocks, num_q_heads, num_reqs),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKN: cute.Tensor,
        mVN: cute.Tensor,
        mKLut: cute.Tensor,
        mVLut: cute.Tensor,
        mO: cute.Tensor,
        mSeqLens: cute.Tensor,
        mQStart: cute.Tensor,
        softmax_scale: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        q_a_base: cutlass.Constexpr[int],
        k_b_base: cutlass.Constexpr[int],
        v_b_base: cutlass.Constexpr[int],
        p_a_base: cutlass.Constexpr[int],
        stream=None,
        mDbgS=None,
        mDbgO=None,
        dbg_mode: cutlass.Constexpr[int] = 0,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_block = cute.arch.block_idx()[0]
        q_head = cute.arch.block_idx()[1]
        req = cute.arch.block_idx()[2]
        kv_head = q_head // self.qhead_per_kvhead

        q_start = mQStart[req]
        q_len = mQStart[req + 1] - q_start
        kv_len = mSeqLens[req]

        L = self._smem_layouts(tiled_mma_qk, tiled_mma_pv)
        smem = utils_basic.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Operands allocated exactly as the reference does: layout=outer with
        # swizzle=inner, so the swizzle lives in the pointer.
        sQ = smem.allocate_tensor(
            element_type=cutlass.Float16, layout=L["sQ"].outer,
            byte_alignment=1024, swizzle=L["sQ"].inner,
        )
        sK_code = smem.allocate_tensor(
            element_type=cutlass.Float16, layout=L["sK_code"].outer,
            byte_alignment=1024, swizzle=L["sK_code"].inner,
        )
        sV_code = smem.allocate_tensor(
            element_type=cutlass.Float16, layout=L["sV_code"].outer,
            byte_alignment=1024, swizzle=L["sV_code"].inner,
        )
        sP = smem.allocate_tensor(
            element_type=cutlass.Float16, layout=L["sP"].outer,
            byte_alignment=1024, swizzle=L["sP"].inner,
        )
        sS = storage.sS.get_tensor(
            cute.make_layout((self.tile_m, self.tile_n), stride=(self.tile_n, 1)))
        sK_packed = storage.sK_packed.get_tensor(
            cute.make_layout((self.tile_n, self.k_packed_bytes)))
        sV_packed = storage.sV_packed.get_tensor(
            cute.make_layout((self.tile_n, self.v_packed_bytes)))
        sKLut = storage.sKLut.get_tensor(
            cute.make_layout(((1 << self.K_BITS), self.tile_hdim)))
        sVLut = storage.sVLut.get_tensor(
            cute.make_layout(((1 << self.V_BITS), self.tile_hdim)))
        sKNorm = storage.sKNorm.get_tensor(cute.make_layout((self.tile_n,)))
        sVNorm = storage.sVNorm.get_tensor(cute.make_layout((self.tile_n,)))
        sRowMax = storage.sRowMax.get_tensor(cute.make_layout((self.tile_m,)))
        sRowSum = storage.sRowSum.get_tensor(cute.make_layout((self.tile_m,)))
        sOf = storage.sOf.get_tensor(
            cute.make_layout((self.tile_m, self.tile_hdim), stride=(self.tile_hdim, 1)))

        # Plain staging tiles: written element-wise, then blitted to the
        # swizzled operands. sK_stage doubles as the P staging tile.
        # K-factor extents of each operand's outer shape, derived from the row
        # width: hdim/tn f16 per row -> 16-f16 atoms, grouped into swizzle atoms.
        if cutlass.const_expr(self.tile_hdim > 64):
            hd_e = (16, self.tile_hdim // 32, 2)
        else:
            hd_e = (16, self.tile_hdim // 16, 1)
        if cutlass.const_expr(self.tile_n > 64):
            tn_e = (16, self.tile_n // 32, 2)
        else:
            tn_e = (16, self.tile_n // 16, 1)

        # Blitting staging -> swizzled operand: apply the MMA's *own* operand
        # partitioning to both tensors. They represent the same logical tile, so
        # the partitions are congruent and `cute.copy` maps element-for-element
        # (the swizzle is applied by the operand's pointer).

        # MMA-completion pipeline, exactly as the canonical Blackwell GEMM does
        # it (fp16_gemm_0.py): `make_participants()` + acquire/commit on the
        # producer handle and wait/release on the consumer handle. The pipeline
        # performs the UMMA/async-proxy mbarrier arrival, which is what makes the
        # tcgen05 completion observable (a hand-rolled mbarrier + tcgen05.commit
        # never arrived and the consumer spun forever).
        s_prod, s_cons = _pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=_pipeline.CooperativeGroup(_pipeline.Agent.Thread),
            consumer_group=_pipeline.CooperativeGroup(
                _pipeline.Agent.Thread, self.num_threads),
            barrier_storage=storage.mbar_s.data_ptr(),
        ).make_participants()

        # --- TMEM allocation --------------------------------------------

        tmem_alloc_barrier = _pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )
        tmem = utils_basic.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(self.tmem_total_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        tCrAq = tiled_mma_qk.make_fragment_A(sQ)
        tCrBk = tiled_mma_qk.make_fragment_B(sK_code)
        tCrP = tiled_mma_pv.make_fragment_A(sP)
        tCrV = tiled_mma_pv.make_fragment_B(sV_code)

        # tcgen05 descriptor path (no cute.gemm). The per-K-block descriptor
        # origins come from the real fragment layouts, and the bases are the
        # host-computed ones (see __call__).
        sl_qk_a = tCrAq[None, None, None, 0].layout
        sl_qk_b = tCrBk[None, None, None, 0].layout
        sl_pv_a = tCrP[None, None, None, 0].layout
        sl_pv_b = tCrV[None, None, None, 0].layout
        q_a_start = _sd.make_smem_desc_start_addr(sQ[None, None, None, 0].iterator)
        k_b_start = _sd.make_smem_desc_start_addr(sK_code[None, None, None, 0].iterator)
        p_a_start = _sd.make_smem_desc_start_addr(sP[None, None, None, 0].iterator)
        v_b_start = _sd.make_smem_desc_start_addr(sV_code[None, None, None, 0].iterator)
        qk_kind = _bh._tcgen05_mma_kind(tiled_mma_qk.op)
        pv_kind = _bh._tcgen05_mma_kind(tiled_mma_pv.op)
        _bh.declare_ptx_smem_desc(q_a_start, q_a_base, sl_qk_a,
                                  var_name_prefix="tq_qk_a")
        _bh.declare_ptx_idesc(tiled_mma_qk.op, var_name="tq_qk_idesc")
        _bh.declare_ptx_smem_desc(p_a_start, p_a_base, sl_pv_a,
                                  var_name_prefix="tq_pv_a")
        _bh.declare_ptx_idesc(tiled_mma_pv.op, var_name="tq_pv_idesc")

        # TMEM accumulators, rebound from the retrieved pointer.
        tS_frag = tiled_mma_qk.make_fragment_C(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        )
        tS = cute.make_tensor(tmem_ptr, tS_frag.layout)
        tO_frag = tiled_mma_pv.make_fragment_C(
            tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim))
        )
        tO = cute.make_tensor(tmem_ptr + self.tmem_o_offset, tO_frag.layout)
        cute.arch.barrier()

        # --- Q tile -----------------------------------------------------
        _copy_lut_to_smem(mKLut, sKLut, tidx, self.K_BITS, self.tile_hdim, self.num_threads)
        _copy_lut_to_smem(mVLut, sVLut, tidx, self.V_BITS, self.tile_hdim, self.num_threads)
        mQh = mQ[None, None, q_head]
        mOh = mO[None, None, q_head]
        q_total: cutlass.Constexpr[int] = self.tile_m * self.tile_hdim
        q_iters: cutlass.Constexpr[int] = (q_total + self.num_threads - 1) // self.num_threads
        for e in cutlass.range_constexpr(q_iters):
            idx = tidx + e * self.num_threads
            if idx < q_total:
                row = idx // self.tile_hdim
                col = idx % self.tile_hdim
                if const_expr(dbg_mode == 6):
                    _nested_store(sQ, row, col, _sentinel_unique(col),
                                  hd_e[0], hd_e[1], hd_e[2])
                elif const_expr(dbg_mode == 4):
                    _nested_store(sQ, row, col, _sentinel_pk(col, q_len),
                                  hd_e[0], hd_e[1], hd_e[2])
                elif row < q_len:
                    _nested_store(sQ, row, col,
                                  mQh[q_start + q_block * self.tile_m + row, col],
                                  hd_e[0], hd_e[1], hd_e[2])
                else:
                    _nested_store(sQ, row, col, cutlass.Float16(0.0),
                                  hd_e[0], hd_e[1], hd_e[2])
        if tidx < self.tile_m:
            sRowMax[tidx] = -Float32.inf
            sRowSum[tidx] = Float32(0.0)
        cute.arch.barrier()
        cute.arch.barrier()

        # --- TMEM -> register -> SMEM staging for S ----------------------
        # Repetition must cover the accumulator's N extent (tile_n columns); the
        # validated twin uses Repetition(N). With Repetition(32) against N=64 the
        # TMEM->SMEM copy covers half the columns and the partition repeats values,
        # so the dumped S is not the accumulator.
        ld_s_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tile_n)), Float32
        )
        # Whole-tile readback in the rank-2 C basis (the pattern validated by
        # ci_probe/modal_probe_sync.py; the epilogue-sub-tiled variant into SMEM
        # scrambled the tile).
        tS2 = tS[(None, None), 0, 0]
        tScS2 = thr_mma_qk.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_n)))[(None, None), 0, 0]
        sS2 = thr_mma_qk.partition_C(sS)[(None, None), 0, 0]
        tmem_copy_S = tcgen05.make_tmem_copy(ld_s_atom, tS2)
        thr_ld_s = tmem_copy_S.get_slice(tidx)
        tDtS = thr_ld_s.partition_S(tS2)
        tDsS = thr_ld_s.partition_D(sS2)
        rS = cute.make_rmem_tensor(thr_ld_s.partition_D(tScS2).shape, Float32)

        n_tiles = cute.ceil_div(kv_len, self.tile_n)
        q_row_base = q_start + q_block * self.tile_m

        # ============================ PASS 1: row max ====================
        for nt in cutlass.range(n_tiles, unroll=1):
            _load_kv_packed(mK, mV, mKN, mVN, sK_packed, sV_packed, sKNorm, sVNorm,
                            kv_head, Int32(0), nt, kv_len, tidx, self,
                            want_v=False)
            cute.arch.barrier()
            if const_expr(dbg_mode == 6):
                _kt6: cutlass.Constexpr[int] = self.tile_n * self.tile_hdim
                for _e6 in cutlass.range_constexpr(
                        (_kt6 + self.num_threads - 1) // self.num_threads):
                    _i6 = tidx + _e6 * self.num_threads
                    if _i6 < _kt6:
                        _nested_store(sK_code, _i6 // self.tile_hdim,
                                      _i6 % self.tile_hdim,
                                      _sentinel_unique(_i6 % self.tile_hdim),
                                      hd_e[0], hd_e[1], hd_e[2])
            elif const_expr(dbg_mode == 4 or dbg_mode == 5):
                _ktot: cutlass.Constexpr[int] = self.tile_n * self.tile_hdim
                for _e in cutlass.range_constexpr(
                        (_ktot + self.num_threads - 1) // self.num_threads):
                    _i = tidx + _e * self.num_threads
                    if _i < _ktot:
                        _nested_store(sK_code, _i // self.tile_hdim,
                                      _i % self.tile_hdim,
                                      _sentinel_pk(_i % self.tile_hdim, q_len),
                                      hd_e[0], hd_e[1], hd_e[2])
            else:
                _dequantize_nested(sK_packed, sKLut, sK_code, self.K_BITS,
                                   self.tile_hdim, self.tile_n, tidx, self.num_threads,
                                   hd_e[0], hd_e[1], hd_e[2])
            cute.arch.barrier()
            cute.arch.barrier()
            _fence_proxy_async_shared()
            _bh.gemm_ptx_precomputed_varname(
                tS.iterator.toint(), k_b_start,
                smem_desc_base_b=k_b_base, tCrB_layout=sl_qk_b,
                smem_var_name_prefix="tq_qk_a", idesc_var_name="tq_qk_idesc",
                smem_offset=0, zero_init=True, cta_group=1, kind=qk_kind)
            if warp_idx == 0:
                s_handle = s_prod.acquire_and_advance()
                s_handle.commit()
            s_full = s_cons.wait_and_advance()
            s_full.release()
            cute.arch.barrier()
            cute.copy(ld_s_atom, tDtS, rS)
            cute.autovec_copy(rS, tDsS)
            cute.arch.barrier()
            if tidx < self.tile_m:
                cur = -Float32.inf
                for n in cutlass.range_constexpr(self.tile_n):
                    kv = nt * self.tile_n + n
                    val = sS[tidx, n] * Float32(sKNorm[n]) * softmax_scale
                    if _valid_tc(kv, tidx, kv_len, q_len, q_row_base, self.is_causal):
                        cur = cute.arch.fmax(cur, val)
                sRowMax[tidx] = cute.arch.fmax(sRowMax[tidx], cur)
            cute.arch.barrier()

        if const_expr(dbg_mode == 5):
            # Physical dump of sK_code (the B operand) AFTER its sentinel write.
            # sQ was validated earlier; the B operand never was.
            sK_flat = cute.make_tensor(
                sK_code.iterator, cute.make_layout(self.tile_n * self.tile_hdim))
            _kt: cutlass.Constexpr[int] = self.tile_n * self.tile_hdim
            for _qe in cutlass.range_constexpr(
                    (_kt + self.num_threads - 1) // self.num_threads):
                _qk = tidx + _qe * self.num_threads
                if _qk < _kt:
                    mDbgS[_qk // self.tile_n, _qk % self.tile_n] = Float32(sK_flat[_qk])
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr, self.tmem_total_cols)
            return

        if const_expr(dbg_mode == 6):
            # Full arrangement dump: sQ -> mDbgO (16384), sK_code -> mDbgS (8192).
            sQf = cute.make_tensor(sQ.iterator,
                                   cute.make_layout(self.tile_m * self.tile_hdim))
            sKf = cute.make_tensor(sK_code.iterator,
                                   cute.make_layout(self.tile_n * self.tile_hdim))
            for _e in cutlass.range_constexpr(
                    (self.tile_m * self.tile_hdim + self.num_threads - 1) // self.num_threads):
                _q = tidx + _e * self.num_threads
                if _q < self.tile_m * self.tile_hdim:
                    mDbgO[_q // self.tile_hdim, _q % self.tile_hdim] = Float32(sQf[_q])
            for _e in cutlass.range_constexpr(
                    (self.tile_n * self.tile_hdim + self.num_threads - 1) // self.num_threads):
                _q = tidx + _e * self.num_threads
                if _q < self.tile_n * self.tile_hdim:
                    mDbgS[_q // self.tile_n, _q % self.tile_n] = Float32(sKf[_q])
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr, self.tmem_total_cols)
            return

        if const_expr(dbg_mode == 4):
            # Read the accumulator through the TWIN's path: Repetition(tile_n)
            # TMEM->register copy with explicit (row, col) placement. No SMEM
            # staging involved. The only structural difference from the validated
            # twin, which is enough to decide whether the accumulator itself is
            # wrong or only production's TMEM->SMEM staging is.
            ld_d = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tile_n)), Float32)
            tc_d = tcgen05.make_tmem_copy(ld_d, tS2)
            thr_d = tc_d.get_slice(tidx)
            tD_d = thr_d.partition_S(tS2)
            id_d = thr_mma_qk.partition_C(
                cute.make_identity_tensor((self.tile_m, self.tile_n)))[(None, None), 0, 0]
            rD = cute.make_rmem_tensor(thr_d.partition_D(id_d).shape, Float32)
            cute.arch.fence_view_async_tmem_load()
            cute.copy(ld_d, tD_d, rD)
            rFd = cute.group_modes(rD, 0, cute.rank(rD))
            if tidx < self.tile_m:
                for _n in cutlass.range_constexpr(self.tile_n):
                    mDbgS[tidx, _n] = rFd[_n]
            if tidx == 0:
                mDbgS[126, 20] = Float32(cute.size(rFd))
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr, self.tmem_total_cols)
            return

        if const_expr(dbg_mode == 1):
            # Diagnostic: dump the raw QK accumulator for the (single) KV tile
            # and stop, so the host can diff it against Q @ dequant(K)^T without
            # softmax or PV in the picture.
            _dbg_tot: cutlass.Constexpr[int] = self.tile_m * self.tile_n
            for e in cutlass.range_constexpr(
                    (_dbg_tot + self.num_threads - 1) // self.num_threads):
                i = tidx + e * self.num_threads
                if i < _dbg_tot:
                    mDbgS[i // self.tile_n, i % self.tile_n] = sS[i // self.tile_n, i % self.tile_n]
        if const_expr(dbg_mode == 4):
            # descriptor facts, written into the last row of the S dump
            if tidx == 0:
                for _q in cutlass.range_constexpr(8):
                    mDbgS[126, _q] = Float32(cute.crd2idx((0, 0, _q), sl_qk_a))
                    mDbgS[126, 8 + _q] = Float32(cute.crd2idx((0, 0, _q), sl_qk_b))
                mDbgS[126, 16] = Float32(cute.size(sl_qk_a, mode=[2]))
                mDbgS[126, 17] = Float32(cute.size(sl_qk_b, mode=[2]))
                mDbgS[127, 0] = Float32(q_a_base & 0x3FFF)
                mDbgS[127, 1] = Float32((q_a_base >> 16) & 0x3FFF)
                mDbgS[127, 2] = Float32((q_a_base >> 32) & 0x3FFF)
                mDbgS[127, 3] = Float32((q_a_base >> 61) & 7)
                mDbgS[127, 4] = Float32((k_b_base >> 16) & 0x3FFF)
                mDbgS[127, 5] = Float32((k_b_base >> 32) & 0x3FFF)
                mDbgS[127, 6] = Float32(qk_kind_idesc())
                mDbgS[127, 7] = Float32(q_a_start)
                mDbgS[127, 8] = Float32(k_b_start)

            # Early exit still has to release the TMEM allocation: CuTeDSL traps
            # with "tensor memory not completely freed" if main() returns with a
            # live allocation.
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr, self.tmem_total_cols)
            return

        # ============================ PASS 2: softmax + PV ===============
        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        for nt in cutlass.range(n_tiles, unroll=1):
            _load_kv_packed(mK, mV, mKN, mVN, sK_packed, sV_packed, sKNorm, sVNorm,
                            kv_head, Int32(0), nt, kv_len, tidx, self,
                            want_v=True)
            cute.arch.barrier()
            _dequantize_nested(sK_packed, sKLut, sK_code, self.K_BITS,
                               self.tile_hdim, self.tile_n, tidx, self.num_threads,
                               hd_e[0], hd_e[1], hd_e[2])
            _dequantize_nested_transposed(sV_packed, sVLut, sV_code, self.V_BITS,
                                          self.tile_hdim, self.tile_n, tidx,
                                          self.num_threads,
                                          tn_e[0], tn_e[1], tn_e[2])
            cute.arch.barrier()

            cute.arch.barrier()
            _fence_proxy_async_shared()
            _bh.gemm_ptx_precomputed_varname(
                tS.iterator.toint(), k_b_start,
                smem_desc_base_b=k_b_base, tCrB_layout=sl_qk_b,
                smem_var_name_prefix="tq_qk_a", idesc_var_name="tq_qk_idesc",
                smem_offset=0, zero_init=True, cta_group=1, kind=qk_kind)
            if warp_idx == 0:
                s_handle = s_prod.acquire_and_advance()
                s_handle.commit()
            s_full = s_cons.wait_and_advance()
            s_full.release()
            cute.arch.barrier()
            cute.copy(ld_s_atom, tDtS, rS)
            cute.autovec_copy(rS, tDsS)
            cute.arch.barrier()
            if tidx < self.tile_m:
                acc = Float32(0.0)
                for n in cutlass.range_constexpr(self.tile_n):
                    kv = nt * self.tile_n + n
                    p = Float32(0.0)
                    if const_expr(dbg_mode == 2):
                        # Deterministic P: no softmax, exactly known on the host.
                        p = Float32((((tidx + n) % 4) + 1)) * Float32(0.25)
                    elif _valid_tc(kv, tidx, kv_len, q_len, q_row_base, self.is_causal):
                        x = sS[tidx, n] * Float32(sKNorm[n]) * softmax_scale
                        p = cute.math.exp2((x - sRowMax[tidx]) * _LOG2E, fastmath=True)
                    _nested_store(sP, tidx, n,
                                  (p * Float32(sVNorm[n])).to(cutlass.Float16),
                                  tn_e[0], tn_e[1], tn_e[2])
                    acc += p
                sRowSum[tidx] = sRowSum[tidx] + acc
            cute.arch.barrier()
            cute.arch.barrier()
            _fence_proxy_async_shared()
            _bh.gemm_ptx_precomputed_varname(
                tO.iterator.toint(), v_b_start,
                smem_desc_base_b=v_b_base, tCrB_layout=sl_pv_b,
                smem_var_name_prefix="tq_pv_a", idesc_var_name="tq_pv_idesc",
                smem_offset=0, zero_init=(nt == 0), cta_group=1, kind=pv_kind)
            if warp_idx == 0:
                s_handle = s_prod.acquire_and_advance()
                s_handle.commit()
            s_full = s_cons.wait_and_advance()
            s_full.release()
            cute.arch.barrier()

        # ============================ Epilogue ==========================
        ld_o_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tO2 = tO[(None, None), 0, 0]
        tOcO2 = thr_mma_pv.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_hdim)))[(None, None), 0, 0]
        sOf2 = thr_mma_pv.partition_C(sOf)[(None, None), 0, 0]
        tmem_copy_O = tcgen05.make_tmem_copy(ld_o_atom, tO2)
        thr_ld_o = tmem_copy_O.get_slice(tidx)
        tDtO = thr_ld_o.partition_S(tO2)
        tDsO = thr_ld_o.partition_D(sOf2)
        rO = cute.make_rmem_tensor(thr_ld_o.partition_D(tOcO2).shape, Float32)
        cute.arch.fence_view_async_tmem_load()
        cute.copy(ld_o_atom, tDtO, rO)
        cute.autovec_copy(rO, tDsO)
        cute.arch.barrier()
        if const_expr(dbg_mode >= 2):
            # Diagnostic: dump the raw PV accumulator (pre row-sum, pre rotation).
            _odb: cutlass.Constexpr[int] = self.tile_m * self.tile_hdim
            for e in cutlass.range_constexpr(
                    (_odb + self.num_threads - 1) // self.num_threads):
                i = tidx + e * self.num_threads
                if i < _odb:
                    mDbgO[i // self.tile_hdim, i % self.tile_hdim] = sOf[i // self.tile_hdim, i % self.tile_hdim]
        if const_expr(dbg_mode == 4):
            # descriptor facts, written into the last row of the S dump
            if tidx == 0:
                for _q in cutlass.range_constexpr(8):
                    mDbgS[126, _q] = Float32(cute.crd2idx((0, 0, _q), sl_qk_a))
                    mDbgS[126, 8 + _q] = Float32(cute.crd2idx((0, 0, _q), sl_qk_b))
                mDbgS[126, 16] = Float32(cute.size(sl_qk_a, mode=[2]))
                mDbgS[126, 17] = Float32(cute.size(sl_qk_b, mode=[2]))
                mDbgS[127, 0] = Float32(q_a_base & 0x3FFF)
                mDbgS[127, 1] = Float32((q_a_base >> 16) & 0x3FFF)
                mDbgS[127, 2] = Float32((q_a_base >> 32) & 0x3FFF)
                mDbgS[127, 3] = Float32((q_a_base >> 61) & 7)
                mDbgS[127, 4] = Float32((k_b_base >> 16) & 0x3FFF)
                mDbgS[127, 5] = Float32((k_b_base >> 32) & 0x3FFF)
                mDbgS[127, 6] = Float32(qk_kind_idesc())
                mDbgS[127, 7] = Float32(q_a_start)
                mDbgS[127, 8] = Float32(k_b_start)

            # Early exit still has to release the TMEM allocation: CuTeDSL traps
            # with "tensor memory not completely freed" if main() returns with a
            # live allocation.
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr, self.tmem_total_cols)
            return

        if tidx < self.tile_m:
            rs = sRowSum[tidx]
            if rs <= Float32(0.0):
                rs = Float32(1.0)
            row_global = q_row_base + tidx
            if row_global < q_start + q_len:
                for c in cutlass.range_constexpr(self.tile_hdim):
                    mOh[row_global, c] = (sOf[tidx, c] / rs).to(cutlass.Float16)
        cute.arch.barrier()

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr, self.tmem_total_cols)


@cute.jit
def _valid_tc(
    kv: Int32,
    row: Int32,
    kv_len: Int32,
    q_len: Int32,
    q_row_base: Int32,
    is_causal: cutlass.Constexpr[bool],
):
    ok = (kv < kv_len) & (row < q_len)
    if const_expr(is_causal):
        ok = ok & (kv <= (q_row_base + row))
    return ok
