"""Fused TurboQuant attention forward, CuTeDSL (SM100/SM110).

Single CTA per ``(q_block, q_head, request)``, cooperative threads:

    load packed K bytes -> SMEM
    dequant (bit-unpack + LUT gather) -> SMEM fp16 code tile
    QK MMA (cute.gemm, f16 tensor cores)
    S fragment -> SMEM
    row max / row sum + exp2 + fold v_norm -> SMEM P
    PV MMA into an fp32 register accumulator
    epilogue: divide by row sum, store to gmem

Two passes over the KV tiles
----------------------------
v0 uses the known-max two-pass formulation instead of a single online-softmax
pass:

* pass 1 computes only the row max (QK plus a row scan),
* pass 2 recomputes QK, applies ``exp2(s - max)``, folds ``v_norm`` into P, and
  accumulates the PV MMA.

Costs one extra K read and one extra QK per tile. Buys: the fp32 output
accumulator is never rescaled mid-loop, and no fragment coordinate arithmetic is
needed anywhere -- softmax and the epilogue round-trip through SMEM and are
indexed with plain integer math. At long-context decode the packed K read is a
small fraction of the V read, so the extra pass is cheap. The single-pass
correction-warp version is the documented follow-up.

Notes
-----
* Self-contained: only ``cutlass``/``cute`` primitives, so the cheap L4 compile
  probe can compile the entire kernel without flash-attention or quack.
* ``head_dim`` must be a multiple of 16 (64/128/256), so padded == true head dim.
* This is an ``mma.sync`` (SM80-style) schedule that is forward-compatible with
  ``sm_100a``; the tcgen05/TMEM schedule is the intended end state.
"""

from __future__ import annotations

import os

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp

from thunder_vllm.utils.logging import get_logger

logger = get_logger("attention.cute_kernel")

from thunder_vllm.attention.splits import choose_split_count  # noqa: E402  (re-export)

KERNEL_STATUS = "coop-two-pass-b200-verified"

VARIANT_BASELINE = 0
VARIANT_FUSED_K = 1
VARIANT_FUSED_KV = 2
VARIANT_K_REUSE = 3
VARIANT_MULTIHEAD_V = 4
VARIANT_SPLIT_K = 5
VARIANT_PACKING = 6

_LOG2E = 1.4426950408889634


class ThunderConfig(NamedTuple):
    """Compile-time configuration (the prototype's public shape)."""

    head_dim: int
    K_BITS: int
    V_BITS: int
    qhead_per_kvhead: int
    is_causal: bool
    m_block_size: int
    n_block_size: int
    num_stages: int
    num_dequant_stages: int
    num_threads: int


@cute.jit
def _unpack(packed: cute.Tensor, row: Int32, col: Int32, bits: cutlass.Constexpr[int]):
    """Read the ``bits``-wide little-endian index for ``(row, col)``."""
    if const_expr(bits == 4):
        byte = packed[row, col // 2]
        return (byte >> ((col % 2) * 4)) & 0x0F
    elif const_expr(bits == 2):
        byte = packed[row, col // 4]
        return (byte >> ((col % 4) * 2)) & 0x03
    elif const_expr(bits == 8):
        return packed[row, col]
    else:
        bit_pos = col * 3
        byte_col = bit_pos // 8
        shift = bit_pos % 8
        b0 = packed[row, byte_col]
        b1 = packed[row, byte_col + 1]
        word = b0.to(Int32) | (b1.to(Int32) << 8)
        return (word >> shift) & 0x07


@cute.jit
def _dequantize_strided(
    sPacked: cute.Tensor,
    sLut: cute.Tensor,
    sCode: cute.Tensor,
    bits: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    """Cooperative unpack + LUT gather: ``sCode[r, c] = sLut[idx, c]``."""
    total: cutlass.Constexpr[int] = rows * head_dim
    iters: cutlass.Constexpr[int] = (total + num_threads - 1) // num_threads
    for e in cutlass.range_constexpr(iters):
        idx = tidx + e * num_threads
        if idx < total:
            row = idx // head_dim
            col = idx % head_dim
            sCode[row, col] = sLut[_unpack(sPacked, row, col, bits), col]


@cute.jit
def _dequantize_transposed(
    sPacked: cute.Tensor,
    sLut: cute.Tensor,
    sCode: cute.Tensor,
    bits: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    """As :func:`_dequantize_strided`, writing ``sCode[c, r]``.

    The PV MMA wants V as ``(N, K) = (head_dim, tile_n)``, so V is dequantized
    straight into the transposed layout instead of transposing in SMEM.
    """
    total: cutlass.Constexpr[int] = rows * head_dim
    iters: cutlass.Constexpr[int] = (total + num_threads - 1) // num_threads
    for e in cutlass.range_constexpr(iters):
        idx = tidx + e * num_threads
        if idx < total:
            row = idx // head_dim
            col = idx % head_dim
            sCode[col, row] = sLut[_unpack(sPacked, row, col, bits), col]


@cute.jit
def _copy_lut_to_smem(
    mLut: cute.Tensor,
    sLut: cute.Tensor,
    tidx: Int32,
    bits: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
):
    n_lut: cutlass.Constexpr[int] = 1 << bits
    total: cutlass.Constexpr[int] = n_lut * head_dim
    iters: cutlass.Constexpr[int] = (total + num_threads - 1) // num_threads
    for e in cutlass.range_constexpr(iters):
        idx = tidx + e * num_threads
        if idx < total:
            sLut[idx // head_dim, idx % head_dim] = mLut[idx // head_dim, idx % head_dim]


class ThunderAttentionForward:
    """Cooperative TurboQuant forward.

    ``num_threads`` is a multiple of 32 and ``m_block_size == tile_m`` must be
    ``(num_threads // 32) * 16`` so the QK MMA atom layout tiles ``M`` exactly.
    """

    def __init__(
        self,
        head_dim: int,
        K_BITS: cutlass.Constexpr[int],
        V_BITS: cutlass.Constexpr[int],
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        m_block_size: int = 64,
        n_block_size: int = 64,
        num_stages: int = 2,
        num_dequant_stages: int = 2,
        num_threads: int = 128,
        use_2cta_instrs: bool = False,
        q_stage: cutlass.Constexpr[int] = 2,
        variant_tag: cutlass.Constexpr[int] = VARIANT_FUSED_KV,
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
        if self.tile_m != self.num_warps * 16:
            raise ValueError(
                f"m_block_size ({self.tile_m}) must equal num_threads//32*16 "
                f"({self.num_warps * 16}) for the QK MMA atom layout"
            )
        if use_2cta_instrs:
            raise NotImplementedError("2-CTA MMAs are not wired in the coop schedule")

    def _get_tiled_mma(self):
        op = warp.MmaF16BF16Op(cutlass.Float16, Float32, (16, 8, 16))
        tiled_mma_qk = cute.make_tiled_mma(
            op,
            (self.num_warps, 1, 1),
            permutation_mnk=(self.num_warps * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            op,
            (self.num_warps, 1, 1),
            permutation_mnk=(self.num_warps * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def _smem_layouts(self):
        """Row-major SMEM layouts, built inside the current CuTeDSL region."""
        f16 = cute.make_layout
        return {
            "sQ": f16((self.tile_m, self.tile_hdim), stride=(self.tile_hdim, 1)),
            "sK_packed": f16((self.tile_n, self.k_packed_bytes)),
            "sV_packed": f16((self.tile_n, self.v_packed_bytes)),
            "sK_code": f16((self.tile_n, self.tile_hdim), stride=(self.tile_hdim, 1)),
            "sV_code": f16((self.tile_hdim, self.tile_n)),
            "sP": f16((self.tile_m, self.tile_n)),
            "sS": f16((self.tile_m, self.tile_n)),
            "sOf": f16((self.tile_m, self.tile_hdim)),
            "sKLut": f16(((1 << self.K_BITS), self.tile_hdim)),
            "sVLut": f16(((1 << self.V_BITS), self.tile_hdim)),
            "sKNorm": f16((self.tile_n,)),
            "sVNorm": f16((self.tile_n,)),
            "sRowMax": f16((self.tile_m,)),
            "sRowSum": f16((self.tile_m,)),
            "sAlpha": f16((self.tile_m,)),
        }

    @cute.jit
    def _get_shared_storage_cls(self):
        dt = cutlass.Float16
        u8 = cutlass.Uint8
        f32 = Float32
        tm, tn, hd = self.tile_m, self.tile_n, self.tile_hdim

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[dt, tm * hd], 1024]
            sK_code: cute.struct.Align[cute.struct.MemRange[dt, tn * hd], 1024]
            sV_code: cute.struct.Align[cute.struct.MemRange[dt, hd * tn], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[dt, tm * tn], 1024]
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
            sAlpha: cute.struct.Align[cute.struct.MemRange[f32, tm], 128]

        return SharedStorage

    # ------------------------------------------------------------------
    # Host entry
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, hdim) fp16
        mK: cute.Tensor,  # (total_kv, Hk, k_packed_bytes) uint8
        mV: cute.Tensor,  # (total_kv, Hk, v_packed_bytes) uint8
        mKN: cute.Tensor,  # (total_kv, Hk) fp16
        mVN: cute.Tensor,  # (total_kv, Hk) fp16
        mKLut: cute.Tensor,  # (2**K_BITS, hdim) fp16
        mVLut: cute.Tensor,  # (2**V_BITS, hdim) fp16
        mO: cute.Tensor,  # (total_q, Hq, hdim) fp16
        mSeqLens: cute.Tensor,  # (num_reqs,) int32
        mQStart: cute.Tensor,  # (num_reqs + 1,) int32
        mDbg: cute.Tensor,  # (>= 8 + num_reqs,) int32
        mPartO: cute.Tensor,  # (num_reqs*num_splits, Hq, hdim) fp32, split mode
        mPartM: cute.Tensor,  # (num_reqs*num_splits, Hq) fp32, split mode
        mPartL: cute.Tensor,  # (num_reqs*num_splits, Hq) fp32, split mode
        softmax_scale: Float32,
        kv_row_stride: int = 1,
        debug: cutlass.Constexpr[int] = 0,
        max_query_len: int = 0,
        num_splits: int = 1,
        split_mode: cutlass.Constexpr[int] = 0,
        gqa_pack: cutlass.Constexpr[int] = 0,
        onepass: cutlass.Constexpr[int] = 0,
        reg_rescale: cutlass.Constexpr[int] = 0,
        causal_bound: cutlass.Constexpr[int] = 0,
        stream=None,
    ):
        # (rows, Hq, hdim) -> (rows, hdim, Hq) so a head slice is rank 2.
        q_perm = [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=q_perm))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=q_perm))
        kv_perm = [0, 2, 1]
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=kv_perm))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=kv_perm))

        num_reqs = mSeqLens.shape[0]
        num_q_heads = mQ.shape[2]
        # ``q_block`` is a per-request index (the kernel reads
        # ``mQh[q_start + q_block*tile_m + row]``), so the grid must be sized by
        # the LONGEST request, not by the total token count. Sizing it globally
        # (total_q / tile_m) makes every request's block index run up to the
        # global block count and the Q load walks past ``mQ``: with vLLM's
        # warm-up (1024 requests x 16 q tokens) the read reached row 32703 of a
        # 16384-row tensor -> CUDA_ERROR_ILLEGAL_ADDRESS. It stayed hidden in
        # every single-request probe because q_start == 0, and in the multi-req
        # probe because total_q <= tile_m so there was only ever block 0.
        # ``max_query_len`` is resolved on the host (a runtime branch cannot set
        # a variable visible after the staged ``if`` in CuTeDSL).
        num_q_blocks = cute.ceil_div(max_query_len, self.tile_m)

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls()

        self.kernel(
            mQ,
            mK,
            mV,
            mKN,
            mVN,
            mKLut,
            mVLut,
            mO,
            mSeqLens,
            mQStart,
            mDbg,
            mPartO,
            mPartM,
            mPartL,
            softmax_scale,
            Int32(kv_row_stride),
            Int32(num_splits),
            debug,
            split_mode,
            gqa_pack,
            onepass,
            reg_rescale,
            causal_bound,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            stream,
        ).launch(
            # Split-K: the request axis is multiplied by num_splits; block idx z
            # maps z -> (req = z // S, split = z % S). With S == 1 this is exactly
            # the baseline grid.
            #
            # gqa_pack puts KV heads, not query heads, on the y axis: one CTA
            # reconstructs a KV tile once and scores every query head sharing it.
            grid=(
                num_q_blocks,
                mK.shape[2] if gqa_pack else num_q_heads,
                num_reqs * num_splits,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Kernel body
    # ------------------------------------------------------------------
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
        mDbg: cute.Tensor,
        mPartO: cute.Tensor,
        mPartM: cute.Tensor,
        mPartL: cute.Tensor,
        softmax_scale: Float32,
        kv_row_stride: Int32,
        num_splits: Int32,
        debug: cutlass.Constexpr[int],
        split_mode: cutlass.Constexpr[int],
        gqa_pack: cutlass.Constexpr[int],
        onepass: cutlass.Constexpr[int],
        reg_rescale: cutlass.Constexpr[int],
        causal_bound: cutlass.Constexpr[int],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        stream=None,
    ):
        tidx = cute.arch.thread_idx()[0]
        q_block = cute.arch.block_idx()[0]
        head_or_kv = cute.arch.block_idx()[1]
        z = cute.arch.block_idx()[2]
        # z -> (request, split). num_splits == 1 leaves this identical to the
        # baseline (req = z, split = 0).
        req = z // num_splits
        split = z % num_splits
        if const_expr(gqa_pack):
            # y axis is the KV head; the query heads sharing it are the M rows.
            kv_head = head_or_kv
            q_head = head_or_kv * self.qhead_per_kvhead
        else:
            q_head = head_or_kv
            kv_head = q_head // self.qhead_per_kvhead
        # Block-unique debug slot, defined before any control-flow region: every
        # write below is keyed by (z, y) so the snapshot is coherent.
        n_q_heads: cutlass.Constexpr[int] = mQ.shape[2]
        dbg_slot = (req * n_q_heads + q_head) * 16
        # Request-major row base for the gathered KV buffers. Published to mDbg so
        # the multi-request failing case can be diagnosed from the host instead of
        # inferred (the debug tensor is a required argument; the launcher passes a
        # small persistent dummy when nobody is watching).
        req_base = req * kv_row_stride
        if tidx == 0:
            if const_expr(debug):
                mDbg[dbg_slot + 0] = req
                mDbg[dbg_slot + 1] = q_head
                mDbg[dbg_slot + 2] = kv_head
                mDbg[dbg_slot + 3] = kv_row_stride
                mDbg[dbg_slot + 4] = req_base
        # Physical offset each tensor maps the same logical row to. K and V should
        # agree; if they do not, the fault is a layout mapping (a permutation, not
        # a proportional stride error -- which is the R16 +4-blocks signature).
        if const_expr(debug):
            _mKh = mK[None, None, kv_head]
            _mVh = mV[None, None, kv_head]
            if tidx == 0:
                mDbg[dbg_slot + 5] = Int32(cute.crd2idx((req_base, 0), _mKh.layout))
                mDbg[dbg_slot + 6] = Int32(cute.crd2idx((req_base, 0), _mVh.layout))


        q_start = mQStart[req]
        q_len = mQStart[req + 1] - q_start
        kv_len = mSeqLens[req]

        layouts = self._smem_layouts()
        smem_alloc = cutlass.utils.SmemAllocator()
        storage = smem_alloc.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(layouts["sQ"])
        sK_packed = storage.sK_packed.get_tensor(layouts["sK_packed"])
        sV_packed = storage.sV_packed.get_tensor(layouts["sV_packed"])
        sK_code = storage.sK_code.get_tensor(layouts["sK_code"])
        sV_code = storage.sV_code.get_tensor(layouts["sV_code"])
        sP = storage.sP.get_tensor(layouts["sP"])
        sS = storage.sS.get_tensor(layouts["sS"])
        sOf = storage.sOf.get_tensor(layouts["sOf"])
        sKLut = storage.sKLut.get_tensor(layouts["sKLut"])
        sVLut = storage.sVLut.get_tensor(layouts["sVLut"])
        sKNorm = storage.sKNorm.get_tensor(layouts["sKNorm"])
        sVNorm = storage.sVNorm.get_tensor(layouts["sVNorm"])
        sRowMax = storage.sRowMax.get_tensor(layouts["sRowMax"])
        sRowSum = storage.sRowSum.get_tensor(layouts["sRowSum"])
        sAlpha = storage.sAlpha.get_tensor(layouts["sAlpha"])

        # ---- LUT staging (request independent) -------------------------
        _copy_lut_to_smem(mKLut, sKLut, tidx, self.K_BITS, self.tile_hdim, self.num_threads)
        _copy_lut_to_smem(mVLut, sVLut, tidx, self.V_BITS, self.tile_hdim, self.num_threads)

        # ---- Q tile ----------------------------------------------------
        mQh = mQ[None, None, q_head]
        mOh = mO[None, None, q_head]
        q_total: cutlass.Constexpr[int] = self.tile_m * self.tile_hdim
        q_iters: cutlass.Constexpr[int] = (q_total + self.num_threads - 1) // self.num_threads
        if const_expr(gqa_pack):
            # Decode: one query token, qhead_per_kvhead heads sharing this KV
            # head. M row r is (token = q_start + q_block*tile_m, head =
            # kv_head*G + r); rows >= G are padding.
            token_q = q_start + q_block * self.tile_m
            for e in cutlass.range_constexpr(q_iters):
                idx = tidx + e * self.num_threads
                if idx < q_total:
                    row = idx // self.tile_hdim
                    col = idx % self.tile_hdim
                    if row < self.qhead_per_kvhead:
                        sQ[row, col] = mQ[token_q, col, kv_head * self.qhead_per_kvhead + row]
                    else:
                        sQ[row, col] = cutlass.Float16(0.0)
        else:
            for e in cutlass.range_constexpr(q_iters):
                idx = tidx + e * self.num_threads
                if idx < q_total:
                    row = idx // self.tile_hdim
                    col = idx % self.tile_hdim
                    # ``row < q_len`` alone is NOT a bounds check: the address adds
                    # q_block*tile_m, so a block past this request's query length must
                    # be skipped entirely (and zeroed) or it reads past mQ.
                    if q_block * self.tile_m + row < q_len:
                        sQ[row, col] = mQh[q_start + q_block * self.tile_m + row, col]
                    else:
                        sQ[row, col] = cutlass.Float16(0.0)

        if tidx < self.tile_m:
            sRowMax[tidx] = -Float32.inf
            sRowSum[tidx] = Float32(0.0)
        cute.arch.barrier()

        # ---- MMA fragments and SMEM copies -----------------------------
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        acc_S = cute.make_rmem_tensor(
            thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_O = cute.make_rmem_tensor(
            thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim)), Float32
        )
        acc_O.fill(0.0)
        # Per-element M row of the accumulator fragment, for the one-pass online
        # rescale (multiply each element by alpha[row]).
        cO = thr_mma_pv.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        )

        # Plain 32-bit universal SMEM->register copies. ldmatrix would be the
        # performance choice but requires a swizzle-compatible SMEM layout and
        # 128-bit-aligned sources; a universal copy has a 32-bit alignment
        # requirement and works directly on the row-major tiles, which keeps the
        # v0 schedule self-contained and easy to verify. Swapping in ldmatrix
        # (with the swizzled layouts from sm80_helpers) is a pure win later.
        ld_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=16
        )
        thr_copy_q = cute.make_tiled_copy_A(ld_atom, tiled_mma_qk).get_slice(tidx)
        thr_copy_k = cute.make_tiled_copy_B(ld_atom, tiled_mma_qk).get_slice(tidx)
        thr_copy_p = cute.make_tiled_copy_A(ld_atom, tiled_mma_pv).get_slice(tidx)
        thr_copy_v = cute.make_tiled_copy_B(ld_atom, tiled_mma_pv).get_slice(tidx)

        rQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        rK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK_code))
        rP = thr_mma_pv.make_fragment_A(thr_mma_pv.partition_A(sP))
        rV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV_code))

        smem_copy_atom_S = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        smem_thr_copy_S = cute.make_tiled_copy_C(smem_copy_atom_S, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
        taccSrS = smem_thr_copy_S.retile(acc_S)
        taccOrO = smem_thr_copy_O.retile(acc_O)
        tSsS = smem_thr_copy_S.partition_D(sS)
        tOsOf = smem_thr_copy_O.partition_D(sOf)

        # Load Q once, upstream of both passes.
        cute.copy(thr_copy_q, thr_copy_q.partition_S(sQ), thr_copy_q.retile(rQ))
        cute.arch.barrier()

        n_tiles = cute.ceil_div(kv_len, self.tile_n)
        q_off = q_block * self.tile_m
        # Query validity / causal use these. gqa_pack: every M row is the SAME
        # query token at a different head, so the row validity test is "row < G"
        # and causal must not advance with the row (decode is not causal anyway;
        # gqa_pack is only enabled for decode).
        q_off_v = q_off
        q_len_v = q_len
        if const_expr(gqa_pack):
            q_off_v = Int32(0)
            q_len_v = Int32(self.qhead_per_kvhead)
        q_row_base = q_start + q_off_v

        # Causal bound: a q-block only attends KV at or below its highest query
        # position, so tiles entirely past that are fully masked. Skipping them
        # removes real work (~half the tile iterations for causal prefill).
        n_eff = n_tiles
        if const_expr(causal_bound):
            if self.is_causal:
                max_row = q_len - q_off_v - 1
                if max_row > self.tile_m - 1:
                    max_row = self.tile_m - 1
                max_kv = (kv_len - q_len) + q_off_v + max_row
                n_eff = cute.ceil_div(max_kv + 1, self.tile_n)
                if n_eff > n_tiles:
                    n_eff = n_tiles
                if n_eff < 0:
                    n_eff = Int32(0)

        # This CTA's contiguous tile range within the request's KV. num_splits == 1
        # covers every tile, so the baseline schedule is unchanged.
        tiles_per_split = cute.ceil_div(n_eff, num_splits)
        nt_lo = split * tiles_per_split
        nt_hi = nt_lo + tiles_per_split
        if nt_hi > n_eff:
            nt_hi = n_eff
        n_local = nt_hi - nt_lo
        # One-pass: PASS 1 is skipped and PASS 2 performs the online-softmax
        # update itself. The loop bound (not an `if`) keeps both branches in the
        # same staged scope.
        n_p1 = n_local
        if const_expr(onepass):
            n_p1 = Int32(0)

        # ============================ PASS 1: row max ====================
        for i in cutlass.range(n_p1, unroll=1):
            nt = nt_lo + i
            _load_kv_packed(mK, mV, mKN, mVN, sK_packed, sV_packed, sKNorm, sVNorm,
                            kv_head, req_base, nt, kv_len, tidx, self,
                            want_v=False)
            if const_expr(debug):
                _record_kv_probe(mDbg, dbg_slot, req_base, nt, kv_len, tidx, self.tile_n)
            cute.arch.barrier()
            _dequantize_strided(sK_packed, sKLut, sK_code, self.K_BITS,
                                self.tile_hdim, self.tile_n, tidx, self.num_threads)
            cute.arch.barrier()
            # rK must be re-read from SMEM every tile: the code buffer is reused.
            cute.copy(thr_copy_k, thr_copy_k.partition_S(sK_code), thr_copy_k.retile(rK))
            acc_S.fill(0.0)
            cute.gemm(tiled_mma_qk, acc_S, rQ, rK, acc_S)
            cute.copy(smem_copy_atom_S, taccSrS, tSsS)
            cute.arch.barrier()
            if tidx < self.tile_m:
                cur = -Float32.inf
                for n in cutlass.range_constexpr(self.tile_n):
                    kv = nt * self.tile_n + n
                    val = sS[tidx, n] * Float32(sKNorm[n]) * softmax_scale
                    if _valid(kv, tidx, kv_len, q_off_v, q_len_v, self.is_causal):
                        cur = cute.arch.fmax(cur, val)
                sRowMax[tidx] = cute.arch.fmax(sRowMax[tidx], cur)
            cute.arch.barrier()

        # ============================ PASS 2: softmax + PV ===============
        for i in cutlass.range(n_local, unroll=1):
            nt = nt_lo + i
            _load_kv_packed(mK, mV, mKN, mVN, sK_packed, sV_packed, sKNorm, sVNorm,
                            kv_head, req_base, nt, kv_len, tidx, self,
                            want_v=True)
            if const_expr(debug):
                _record_kv_probe(mDbg, dbg_slot, req_base, nt, kv_len, tidx, self.tile_n)
            cute.arch.barrier()
            if const_expr(debug):
                if tidx == 0:
                    # P0: the raw packed byte staged for (kv row 0, byte 0) of THIS
                    # (request, head) block. Sentinel bytes are const_byte(c), so the
                    # low nibble is the code.
                    mDbg[dbg_slot + 11] = Int32(sV_packed[0, 0])
            _dequantize_strided(sK_packed, sKLut, sK_code, self.K_BITS,
                                self.tile_hdim, self.tile_n, tidx, self.num_threads)
            _dequantize_transposed(sV_packed, sVLut, sV_code, self.V_BITS,
                                   self.tile_hdim, self.tile_n, tidx, self.num_threads)
            if const_expr(debug):
                if tidx == 0:
                    # The value the PV will actually multiply: sV_code[hdim 0, kv 0].
                    # Small integer sentinels, so the fp16 -> int truncation is exact.
                    mDbg[dbg_slot + 12] = Int32(sV_code[0, 0])
            cute.arch.barrier()
            cute.copy(thr_copy_k, thr_copy_k.partition_S(sK_code), thr_copy_k.retile(rK))
            acc_S.fill(0.0)
            cute.gemm(tiled_mma_qk, acc_S, rQ, rK, acc_S)
            cute.copy(smem_copy_atom_S, taccSrS, tSsS)
            cute.arch.barrier()
            if const_expr(onepass):
                # Online-softmax update: this tile is the only time K is
                # reconstructed and QK is run. Fold the tile max into the running
                # max and record the rescale factor for the accumulator and l.
                if tidx < self.tile_m:
                    cur = -Float32.inf
                    for n in cutlass.range_constexpr(self.tile_n):
                        kv = nt * self.tile_n + n
                        if _valid(kv, tidx, kv_len, q_off_v, q_len_v, self.is_causal):
                            cur = cute.arch.fmax(
                                cur, sS[tidx, n] * Float32(sKNorm[n]) * softmax_scale)
                    new_m = cute.arch.fmax(sRowMax[tidx], cur)
                    alpha = Float32(1.0)
                    if new_m != -Float32.inf:
                        alpha = cute.math.exp2(
                            (sRowMax[tidx] - new_m) * _LOG2E, fastmath=True)
                    sAlpha[tidx] = alpha
                    sRowMax[tidx] = new_m
                cute.arch.barrier()
            if tidx < self.tile_m:
                acc = Float32(0.0)
                for n in cutlass.range_constexpr(self.tile_n):
                    kv = nt * self.tile_n + n
                    p = Float32(0.0)
                    if _valid(kv, tidx, kv_len, q_off_v, q_len_v, self.is_causal):
                        x = sS[tidx, n] * Float32(sKNorm[n]) * softmax_scale
                        p = cute.math.exp2((x - sRowMax[tidx]) * _LOG2E, fastmath=True)
                    sP[tidx, n] = (p * Float32(sVNorm[n])).to(cutlass.Float16)
                    acc += p
                if const_expr(onepass):
                    sRowSum[tidx] = sRowSum[tidx] * sAlpha[tidx] + acc
                else:
                    sRowSum[tidx] = sRowSum[tidx] + acc
            cute.arch.barrier()
            if const_expr(onepass):
                if const_expr(reg_rescale):
                    # Fragment-local rescale: no SMEM round-trip, no barrier per
                    # copy. cO carries each accumulator element's M row.
                    cute.arch.barrier()
                    for e in cutlass.range_constexpr(cute.size(acc_O)):
                        acc_O[e] = acc_O[e] * sAlpha[cute.get(cO[e], 0)]
                    cute.arch.barrier()
                else:
                    # Rescale the running PV accumulator by alpha (per row) before
                    # adding this tile. Done through the existing fp32 sOf staging
                    # buffer: no fragment-coordinate arithmetic.
                    cute.copy(smem_copy_atom_O, taccOrO, tOsOf)
                    cute.arch.barrier()
                    if tidx < self.tile_m:
                        a = sAlpha[tidx]
                        for c in cutlass.range_constexpr(self.tile_hdim):
                            sOf[tidx, c] = sOf[tidx, c] * a
                    cute.arch.barrier()
                    cute.copy(smem_copy_atom_O, tOsOf, taccOrO)
                    cute.arch.barrier()
            # P and V are produced in this iteration; re-read them for the PV MMA.
            cute.copy(thr_copy_v, thr_copy_v.partition_S(sV_code), thr_copy_v.retile(rV))
            cute.copy(thr_copy_p, thr_copy_p.partition_S(sP), thr_copy_p.retile(rP))
            cute.arch.barrier()
            cute.gemm(tiled_mma_pv, acc_O, rP, rV, acc_O)
            cute.arch.barrier()

        # ============================ Epilogue ==========================
        cute.copy(smem_copy_atom_O, taccOrO, tOsOf)
        cute.arch.barrier()
        if const_expr(gqa_pack):
            # M row r is the query head kv_head*G + r; all rows share the same
            # query token (q_row_base). Only G rows are live.
            if tidx < self.qhead_per_kvhead:
                head_t = kv_head * self.qhead_per_kvhead + tidx
                if const_expr(split_mode):
                    for c in cutlass.range_constexpr(self.tile_hdim):
                        mPartO[z, head_t, c] = sOf[tidx, c]
                    mPartM[z, head_t] = sRowMax[tidx]
                    mPartL[z, head_t] = sRowSum[tidx]
                else:
                    rs = sRowSum[tidx]
                    if rs <= Float32(0.0):
                        rs = Float32(1.0)
                    mOh_t = mO[None, None, head_t]
                    if q_row_base < q_start + q_len:
                        for c in cutlass.range_constexpr(self.tile_hdim):
                            mOh_t[q_row_base, c] = (sOf[tidx, c] / rs).to(cutlass.Float16)
        elif const_expr(split_mode):
            # Split-K: publish the UNNORMALISED partial state for this
            # (request, head, split) so the host merge can rescale by the global
            # max. Decode has q_len == 1, so only row 0 is a live query row.
            if tidx < self.tile_m:
                if tidx == 0:
                    for c in cutlass.range_constexpr(self.tile_hdim):
                        mPartO[z, q_head, c] = sOf[0, c]
                    mPartM[z, q_head] = sRowMax[0]
                    mPartL[z, q_head] = sRowSum[0]
        else:
            if tidx < self.tile_m:
                rs = sRowSum[tidx]
                if rs <= Float32(0.0):
                    rs = Float32(1.0)
                row_global = q_row_base + tidx
                if row_global < q_start + q_len:
                    for c in cutlass.range_constexpr(self.tile_hdim):
                        mOh[row_global, c] = (sOf[tidx, c] / rs).to(cutlass.Float16)
        cute.arch.barrier()


@cute.jit
def _valid(
    kv: Int32,
    row: Int32,
    kv_len: Int32,
    q_off: Int32,
    q_len: Int32,
    is_causal: cutlass.Constexpr[bool],
):
    """Whether ``(row, kv)`` is a live attention entry for this request.

    ``q_off`` is this q-block's offset inside the request's query range, so the
    query-validity test is ``q_off + row < q_len`` (not ``row < q_len``): the row
    index is local to the tile, the query index is not.

    Causal masking must compare against the query's position WITHIN THE REQUEST
    (``kv_len - q_len + q_off + row``), not the global flat token index
    ``query_start_loc``. vLLM appends the query tokens to the request's existing
    context, so for decode (q_len == 1, q_off == 0) every context token is
    visible; using the global index instead masked it down to token 0 (measured:
    a 512-token decode returned only block 0's value).
    """
    ok = (kv < kv_len) & ((q_off + row) < q_len)
    if const_expr(is_causal):
        ok = ok & (kv <= (kv_len - q_len) + q_off + row)
    return ok


@cute.jit
def _load_kv_packed(
    mK: cute.Tensor,
    mV: cute.Tensor,
    mKN: cute.Tensor,
    mVN: cute.Tensor,
    sK_packed: cute.Tensor,
    sV_packed: cute.Tensor,
    sKNorm: cute.Tensor,
    sVNorm: cute.Tensor,
    kv_head: Int32,
    req_base: Int32,
    nt: Int32,
    kv_len: Int32,
    tidx: Int32,
    self: cutlass.Constexpr,
    want_v: cutlass.Constexpr[bool] = True,
):
    """Cooperative load of one packed K/V tile plus its norms.

    Rows past ``kv_len`` are left untouched: the softmax mask zeroes them, and
    skipping the global load keeps the last partial tile in bounds.

    The gathered buffers are REQUEST-MAJOR: row = req * kv_row_stride + block *
    block_size + offset. ``base`` below is the per-request row base; without it
    every request read request 0's rows (verified: with R=2 the second request
    returned the first request's V sentinel, and with larger R the output was
    exactly the mean of request 0's block range).
    """
    base = req_base
    mKh = mK[None, None, kv_head]
    mVh = mV[None, None, kv_head]
    k_tot: cutlass.Constexpr[int] = self.tile_n * self.k_packed_bytes
    k_iters: cutlass.Constexpr[int] = (k_tot + self.num_threads - 1) // self.num_threads
    for e in cutlass.range_constexpr(k_iters):
        i = tidx + e * self.num_threads
        if i < k_tot:
            row = i // self.k_packed_bytes
            if nt * self.tile_n + row < kv_len:
                sK_packed[row, i % self.k_packed_bytes] = mKh[
                    base + nt * self.tile_n + row, i % self.k_packed_bytes
                ]
            else:
                # Zero rows past kv_len instead of leaving stale SMEM. They are
                # masked out of the softmax, but the PV GEMM multiplies p (=0)
                # against the dequantized V, and 0 * NaN = NaN -- which is how a
                # stale tail tile turned the output NaN.
                sK_packed[row, i % self.k_packed_bytes] = cutlass.Uint8(0)
    v_tot: cutlass.Constexpr[int] = self.tile_n * self.v_packed_bytes
    v_iters: cutlass.Constexpr[int] = (v_tot + self.num_threads - 1) // self.num_threads
    if const_expr(want_v):
        for e in cutlass.range_constexpr(v_iters):
            i = tidx + e * self.num_threads
            if i < v_tot:
                row = i // self.v_packed_bytes
                if nt * self.tile_n + row < kv_len:
                    sV_packed[row, i % self.v_packed_bytes] = mVh[
                        base + nt * self.tile_n + row, i % self.v_packed_bytes
                    ]
                else:
                    sV_packed[row, i % self.v_packed_bytes] = cutlass.Uint8(0)
    if tidx < self.tile_n:
        if nt * self.tile_n + tidx < kv_len:
            sKNorm[tidx] = mKN[base + nt * self.tile_n + tidx, kv_head]
            if const_expr(want_v):
                sVNorm[tidx] = mVN[base + nt * self.tile_n + tidx, kv_head]
        else:
            sKNorm[tidx] = cutlass.Float16(0.0)
            if const_expr(want_v):
                sVNorm[tidx] = cutlass.Float16(0.0)


@cute.jit
def _record_kv_probe(
    mDbg: cute.Tensor,
    dbg_slot: Int32,
    req_base: Int32,
    nt: Int32,
    kv_len: Int32,
    tidx: Int32,
    tile_n: cutlass.Constexpr[int],
):
    """Record the load-time inputs used by this thread block at one slot.

    Debug writes stay out of the memory path that tcgen05 also imports.
    """
    if tidx == 0:
        mDbg[dbg_slot + 7] = req_base
        mDbg[dbg_slot + 8] = nt
        mDbg[dbg_slot + 9] = kv_len
        mDbg[dbg_slot + 10] = req_base + nt * tile_n


class KernelNotReadyError(RuntimeError):
    """Kept for API compatibility; the coop schedule is launchable."""


_DBG_BUFFER = None
_SPLIT_BUFFERS: dict = {}
_FAST: dict = {}
_FASTLAUNCH = os.environ.get("THUNDER_FASTLAUNCH", "0").strip().lower() not in (
    "", "0", "false", "no", "off"
)


def _jitcache_keys(cache) -> set:
    """Keys of a CuTeDSL ``JitCacheDict`` (its backing dict is ``_dict``)."""
    d = getattr(cache, "_dict", None)
    return set(d.keys()) if isinstance(d, dict) else set()


def _dsl_object(kernel):
    """The CuTeDSL object behind ``kernel``'s ``@cute.jit __call__``."""
    wrapper = getattr(type(kernel), "__call__", None)
    orig = getattr(wrapper, "__wrapped__", None)
    if orig is None:
        orig = getattr(getattr(kernel, "__call__", None), "__wrapped__", None)
    return getattr(orig, "_dsl_object", None)

# Host-stage timing for THUNDER_TIME_LAUNCH, dumped at exit. Separates the torch
# plumbing (reshape/contiguous/from_dlpack) from the CuTeDSL host call and the
# split-K merge so a slow launch can be attributed.
_LAUNCH_TIME: dict = {"plumbing": 0.0, "kernel": 0.0, "merge": 0.0, "n": 0}


def _dump_launch_time() -> None:
    import os

    if os.environ.get("THUNDER_TIME_LAUNCH", "0").strip().lower() in (
        "", "0", "false", "no", "off"
    ):
        return
    n = max(_LAUNCH_TIME["n"], 1)
    print(
        "[TQ-LAUNCH] n=%d  plumbing=%.2fms  kernel=%.2fms  merge=%.2fms  "
        "total=%.2fms"
        % (
            _LAUNCH_TIME["n"],
            _LAUNCH_TIME["plumbing"] / n * 1e3,
            _LAUNCH_TIME["kernel"] / n * 1e3,
            _LAUNCH_TIME["merge"] / n * 1e3,
            sum(_LAUNCH_TIME[k] for k in ("plumbing", "kernel", "merge")) / n * 1e3,
        ),
        flush=True,
    )


import atexit as _atexit

_atexit.register(_dump_launch_time)



def _split_buffers(num_reqs: int, num_splits: int, hq: int, hd: int, device, dtype):
    """Persistent split-K partial buffers (pointer-stable for CUDA graphs).

    ``part_o`` holds the UNNORMALISED PV accumulator per (req, head, split);
    ``part_m`` / ``part_l`` the per-split row max and exp-sum. Decode has one
    live query row, so only row 0 is published.
    """
    import torch

    key = (int(num_reqs), int(num_splits), int(hq), int(hd), str(device))
    bufs = _SPLIT_BUFFERS.get(key)
    if bufs is None:
        rows = int(num_reqs) * int(num_splits)
        bufs = (
            torch.empty((rows, hq, hd), dtype=torch.float32, device=device),
            torch.empty((rows, hq), dtype=torch.float32, device=device),
            torch.empty((rows, hq), dtype=torch.float32, device=device),
        )
        _SPLIT_BUFFERS[key] = bufs
    return bufs


def _merge_splits(part_o, part_m, part_l, num_reqs, num_splits, hq, hd, o3, q_start):
    """Rescale and reduce split-K partials, then scatter into the output rows.

    Same online-softmax algebra the kernel uses across tiles: with global
    ``M = max_s m_s``, each split contributes ``exp(m_s - M)``.
    """
    import torch

    S = int(num_splits)
    po = part_o[: num_reqs * S].view(num_reqs, S, hq, hd)
    pm = part_m[: num_reqs * S].view(num_reqs, S, hq)
    pl = part_l[: num_reqs * S].view(num_reqs, S, hq)
    M = pm.amax(dim=1)  # (R, Hq)
    M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
    w = torch.exp(pm - M[:, None, :])  # (R, S, Hq)
    acc = (po * w[..., None]).sum(dim=1)  # (R, Hq, Hd)
    den = (pl * w).sum(dim=1)  # (R, Hq)
    res = acc / den.clamp_min(1e-20)[..., None]
    # Decode: exactly one query row per request, so q_start indexes the output row.
    o3.index_copy_(0, q_start[:num_reqs].to(torch.int64), res.to(o3.dtype))


def launch_thunder_attention(
    kernel: ThunderAttentionForward,
    q,
    gathered,
    out,
    metadata,
    softmax_scale: float,
    *,
    quantizer,
    debug: bool = False,
    num_splits: int = 1,
    gqa_pack: bool = False,
    onepass: bool = False,
    reg_rescale: bool = False,
    causal_bound: bool = False,
) -> None:
    """Launch the compiled cooperative kernel on gathered, contiguous tensors.

    Converts the torch tensors to CuTe tensors (``from_dlpack``) and passes them
    straight to the ``@cute.jit`` entry point. On the first call this triggers
    compilation and caches it (keyed by the constexpr arguments); subsequent
    calls are pure launches, which is what a CUDA graph records.

    ``num_splits > 1`` selects split-K decode: the request axis is multiplied by
    the split count, each CTA covers a contiguous slice of the KV tiles and
    publishes unnormalised partial state, and ``_merge_splits`` reduces them.
    Only valid for single-query decode (``max_query_len == 1``).
    """
    import torch
    from cutlass.cute.runtime import from_dlpack

    if quantizer is None:
        raise ValueError("quantizer is required to source the K/V LUTs")

    # Diagnostic only: skip the CuTe launch so a capture-path CUDA fault can be
    # attributed to this kernel vs the surrounding torch ops. Set
    # THUNDER_SKIP_KERNEL=1; the output is left untouched.
    import os as _os

    if _os.environ.get("THUNDER_SKIP_KERNEL", "0").strip().lower() not in (
        "", "0", "false", "no", "off"
    ):
        return

    _TIME = _os.environ.get("THUNDER_TIME_LAUNCH", "0").strip().lower() not in (
        "", "0", "false", "no", "off"
    )
    if _TIME:
        import time as _time

        _t0 = _time.perf_counter()

    hk = int(gathered.k_packed.shape[2])
    hq = int(q.shape[1])
    hd = int(q.shape[2])
    page_rows, bs = gathered.k_packed.shape[0], gathered.k_packed.shape[1]
    total_kv = page_rows * bs

    k = gathered.k_packed.reshape(total_kv, hk, kernel.k_packed_bytes).contiguous()
    v = gathered.v_packed.reshape(total_kv, hk, kernel.v_packed_bytes).contiguous()
    kn = gathered.k_norm.reshape(total_kv, hk).contiguous()
    vn = gathered.v_norm.reshape(total_kv, hk).contiguous()
    k_lut = quantizer.k_lut.contiguous()
    v_lut = quantizer.v_lut.contiguous()

    seq_lens = metadata.seq_lens.to(torch.int32).contiguous()
    q_start = metadata.query_start_loc.to(torch.int32).contiguous()
    o3 = out.reshape(q.shape[0], q.shape[1], q.shape[2]).contiguous()

    _torch_args = [q.contiguous(), k, v, kn, vn, k_lut, v_lut, o3, seq_lens, q_start]
    args = [from_dlpack(t) for t in _torch_args]
    stream = torch.cuda.current_stream().cuda_stream
    import cuda.bindings.driver as cuda


    # Request-major row base. The gather emits row = req * max_blocks_per_req +
    # block, so the kernel needs that stride to keep request identity in the
    # address. metadata carries it; deriving from the buffer would be wrong
    # whenever the gather padded up to max_num_reqs.
    num_reqs = int(seq_lens.shape[0])
    # The kernel's row index is in TOKENS (the launcher reshapes the gathered
    # buffer to (page_rows * block_size, Hk, pb)), while the gather emits rows in
    # BLOCKS. The request-major base must therefore be in tokens:
    #   kv_row_stride = max_blocks_per_req * block_size   (tokens per request)
    # The 1/block_size mismatch was the whole aliasing bug: for R=2,B=2 the kernel
    # computed row 2 (blocks) and landed on token 2 -- block 0's second token --
    # reading 0x11 instead of 0x33.
    bs = int(gathered.k_packed.shape[1])
    kv_row_stride = getattr(metadata, "max_blocks_per_req", None)
    if kv_row_stride is None:
        page_rows = int(page_rows)
        kv_row_stride = page_rows // max(num_reqs, 1)
    kv_row_stride = int(kv_row_stride) * bs

    # Diagnostic writes are opt-in. The kernel publishes per-(request, head)
    # snapshots only when debug is true; otherwise it receives an existing CUDA
    # tensor solely because the JIT signature has a debug slot.
    if debug:
        want = 16 * num_reqs * hk
        global _DBG_BUFFER
        if _DBG_BUFFER is None or _DBG_BUFFER.device != q.device or _DBG_BUFFER.numel() < want:
            _DBG_BUFFER = torch.zeros(want, dtype=torch.int32, device=q.device)
        dbg_arg = from_dlpack(_DBG_BUFFER)
        _torch_args.append(_DBG_BUFFER)
    else:
        dbg_arg = from_dlpack(seq_lens)
        _torch_args.append(seq_lens)
    args = list(args) + [dbg_arg]

    # The q-block grid is per-request, so it must cover the LONGEST request, not
    # the token count. metadata carries it; for a single-request caller without
    # the field, the token count IS the longest request (q_start == 0).
    max_query_len = int(getattr(metadata, "max_query_len", 0) or 0)
    if max_query_len <= 0:
        max_query_len = int(q.shape[0])
    max_query_len = max(max_query_len, 1)

    # Split-K partials. num_splits == 1 passes one split's worth of buffers and
    # compiles the baseline epilogue (split_mode = 0), so that path is unchanged.
    S = max(int(num_splits), 1)
    split_mode = 1 if S > 1 else 0
    if split_mode and max_query_len > 1:
        raise ValueError(
            "split-K decode requires max_query_len == 1, "
            f"got {max_query_len}"
        )
    gqa_mode = 1 if (gqa_pack and int(kernel.qhead_per_kvhead) > 1) else 0
    if gqa_mode and max_query_len > 1:
        raise ValueError(
            "gqa_pack decode requires max_query_len == 1, "
            f"got {max_query_len}"
        )
    part_o_t, part_m_t, part_l_t = _split_buffers(
        num_reqs, S, hq, hd, q.device, q.dtype
    )
    args = args + [
        from_dlpack(part_o_t),
        from_dlpack(part_m_t),
        from_dlpack(part_l_t),
    ]
    _torch_args += [part_o_t, part_m_t, part_l_t]

    if _TIME:
        _t1 = _time.perf_counter()

    _all_args = (
        *args,
        softmax_scale,
        kv_row_stride,
        int(debug),
        max_query_len,
        S,
        int(split_mode),
        int(gqa_mode),
        int(1 if onepass else 0),
        int(1 if reg_rescale else 0),
        int(1 if causal_bound else 0),
        cuda.CUstream(stream),
    )
    if _FASTLAUNCH:
        # EXPERIMENTAL. ``generate_mlir`` regenerates the MLIR module to recompute
        # its hash on every call (~400ms) even though the compiled function is
        # already in ``jit_cache``; only then does it run the cached function.
        # This tries to cache the jit_cache entry and call it directly. Status:
        # the direct call currently fails inside the DSL with "cannot be converted
        # to pointer" (arg adaptation differs from the generate_mlir path), so this
        # needs the proper CUTLASS compiled-function reuse API. Default off.
        key = (
            id(kernel),
            tuple(tuple(t.shape) for t in _torch_args),
            num_reqs, max_query_len, S, int(debug), int(split_mode), int(gqa_mode),
            int(1 if onepass else 0), int(1 if reg_rescale else 0),
            int(1 if causal_bound else 0),
        )
        jf = _FAST.get(key)
        if jf is None:
            dsl = _dsl_object(kernel)
            before = _jitcache_keys(dsl.jit_cache) if dsl is not None else set()
            kernel(*_all_args)
            if dsl is not None:
                new = _jitcache_keys(dsl.jit_cache) - before
                if not new and len(before) == 1:
                    new = before
                if new:
                    jf = dsl.jit_cache.get(next(iter(new)))
                    if jf is not None:
                        _FAST[key] = jf
        else:
            jf(*_all_args)
    else:
        kernel(*_all_args)

    if _TIME:
        _t2 = _time.perf_counter()

    if split_mode:
        _merge_splits(
            part_o_t, part_m_t, part_l_t, num_reqs, S, hq, hd, o3, q_start
        )

    if _TIME:
        _t3 = _time.perf_counter()
        _LAUNCH_TIME["plumbing"] += _t1 - _t0
        _LAUNCH_TIME["kernel"] += _t2 - _t1
        _LAUNCH_TIME["merge"] += _t3 - _t2
        _LAUNCH_TIME["n"] += 1


