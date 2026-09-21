"""Raw-PTX tcgen05.mma kernel source (exec'd inside the container).

Hand-written everywhere it matters:
  * SWIZZLE_128B operand placement computed by hand
  * descriptor words hand-built
  * instruction descriptor hand-built
  * raw inline PTX `tcgen05.mma.cta_group::1.kind::f16`

Only the MMA is raw; sync/alloc/readback use the helpers already proven correct.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Float16, Float32, Int32, Int64, Uint8
from cutlass import pipeline as _pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05

M, N, K = 128, 64, 64
CELL_F16 = 8          # fp16 per 16B cell
ROW_F16 = 64          # f16 per row = 128 B = 8 cells

# descriptor words, hand-built
#   bits [0:14)  start_address (16B units)  -- filled per issue
#   bits [16:30) LBO = 1
#   bits [32:46) SBO = 64
#   bits [46:48) version = 1
#   bits [61:64) layout_type = SWIZZLE_128B = 2
DESC_LO = 0x00010000
DESC_HI = 0x40004040

# instruction descriptor, hand-built
#   c_format=F32(1) a_format=F16(0) b_format=F16(0)
#   a_major and b_major are BOTH Major.K, and in this encoding Major.K == 0
#   (Major.MN == 1), so neither bit 15 nor bit 16 is set. This matters: an
#   earlier version of this file set both bits, i.e. declared MN-major operands
#   with a K-major descriptor, and the all-ones operand data hid it.
IDESC = ((1 << 4) | ((N >> 3) << 17) | ((M >> 4) << 24))


@cute.struct
class Store:
    hold: cute.struct.MemRange[cutlass.Int32, 1]
    mbar: cute.struct.MemRange[Int64, 2]
    pad: cute.struct.Align[cute.struct.MemRange[Uint8, 1024], 1024]


@cute.jit
def swipe(row, cell):
    """SWIZZLE_128B: 16B-cell index XOR row-within-8."""
    return cell ^ (row % 8)


@cute.jit
def mma_raw(a_start_u128: Int32, b_start_u128: Int32, acc_tmem: Int32, pred: Int32):
    """One raw tcgen05.mma, cta_group::1, kind::f16, instruction K=16."""
    llvm.inline_asm(
        None,
        [
            Int32(DESC_LO | a_start_u128).ir_value(),
            Int32(DESC_HI).ir_value(),
            Int32(DESC_LO | b_start_u128).ir_value(),
            Int32(DESC_HI).ir_value(),
            Int32(IDESC).ir_value(),
            Int32(pred).ir_value(),
            Int32(acc_tmem).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "setp.ne.b32 p, $5, 0;\n\t"
        "mov.b64 smem_desc_a, {$0, $1};\n\t"
        "mov.b64 smem_desc_b, {$2, $3};\n\t"
        "@leader_thread tcgen05.mma.cta_group::1.kind::f16 "
        "[$6], smem_desc_a, smem_desc_b, $4, p;\n\t"
        "}\n",
        "r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.kernel
def k_raw(mOut: cute.Tensor, mDbg: cute.Tensor, issue_mask: Int32, zero_mask: Int32):
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils_basic.SmemAllocator()
    storage = smem.allocate(Store)
    sA = smem.allocate_tensor(
        element_type=Float16, layout=cute.make_layout(M * ROW_F16), byte_alignment=1024)
    sB = smem.allocate_tensor(
        element_type=Float16, layout=cute.make_layout(N * ROW_F16), byte_alignment=1024)

    for e in cutlass.range_constexpr((M * K + 127) // 128):
        i = tidx + e * 128
        if i < M * K:
            r = i // K
            k = i % K
            # sentinel matrix: k 0..15 -> 1.0, 16..31 -> 2.0, 32..47 -> 4.0,
            # 48..63 -> 8.0. Constant data cannot reveal a wrong cell map or a
            # wrong major mode; this can.
            sv = Float16(1.0)
            if k >= 16:
                sv = Float16(2.0)
            if k >= 32:
                sv = Float16(4.0)
            if k >= 48:
                sv = Float16(8.0)
            sA[r * ROW_F16 + swipe(r, k // CELL_F16) * CELL_F16 + (k % CELL_F16)] = sv
    for e in cutlass.range_constexpr((N * K + 127) // 128):
        i = tidx + e * 128
        if i < N * K:
            r = i // K
            k = i % K
            sB[r * ROW_F16 + swipe(r, k // CELL_F16) * CELL_F16 + (k % CELL_F16)] = Float16(1.0)
    cute.arch.barrier()

    bar = _pipeline.NamedBarrier(barrier_id=1, num_threads=128)
    tmem = utils_basic.TmemAllocator(storage.hold.data_ptr(), barrier_for_retrieve=bar)
    tmem.allocate(256)
    tmem.wait_for_alloc()
    tp = tmem.retrieve_ptr(Float32)
    acc_addr = tp.toint()

    a_base = (sA.iterator.toint() & 0x3FFFF) >> 4
    b_base = (sB.iterator.toint() & 0x3FFFF) >> 4
    if tidx == 0:
        mDbg[0] = Int32(a_base)
        mDbg[1] = Int32(b_base)
        mDbg[2] = Int32(DESC_LO)
        mDbg[3] = Int32(DESC_HI)
        mDbg[4] = Int32(IDESC)
    cute.arch.barrier()

    prod, cons = _pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=_pipeline.CooperativeGroup(_pipeline.Agent.Thread),
        consumer_group=_pipeline.CooperativeGroup(_pipeline.Agent.Thread, 128),
        barrier_storage=storage.mbar.data_ptr(),
    ).make_participants()

    if warp_idx == 0:
        for j in cutlass.range_constexpr(4):
            if ((issue_mask >> j) & 1) == 1:
                mma_raw(a_base + 2 * j, b_base + 2 * j, acc_addr,
                        Int32(1) - ((zero_mask >> j) & 1))
        ph = prod.acquire_and_advance()
        ph.commit()
    cf = cons.wait_and_advance()
    cf.release()
    cute.arch.barrier()

    # TMEM accumulator layout. A TiledMma is constructed ONLY to obtain the C
    # fragment layout for the readback; the MMA itself above is raw PTX.
    _op = tcgen05.MmaF16BF16Op(
        Float16, Float32, (M, N, 16), tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    _tiled = cute.make_tiled_mma(_op)
    _frag = _tiled.make_fragment_C(_tiled.partition_shape_C((M, N)))
    tS = cute.make_tensor(tp, _frag.layout)
    ld = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(N)), Float32)
    tS2 = tS[(None, None), 0, 0]
    id2 = _tiled.get_slice(0).partition_C(
        cute.make_identity_tensor((M, N)))[(None, None), 0, 0]
    tc = tcgen05.make_tmem_copy(ld, tS2)
    thr = tc.get_slice(tidx)
    tD = thr.partition_S(tS2)
    rS = cute.make_rmem_tensor(thr.partition_D(id2).shape, Float32)
    cute.arch.fence_view_async_tmem_load()
    cute.copy(ld, tD, rS)
    rF = cute.group_modes(rS, 0, cute.rank(rS))
    if tidx < M:
        for n in cutlass.range_constexpr(N):
            mOut[tidx, n] = rF[n]
    cute.arch.barrier()

    tmem.relinquish_alloc_permit()
    tmem.free(tp, 256)


@cute.jit
def run_kernel(mOut: cute.Tensor, mDbg: cute.Tensor, issue_mask, zero_mask):
    k_raw(mOut, mDbg, Int32(issue_mask), Int32(zero_mask)).launch(
        grid=(1, 1, 1), block=[128, 1, 1])
