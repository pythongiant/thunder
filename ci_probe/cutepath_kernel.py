"""CuTe/CUTLASS-path twin of rawptx_kernel.py at IDENTICAL geometry.

Same M=128 N=64 K=64, same one-atom SWIZZLE_128B operand contents, same issue
count (4), same sync, same readback. The ONLY difference is HOW the operand and
descriptor are produced:

  rawptx_kernel.py : placement hand-computed, descriptor words hand-built, raw
                     inline `tcgen05.mma`
  this file        : placement via the CuTe swizzled layout, descriptor via
                     make_smem_desc_base, MMA via the vendored
                     gemm_ptx_precomputed_varname helper

Purpose: a like-for-like PTX diff. Used by modal_probe_ptxdump.py.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float16, Float32, Int32, Int64, Uint8
from cutlass import pipeline as _pipeline
from cutlass.cute.nvgpu import tcgen05

from thunder_vllm.attention._vendor import blackwell_helpers as _bh
from thunder_vllm.attention._vendor import mma_sm100_desc as _sd

@cute.jit
def sentinel(k):
    """1,2,4,8,... per 16-element K group. Constant data hides cell-map and
    major-mode errors; this does not."""
    v = Float16(1.0)
    if k >= 16:
        v = Float16(2.0)
    if k >= 32:
        v = Float16(4.0)
    if k >= 48:
        v = Float16(8.0)
    if k >= 64:
        v = Float16(16.0)
    if k >= 80:
        v = Float16(32.0)
    if k >= 96:
        v = Float16(64.0)
    if k >= 112:
        v = Float16(128.0)
    return v



M, N, K = 128, 64, 64


@cute.struct
class Store:
    hold: cute.struct.MemRange[cutlass.Int32, 1]
    mbar: cute.struct.MemRange[Int64, 2]
    pad: cute.struct.Align[cute.struct.MemRange[Uint8, 1024], 1024]


@cute.kernel
def k_cute(mOut: cute.Tensor, mDbg: cute.Tensor):
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils_basic.SmemAllocator()
    storage = smem.allocate(Store)

    op = tcgen05.MmaF16BF16Op(
        Float16, Float32, (M, N, 16), tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    la = sm100_utils.make_smem_layout_a(tiled_mma, (M, N, K), Float16, 1)
    lb = sm100_utils.make_smem_layout_b(tiled_mma, (M, N, K), Float16, 1)
    sA = smem.allocate_tensor(element_type=Float16, layout=la.outer,
                              byte_alignment=1024, swizzle=la.inner)
    sB = smem.allocate_tensor(element_type=Float16, layout=lb.outer,
                              byte_alignment=1024, swizzle=lb.inner)

    # operand contents: logical (row, k) -> 1.0, via the operand's own coordinate
    # space (single-atom case here, so the K factor is flat).
    for e in cutlass.range_constexpr((M * K + 127) // 128):
        i = tidx + e * 128
        if i < M * K:
            r = i // K
            k = i % K
            sA[((r, k % 16), 0, k // 16, 0)] = sentinel(k)
    for e in cutlass.range_constexpr((N * K + 127) // 128):
        i = tidx + e * 128
        if i < N * K:
            r = i // K
            k = i % K
            sB[((r, k % 16), 0, k // 16, 0)] = sentinel(k)
    cute.arch.barrier()

    bar = _pipeline.NamedBarrier(barrier_id=1, num_threads=128)
    tmem = utils_basic.TmemAllocator(storage.hold.data_ptr(), barrier_for_retrieve=bar)
    tmem.allocate(256)
    tmem.wait_for_alloc()
    tp = tmem.retrieve_ptr(Float32)
    _frag = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((M, N)))
    tS = cute.make_tensor(tp, _frag.layout)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    la_off = tCrA[None, None, None, 0].layout
    lb_off = tCrB[None, None, None, 0].layout
    a_base = _sd.make_smem_desc_base(
        cute.recast_layout(128, Float16.width, la.outer[0]), la.inner, _sd.Major.K)
    b_base = _sd.make_smem_desc_base(
        cute.recast_layout(128, Float16.width, lb.outer[0]), lb.inner, _sd.Major.K)
    a_start = _sd.make_smem_desc_start_addr(sA[None, None, None, 0].iterator)
    b_start = _sd.make_smem_desc_start_addr(sB[None, None, None, 0].iterator)
    idesc = _bh.sm100_desc.mma_op_to_idesc(op)
    if tidx == 0:
        mDbg[0] = Int32(a_start)
        mDbg[1] = Int32(b_start)
        mDbg[2] = Int32(a_base & 0xFFFFFFFF)
        mDbg[3] = Int32((a_base >> 32) & 0xFFFFFFFF)
        mDbg[4] = Int32(idesc)
        mDbg[5] = Int32(cute.size(la_off, mode=[2]))
        mDbg[6] = Int32(cute.crd2idx((0, 0, 1), la_off))
        mDbg[7] = Int32(cute.crd2idx((0, 0, 1), lb_off))
    cute.arch.barrier()

    _bh.declare_ptx_smem_desc(a_start, a_base, la_off, var_name_prefix="cu_a")
    _bh.declare_ptx_idesc(op, var_name="cu_idesc")
    prod, cons = _pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=_pipeline.CooperativeGroup(_pipeline.Agent.Thread),
        consumer_group=_pipeline.CooperativeGroup(_pipeline.Agent.Thread, 128),
        barrier_storage=storage.mbar.data_ptr(),
    ).make_participants()
    if warp_idx == 0:
        _bh.gemm_ptx_precomputed_varname(
            tS.iterator.toint(), b_start,
            smem_desc_base_b=b_base, tCrB_layout=lb_off,
            smem_var_name_prefix="cu_a", idesc_var_name="cu_idesc",
            smem_offset=0, zero_init=True, cta_group=1,
            kind=_bh._tcgen05_mma_kind(op))
        ph = prod.acquire_and_advance()
        ph.commit()
    cf = cons.wait_and_advance()
    cf.release()
    cute.arch.barrier()

    ld = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(N)), Float32)
    tS2 = tS[(None, None), 0, 0]
    id2 = tiled_mma.get_slice(0).partition_C(
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
def run_kernel(mOut: cute.Tensor, mDbg: cute.Tensor):
    k_cute(mOut, mDbg).launch(grid=(1, 1, 1), block=[128, 1, 1])
