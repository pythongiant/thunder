"""Minimal cp.async G2S probe: runtime row offset + word loads.

Establishes the CuTeDSL idiom before wiring it into the attention kernel:
  * pointer arithmetic on a tensor iterator (`t.iterator + k`)
  * `cute.arch.cp_async_shared_global` with a runtime (non-tile-aligned) offset
  * commit/wait group + barrier

    python ci_probe/cpasync_probe.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import Int32, const_expr  # noqa: E402

TILE = 32
W = 16  # words per row
THREADS = 128
CP = 16  # bytes per cp.async instruction (.cg requires 16)


@cute.kernel
def _copy_kernel(mIn: cute.Tensor, mOut: cute.Tensor, off: Int32):
    tidx = cute.arch.thread_idx()[0]

    @cute.struct
    class S:
        buf: cute.struct.MemRange[cutlass.Int32, TILE * W]

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(S)
    # Explicit row-major stride: cute.make_layout(shape) is column-major.
    sBuf = st.buf.get_tensor(cute.make_layout((TILE, W), stride=(W, 1)))

    # 16-byte cp.async chunks (.cg requires 16 B): 4 words per chunk.
    per_row: const_expr = W * 4 // CP
    total: const_expr = TILE * per_row
    iters: const_expr = (total + THREADS - 1) // THREADS
    for e in cutlass.range_constexpr(iters):
        i = tidx + e * THREADS
        if i < total:
            row = i // per_row
            c = (i % per_row) * (CP // 4)
            cute.arch.cp_async_shared_global(
                sBuf.iterator + (row * W + c),
                mIn.iterator + ((off + row) * W + c),
                CP, "cg",
            )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    el_total: const_expr = TILE * W
    el_iters: const_expr = (el_total + THREADS - 1) // THREADS
    for e in cutlass.range_constexpr(el_iters):
        i = tidx + e * THREADS
        if i < el_total:
            row = i // W
            col = i % W
            mOut[off + row, col] = sBuf[row, col]


@cute.jit
def run_copy(mIn: cute.Tensor, mOut: cute.Tensor, off: int):
    _copy_kernel(mIn, mOut, Int32(off)).launch(
        grid=(1, 1, 1), block=(THREADS, 1, 1))


def main() -> int:
    from cutlass.cute.runtime import from_dlpack

    n, w = 512, W
    src = torch.arange(n * w, dtype=torch.int32, device="cuda").reshape(n, w)
    dst = torch.full((n, w), -1, dtype=torch.int32, device="cuda")
    for off in (0, 37):
        dst.fill_(-1)
        run_copy(from_dlpack(src), from_dlpack(dst), off)
        torch.cuda.synchronize()
        want = src[off:off + TILE]
        got = dst[off:off + TILE]
        ok = bool(torch.equal(got, want))
        print(f"cp.async probe off={off}: {'PASS' if ok else 'FAIL'} "
              f"(maxdiff={int((got-want).abs().max())})", flush=True)
        if not ok:
            print("  want[0,:6] =", want[0, :6].tolist(), flush=True)
            print("  got [0,:6] =", got[0, :6].tolist(), flush=True)
            print("  want[1,:6] =", want[1, :6].tolist(), flush=True)
            print("  got [1,:6] =", got[1, :6].tolist(), flush=True)
            bad = (got != want).any(dim=1).nonzero().flatten().tolist()
            print(f"  bad rows: n={len(bad)} first={bad[:10]}", flush=True)
            if bad:
                r = bad[0]
                print(f"  row {r} want[:4]={want[r,:4].tolist()} "
                      f"got[:4]={got[r,:4].tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
