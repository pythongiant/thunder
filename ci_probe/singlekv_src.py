"""GPU-side driver for the single-KV-tile state-transition diagnostic.

Run by ci_probe/modal_probe_singlekv.py inside the B200 container.
See that file for what each dbg_mode asserts.
"""

import math
import os
import traceback

import torch

TILE_M, TILE_N, HDIM, BITS = 128, 64, 128, int(os.environ.get('TQ_BITS', '4'))
Q_LEN, KV_LEN = 128, 64


def build():
    dev = "cuda"
    torch.manual_seed(0)
    # Identity LUTs -> dequant value == code, so the host reference is exact and
    # the quantizer is out of the picture.
    k_lut = (
        torch.arange(2**BITS, device=dev, dtype=torch.float16)[:, None]
        .expand(2**BITS, HDIM)
        .contiguous()
    )
    v_lut = k_lut.clone()
    code_k = torch.randint(0, 2**BITS, (KV_LEN, HDIM), device=dev, dtype=torch.int64)
    code_v = torch.randint(0, 2**BITS, (KV_LEN, HDIM), device=dev, dtype=torch.int64)
    return k_lut, v_lut, code_k, code_v


def run(dbg_mode):
    import importlib as _il

    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    mod = _il.import_module(
        os.environ.get("TQ_KERNEL_MODULE", "turboquant_vllm.attention.cute_kernel_tcgen05")
    )
    Fwd = mod.TurboQuantAttentionForward
    from turboquant_vllm.quant.packing import pack_indices

    k_lut, v_lut, code_k, code_v = build()

    q = torch.randn(Q_LEN, 1, HDIM, device="cuda", dtype=torch.float16) * 0.5
    kn = torch.rand(KV_LEN, 1, device="cuda", dtype=torch.float16) + 0.5
    vn = torch.rand(KV_LEN, 1, device="cuda", dtype=torch.float16) + 0.5
    k_packed = pack_indices(code_k, BITS, HDIM).reshape(KV_LEN, 1, -1).contiguous()
    v_packed = pack_indices(code_v, BITS, HDIM).reshape(KV_LEN, 1, -1).contiguous()
    out = torch.zeros(Q_LEN, 1, HDIM, device="cuda", dtype=torch.float16)
    seq_lens = torch.tensor([KV_LEN], device="cuda", dtype=torch.int32)
    q_start = torch.tensor([0, Q_LEN], device="cuda", dtype=torch.int32)
    dbgS = torch.full((TILE_M, TILE_N), -777.0, device="cuda", dtype=torch.float32)
    dbgO = torch.full((TILE_M, HDIM), -777.0, device="cuda", dtype=torch.float32)

    fwd = Fwd(
        head_dim=HDIM,
        K_BITS=BITS,
        V_BITS=BITS,
        qhead_per_kvhead=1,
        is_causal=False,
        m_block_size=TILE_M,
        n_block_size=TILE_N,
        num_threads=128,
    )
    scale = HDIM**-0.5
    args = [
        from_dlpack(t)
        for t in (q, k_packed, v_packed, kn, vn, k_lut, v_lut, out, seq_lens, q_start)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fwd(*args, scale, stream, from_dlpack(dbgS), from_dlpack(dbgO), dbg_mode)
    torch.cuda.synchronize()

    Kd = code_k.to(torch.float32)
    Vd = code_v.to(torch.float32)
    Qf = q[:, 0, :].float()
    S = Qf @ Kd.T

    if dbg_mode == 1:
        got, ref, name = dbgS, S, "S (raw QK)"
    elif dbg_mode == 5:
        got, ref, name = dbgS, torch.zeros_like(dbgS), "sK_code physical dump"
    elif dbg_mode == 6:
        # Arrangement test. sQ -> dbgO (128x128 f16), sK_code -> dbgS (128x64 f16).
        # Sentinel is unique per logical k (k+1), so each logical row must contain
        # a permutation of 1..128 across its two 64-f16 swizzle atoms:
        #   atom0 window = [r*64, r*64+64)          (k = 0..63)
        #   atom1 window = [8192 + r*64, 8192+r*64+64) (k = 64..127)
        def check(flat, rows, atom1, label):
            d = flat.float()
            EXP = torch.arange(1, 129, device="cuda", dtype=torch.float32)
            bad_rows = 0
            first = None
            for r in range(rows):
                row = torch.cat([d[r * 64: r * 64 + 64],
                                 d[atom1 + r * 64: atom1 + r * 64 + 64]])
                if not torch.equal(torch.sort(row).values, EXP):
                    bad_rows += 1
                    if first is None:
                        vals, cnts = torch.unique(row, return_counts=True)
                        dup = [(float(v), int(c)) for v, c in zip(vals, cnts) if c > 1]
                        miss = sorted(set(EXP.tolist()) - set(vals.tolist()))
                        first = (r, dup[:5], miss[:5])
            print(f"  {label}: rows={rows} rows_not_a_permutation_of_1..128 = {bad_rows}")
            if first is not None:
                print(f"    first bad row {first[0]}: duplicated={first[1]} missing(<=5)={first[2]}")
            return bad_rows

        # sQ: 128 rows x 64 f16 per atom -> atom1 at 8192
        b1 = check(dbgO.reshape(-1), 128, 8192, "sQ  (Q operand)")
        # sK: 64 rows x 64 f16 per atom -> atom1 at 4096 (mode2 stride was 4096 f16)
        b2 = check(dbgS.reshape(-1), 64, 4096, "sK_code (B operand)")
        print()
        print("  INTERPRETATION:")
        print("   both 0 -> every logical (row,k) stored exactly once => 4x is in the")
        print("             MMA's K addressing (deposit exonerated for good)")
        print("   nonzero -> that operand is stored more than once per logical (row,k)")
        return

    elif dbg_mode == 4:
        # Per-k one-hot: acc == mult(k). Correct kernel -> 1.0 for every k.
        print("  per-k read multiplicity (expect 1.0 everywhere):")
        mult = []
        for kcol in range(128):
            dbgS.zero_()
            qs = torch.tensor([0, kcol], device="cuda", dtype=torch.int32)
            a2 = list(args)
            a2[9] = from_dlpack(qs)
            fwd(*a2, 1.0, stream, from_dlpack(dbgS), from_dlpack(dbgO), dbg_mode)
            torch.cuda.synchronize()
            mult.append(round(float(dbgS[0, 0]), 4))
        for b in range(0, 128, 16):
            print(f"    k={b:>3}..{b+15:>3}: {mult[b:b+16]}")
        bad = sum(1 for x in mult if abs(x - 1.0) > 1e-3)
        print(f"    correct k: {128 - bad}/128")
        return

    elif dbg_mode == 6:
        # Arrangement test. sQ -> dbgO (128x128 f16), sK_code -> dbgS (128x64 f16).
        # Sentinel is unique per logical k (k+1), so each logical row must contain
        # a permutation of 1..128 across its two 64-f16 swizzle atoms:
        #   atom0 window = [r*64, r*64+64)          (k = 0..63)
        #   atom1 window = [8192 + r*64, 8192+r*64+64) (k = 64..127)
        def check(flat, rows, atom1, label):
            d = flat.float()
            EXP = torch.arange(1, 129, device="cuda", dtype=torch.float32)
            bad_rows = 0
            first = None
            for r in range(rows):
                row = torch.cat([d[r * 64: r * 64 + 64],
                                 d[atom1 + r * 64: atom1 + r * 64 + 64]])
                if not torch.equal(torch.sort(row).values, EXP):
                    bad_rows += 1
                    if first is None:
                        vals, cnts = torch.unique(row, return_counts=True)
                        dup = [(float(v), int(c)) for v, c in zip(vals, cnts) if c > 1]
                        miss = sorted(set(EXP.tolist()) - set(vals.tolist()))
                        first = (r, dup[:5], miss[:5])
            print(f"  {label}: rows={rows} rows_not_a_permutation_of_1..128 = {bad_rows}")
            if first is not None:
                print(f"    first bad row {first[0]}: duplicated={first[1]} missing(<=5)={first[2]}")
            return bad_rows

        # sQ: 128 rows x 64 f16 per atom -> atom1 at 8192
        b1 = check(dbgO.reshape(-1), 128, 8192, "sQ  (Q operand)")
        # sK: 64 rows x 64 f16 per atom -> atom1 at 4096 (mode2 stride was 4096 f16)
        b2 = check(dbgS.reshape(-1), 64, 4096, "sK_code (B operand)")
        print()
        print("  INTERPRETATION:")
        print("   both 0 -> every logical (row,k) stored exactly once => 4x is in the")
        print("             MMA's K addressing (deposit exonerated for good)")
        print("   nonzero -> that operand is stored more than once per logical (row,k)")
        return

    elif dbg_mode == 4:
        # Two live K blocks: block a at amplitude 1, block b at amplitude amp.
        # acc/16 = mult(a) + amp^2 * mult(b).  amp = 2 -> acc/16 = m_a + 4*m_b.
        CASES = [
            (0, 0, 1.0, "calib single block 0"),
            (1, 1, 1.0, "calib single block 1"),
            (0, 1, 2.0, "{0,1}"),
            (1, 2, 2.0, "{1,2}"),
            (2, 3, 2.0, "{2,3}"),
            (3, 4, 2.0, "{3,4}"),
            (6, 7, 2.0, "{6,7}"),
            (1, 3, 2.0, "{1,3} non-adjacent"),
            (0, 7, 2.0, "{0,7} distant"),
        ]
        print("  two-block footprint probe (amp=2 -> acc/16 = m_a + 4*m_b):")
        print("    block0-only mult=1, blocks1..7 mult=4 predicts:")
        print("      single0 -> 16 | single1 -> 64")
        print("      {0,1} -> 1+16=17 -> 272   {1,2}/{2,3}/{3,4}/{6,7}/{1,3} -> 4+16=20 -> 320")
        for a, b, amp, tag in CASES:
            dbgS.zero_()
            qs = torch.tensor([0, a * 8 + b], device="cuda", dtype=torch.int32)
            a2 = list(args)
            a2[9] = from_dlpack(qs)
            fwd(*a2, float(amp), stream, from_dlpack(dbgS), from_dlpack(dbgO), dbg_mode)
            torch.cuda.synchronize()
            v = float(dbgS[0, 0])
            print(f"    {tag:<22} acc={v:9.3f}  acc/16={v/16.0:7.3f}")
        return

    else:
        if dbg_mode == 2:
            m = torch.arange(TILE_M, device="cuda").float()[:, None]
            n = torch.arange(TILE_N, device="cuda").float()[None, :]
            P = (((m + n) % 4) + 1) * 0.25
        else:
            x = S * kn[:, 0].float()[None, :] * scale
            P = torch.exp2(
                (x - x.max(dim=1, keepdim=True).values) * math.log2(math.e)
            )
        Pe = P * vn[:, 0].float()[None, :]
        got, ref, name = dbgO, Pe @ Vd, "O (raw PV)"

    if dbg_mode == 4:
        r = dbgS[127]
        print(f"  production descriptors: a_base LBO={int(r[1])} SBO={int(r[2])} "
              f"ltype={int(r[3])} | b_base LBO={int(r[4])} SBO={int(r[5])}")
        print(f"  production idesc={hex(int(r[6]))} a_start={int(r[7])} b_start={int(r[8])}")
        print(f"  (twin was: a_base 0xc0 LBO=1 SBO=64 ltype=2, b_base 0x8c0, "
              f"idesc=0x8100010)")
    if dbg_mode == 4:
        r = dbgS[126]
        print(f"  sl_qk_a crd2idx k=0..7: {[int(v) for v in r[0:8]]}")
        print(f"  sl_qk_b crd2idx k=0..7: {[int(v) for v in r[8:16]]}")
        print(f"  num_k_tile a={int(r[16])} b={int(r[17])}  per-thread regs={int(r[20])}")
        print(f"  (twin: A [0,2,4,6,1024,1026,1028,1030]  "
              f"B [0,2,4,6,512,514,516,518]  num_k_tile 8)")
    d = (got - ref).abs()
    print(
        f"  [{name}] max_abs={d.max().item():.6e} mean_abs={d.mean().item():.6e} "
        f"ref_absmax={ref.abs().max().item():.4f}"
    )
    for r in (0, 1, 5):
        print(f"  [{name}] got[{r},:6]={[round(float(v), 4) for v in got[r, :6]]}")
        print(f"  [{name}] ref[{r},:6]={[round(float(v), 4) for v in ref[r, :6]]}")
    sent = int((got == -777.0).all(dim=1).sum().item())
    print(f"  [{name}] fully-untouched rows: {sent}/{got.shape[0]}")


def main():
    modes = [int(os.environ["DBG"])] if os.environ.get("DBG") else [1, 2, 3]
    print(
        "device:",
        torch.cuda.get_device_name(0),
        "cc:",
        torch.cuda.get_device_capability(0),
    )
    for m in modes:
        print(f"===== dbg_mode {m} =====")
        try:
            run(m)
        except Exception:
            traceback.print_exc()
        print()


main()
