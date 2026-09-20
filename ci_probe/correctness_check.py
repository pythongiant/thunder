"""Standalone correctness check for the TurboQuant kernel + store.

Runs on any CUDA GPU (no vLLM): store parity (Triton vs reference), gather
parity, dequant parity, and the attention kernel vs the rotated dequant oracle.

    THUNDER_STORE3=1 python ci_probe/correctness_check.py

Exits non-zero on any failure. Intended for cheap non-Blackwell sandboxes
(set THUNDER_ALLOW_ARCH if you also drive the vLLM backend).
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thunder_vllm.attention.cache_layout import (  # noqa: E402
    ThunderCacheLayout,
    allocate_kv_cache,
    reshape_and_cache,
    reshape_and_cache_ref,
)
from thunder_vllm.attention.cute_kernel import (  # noqa: E402
    ThunderAttentionForward,
    launch_thunder_attention,
)
from thunder_vllm.attention.paged_kv import make_paged_kv_manager  # noqa: E402
from thunder_vllm.quant.quantizer import ThunderQuantizer  # noqa: E402

HD, HK, BS = 128, 8, 16
MB = int(os.environ.get("THUNDER_M_BLOCK", 64))
NB = int(os.environ.get("THUNDER_N_BLOCK", 64))
dev = "cuda"
FAILURES: list[str] = []


def check(name: str, value: float, tol: float) -> None:
    ok = math.isfinite(value) and value <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {value:.3e} (tol {tol:.0e})",
          flush=True)
    if not ok:
        FAILURES.append(name)


def md(a, b) -> float:
    return float((a.float() - b.float()).abs().max().item())


def store_and_gather(k_bits: int, v_bits: int, nt: int = 256) -> None:
    print(f"[store/gather] K={k_bits} V={v_bits} n={nt}", flush=True)
    layout = ThunderCacheLayout(num_kv_heads=HK, head_dim=HD, k_bits=k_bits,
                                v_bits=v_bits, block_size=BS)
    quant = ThunderQuantizer(HD, k_bits, v_bits, device=dev)
    nb = math.ceil(nt / BS)
    key = torch.randn(nt, HK, HD, device=dev, dtype=torch.float16)
    val = torch.randn(nt, HK, HD, device=dev, dtype=torch.float16)
    slots = torch.arange(nt, device=dev, dtype=torch.long)

    kv_r, sc_r = allocate_kv_cache(nb, BS, HK, HD, k_bits, v_bits, device=dev)
    kv_e, sc_e = allocate_kv_cache(nb, BS, HK, HD, k_bits, v_bits, device=dev)
    reshape_and_cache_ref(key, val, slots, kv_r, sc_r, quant, layout)
    reshape_and_cache(key, val, slots, kv_e, sc_e, quant, layout)
    torch.cuda.synchronize()

    check(f"store K codes ({k_bits}b)", md(layout.k_codes(kv_e), layout.k_codes(kv_r)), 0.0)
    check(f"store V codes ({v_bits}b)", md(layout.v_codes(kv_e), layout.v_codes(kv_r)), 0.0)
    check("store K norms", md(layout.k_norm(sc_e), layout.k_norm(sc_r)), 2e-2)
    check("store V norms", md(layout.v_norm(sc_e), layout.v_norm(sc_r)), 2e-2)

    mgr = make_paged_kv_manager(layout, max_num_reqs=1, max_model_len=nb * BS, device=dev)
    bt = torch.arange(nb, device=dev, dtype=torch.int32).reshape(1, nb)
    sl = torch.tensor([nt], device=dev, dtype=torch.int32)
    g = mgr.gather_packed_tiles(bt, kv_e, sc_e, sl)
    check("gather K", md(g.k_packed.reshape(-1, HK, layout.k_packed_bytes)[:nt],
          layout.k_codes(kv_e).reshape(-1, HK, layout.k_packed_bytes)[:nt]), 0.0)
    check("gather V", md(g.v_packed.reshape(-1, HK, layout.v_packed_bytes)[:nt],
          layout.v_codes(kv_e).reshape(-1, HK, layout.v_packed_bytes)[:nt]), 0.0)


def kernel_parity(k_bits: int, v_bits: int, causal: bool, nt: int = 256,
                  flags: tuple = (False, False, False)) -> None:
    op, rr, cb = flags
    tag = "prefill" if causal else "decode"
    print(f"[kernel] {tag} K={k_bits} V={v_bits} n={nt} "
          f"onepass={op} reg_rescale={rr} causal_bound={cb}", flush=True)
    layout = ThunderCacheLayout(num_kv_heads=HK, head_dim=HD, k_bits=k_bits,
                                v_bits=v_bits, block_size=BS)
    quant = ThunderQuantizer(HD, k_bits, v_bits, device=dev)
    nb = math.ceil(nt / BS)
    key = torch.randn(nt, HK, HD, device=dev, dtype=torch.float16)
    val = torch.randn(nt, HK, HD, device=dev, dtype=torch.float16)
    slots = torch.arange(nt, device=dev, dtype=torch.long)
    kv, sc = allocate_kv_cache(nb, BS, HK, HD, k_bits, v_bits, device=dev)
    reshape_and_cache(key, val, slots, kv, sc, quant, layout)
    torch.cuda.synchronize()

    mgr = make_paged_kv_manager(layout, max_num_reqs=1, max_model_len=nb * BS, device=dev)
    bt = torch.arange(nb, device=dev, dtype=torch.int32).reshape(1, nb)
    sl = torch.tensor([nt], device=dev, dtype=torch.int32)
    g = mgr.gather_packed_tiles(bt, kv, sc, sl)

    nq = nt if causal else 1
    q = torch.randn(nq, HK, HD, device=dev, dtype=torch.float16)
    qr = (q.float() @ quant.rotation.matrix.float()).to(torch.float16)
    out = torch.zeros_like(qr)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([nt], device=dev, dtype=torch.int32),
        query_start_loc=torch.tensor([0, nq], device=dev, dtype=torch.int32),
        max_blocks_per_req=nb, max_query_len=nq)
    fwd = ThunderAttentionForward(head_dim=HD, K_BITS=k_bits, V_BITS=v_bits,
                                  qhead_per_kvhead=1, is_causal=causal,
                                  m_block_size=MB, n_block_size=NB, num_threads=128)
    launch_thunder_attention(fwd, qr, g, out, meta, HD ** -0.5, quantizer=quant,
                             onepass=op, reg_rescale=rr, causal_bound=cb)
    torch.cuda.synchronize()

    k_hat = quant.dequantize_k_rotated(
        g.k_packed.reshape(-1, HK, layout.k_packed_bytes), g.k_norm.reshape(-1, HK))
    v_hat = quant.dequantize_v_rotated(
        g.v_packed.reshape(-1, HK, layout.v_packed_bytes), g.v_norm.reshape(-1, HK))
    ref = F.scaled_dot_product_attention(
        qr.float().transpose(0, 1).unsqueeze(0),
        k_hat.float().transpose(0, 1).unsqueeze(0),
        v_hat.float().transpose(0, 1).unsqueeze(0),
        is_causal=causal, scale=HD ** -0.5)[0].transpose(0, 1).float()
    check(f"kernel {tag} vs rotated oracle", md(out, ref), 5e-3)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device", flush=True)
        return 2
    print(f"device: {torch.cuda.get_device_name(0)} m_block={MB} n_block={NB} "
          f"cc={torch.cuda.get_device_capability(0)} "
          f"store3={os.environ.get('THUNDER_STORE3', '0')}", flush=True)

    store_and_gather(4, 4)
    if os.environ.get("THUNDER_STORE3", "0").strip().lower() not in (
            "", "0", "false", "no", "off"):
        store_and_gather(3, 4)
    # Baseline and the now-default fast paths (onepass + reg_rescale + causal).
    for flags in ((False, False, False), (True, True, True)):
        kernel_parity(4, 4, causal=True, flags=flags)
        kernel_parity(4, 4, causal=False, flags=flags)
        kernel_parity(3, 4, causal=True, flags=flags)
        kernel_parity(3, 4, causal=False, flags=flags)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)}): {FAILURES}", flush=True)
        return 1
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
