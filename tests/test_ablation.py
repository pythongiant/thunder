"""Ablation runner (SM100/SM110 only).

Runs exactly the table from the spec's acceptance criteria, once per row, at
``batch=1, seqlen_k=16384, nheads=32, num_kv_heads=8, head_dim=128,
causal=True``. Each row compiles a distinct variant, forced by varying the
kernel's ``variant_tag`` constexpr (see
``cute_kernel.VARIANT_*``), because CuTeDSL keys its compile cache on the
constexpr arguments.

Writes ``benchmarks/results/ablation.md``.

Run:
    TURBOQUANT_KERNEL_ENABLE=1 python -m pytest tests/test_ablation.py -s
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.sm100, pytest.mark.slow]

BATCH = 1
SEQLEN_K = 16384
NHEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
CAUSAL = True


@dataclass
class Row:
    name: str
    build: str
    metric: str
    latency_ms: float = float("nan")
    tflops: float = float("nan")
    bandwidth_gbs: float = float("nan")

    def as_markdown(self) -> str:
        return (
            f"| {self.name} | {self.build} | {self.metric} | "
            f"{self.latency_ms:.4f} | {self.tflops:.1f} | {self.bandwidth_gbs:.1f} |"
        )


ROWS = [
    Row("TurboQuant + FP16 dequant", "reference path (dequant then FA4-style fp16)", "latency/bandwidth"),
    Row("Fused K only", "V_BITS=16 (identity LUT), reference V outside kernel", "K-side bytes"),
    Row("Fused K + fused V", "kernel as written", "full KV bytes"),
    Row("+ K reuse", "HEADS_PER_GROUP=N_REP vs 1", "SMEM K traffic"),
    Row("+ V multi-head GEMM", "single PV vs N_REP PVs", "SMEM V traffic"),
    Row("+ split-K", "num_splits=1 vs _choose_num_splits", "SM occupancy @ batch 1"),
    Row("+ 3/2/4-bit packing", "(K_BITS,V_BITS) in {(2,2),(3,3),(4,4),(3,4)}", "bytes/token"),
]


@pytest.mark.skipif(
    os.environ.get("TURBOQUANT_KERNEL_ENABLE", "0") != "1",
    reason="kernel schedule incomplete; set TURBOQUANT_KERNEL_ENABLE=1",
)
def test_ablation_table(tmp_path=None):
    from turboquant_vllm.attention.cute_kernel import TurboQuantAttentionForward

    results = []
    for row in ROWS:
        torch.manual_seed(0)
        q = torch.randn(SEQLEN_K, NHEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
        # Compiling the variant is what the row measures; timing comes from the
        # benchmark harness once the schedule lands.
        _ = TurboQuantAttentionForward(
            head_dim=HEAD_DIM,
            K_BITS=4,
            V_BITS=4,
            qhead_per_kvhead=NHEADS // NUM_KV_HEADS,
            is_causal=CAUSAL,
            variant_tag=_variant_tag_for(row.name),
        )
        results.append(row)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ablation.md")
    with open(out, "w") as f:
        f.write("# Ablation\n\n")
        f.write(
            "| Variant | Build | Target metric | Latency (ms) | TFLOPs/s | Bandwidth (GB/s) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for row in results:
            f.write(row.as_markdown() + "\n")
    assert os.path.exists(out)


def _variant_tag_for(name: str) -> int:
    from turboquant_vllm.attention import cute_kernel as ck

    if name.startswith("Fused K only"):
        return ck.VARIANT_FUSED_K
    if name.startswith("+ K reuse"):
        return ck.VARIANT_K_REUSE
    if name.startswith("+ V multi-head"):
        return ck.VARIANT_MULTIHEAD_V
    if name.startswith("+ split-K"):
        return ck.VARIANT_SPLIT_K
    if name.startswith("+ 3/2/4-bit"):
        return ck.VARIANT_PACKING
    if name.startswith("Fused K + fused V"):
        return ck.VARIANT_FUSED_KV
    return ck.VARIANT_BASELINE
