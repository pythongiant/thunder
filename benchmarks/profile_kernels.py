"""``ncu`` profiling hooks.

Runs each ablation variant under Nsight Compute with the metric set from the
spec and writes ``benchmarks/results/ncu_<variant>_<shape>.csv``. The metric
list is chosen so the CSV answers *why* a variant is slow:

* tensor-core utilisation      -> is the MMA the bottleneck?
* DRAM read bytes              -> is the kernel bandwidth-bound (desired at long
                                  context) or is dequant re-reading?
* SMEM bank conflicts          -> is the LUT gather thrashing banks?
* barrier stall %              -> is the dequant warp blocking the MMA warp?
* TMEM allocation columns      -> accumulator pressure / occupancy limit

Run:
    PROFILE=1 python -m benchmarks.profile_kernels --out-dir benchmarks/results
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

METRICS = [
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "sm__tmem_alloc_cols",
]

VARIANTS = [
    ("baseline", 0),
    ("fused_k", 1),
    ("fused_kv", 2),
    ("k_reuse", 3),
    ("multihead_v", 4),
    ("split_k", 5),
]

SHAPES = [
    ("decode_long", 1, 1, 32768, 32, 8, 128),
    ("prefill", 1, 4096, 4096, 32, 8, 128),
]


def _has_ncu() -> bool:
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="benchmarks/results")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not _has_ncu():
        print("ncu not found; skipping profiling", file=sys.stderr)
        return

    for variant, tag in VARIANTS:
        for shape, batch, sq, sk, nh, nkv, hd in SHAPES:
            out_csv = os.path.join(args.out_dir, f"ncu_{variant}_{shape}.csv")
            cmd = [
                "ncu",
                "--csv",
                "--metrics",
                ",".join(METRICS),
                "--target-processes",
                "all",
                "-o",
                out_csv.replace(".csv", ""),
                args.python,
                "-m",
                "benchmarks._profile_entry",
                "--variant-tag",
                str(tag),
                "--shape",
                shape,
            ]
            print("running", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=False)
    print(f"profiles written under {args.out_dir}")


if __name__ == "__main__":
    main()
