"""Collapse ``ncu_*.csv`` into ``ncu_summary.md``.

The summary is deliberately interpretive: it states, per variant, whether the
kernel is bandwidth-bound (good -- that is the TurboQuant design point) or
dequant/barrier-bound (bad -- the dequant pipeline is the bottleneck).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

BOUNDS = {
    "tensor_core_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "dram_read_bytes": "dram__bytes_read.sum",
    "smem_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "barrier_stall_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "tmem_cols": "sm__tmem_alloc_cols",
}


def _read_rows(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            metric = row.get("Metric Name") or row.get("metric_name")
            value = row.get("Metric Value") or row.get("metric_value")
            if metric:
                out[metric] = value
    return out


def classify(rows: dict[str, str]) -> str:
    try:
        tc = float(rows.get(BOUNDS["tensor_core_pct"], "0"))
        stall = float(rows.get(BOUNDS["barrier_stall_pct"], "0"))
    except ValueError:
        return "unknown"
    if stall > 30.0:
        return "dequant-warp-bound (barrier stall high)"
    if tc < 20.0:
        return "bandwidth-bound (tensor cores idle)"
    return "compute-bound (tensor cores active)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchmarks/results")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "ncu_*.csv")))
    out = os.path.join(args.results_dir, "ncu_summary.md")
    with open(out, "w") as f:
        f.write("# ncu summary\n\n")
        if not files:
            f.write("No `ncu_*.csv` files found; run with `PROFILE=1`.\n")
        else:
            f.write(
                "| file | tensor-core % | DRAM read (B) | SMEM bank conflicts | "
                "barrier stall % | TMEM cols | verdict |\n"
            )
            f.write("|---" * 7 + "|\n")
            for path in files:
                r = _read_rows(path)
                f.write(
                    f"| {os.path.basename(path)} | {r.get(BOUNDS['tensor_core_pct'],'')} | "
                    f"{r.get(BOUNDS['dram_read_bytes'],'')} | "
                    f"{r.get(BOUNDS['smem_bank_conflicts'],'')} | "
                    f"{r.get(BOUNDS['barrier_stall_pct'],'')} | "
                    f"{r.get(BOUNDS['tmem_cols'],'')} | {classify(r)} |\n"
                )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
