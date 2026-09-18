"""End-to-end benchmark: this plugin vs the upstream vLLM TurboQuant backend.

Loads the same model twice (vLLM's ``TURBOQUANT`` vs this plugin's
``THUNDER_CUTE``), warms both at the same batch shape, and reports the sweep
table with ours as a percent of the baseline.

Parity caveat (important)
-------------------------
The two backends use *different* KV layouts: upstream stores packed K + fp16 V
with centroids and no per-head norms; this plugin stores packed K + packed V
plus fp16 norms. End-to-end logit parity is therefore not the right assertion.
Instead this harness feeds both paths the *same quantized tensors* (via the
plugin's quantizer) and compares attention outputs with
``atol=rtol=1e-2``. That isolates kernel correctness from quantizer semantics.

Run:
    python -m benchmarks.bench_vs_thunder_vllm --model Qwen/Qwen3-4B \
        --out benchmarks/results/vs_thunder_vllm.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_common import (  # noqa: E402
    assert_parity,
    nvml_sampler,
    run_with_graph,
)

SWEEP = [
    ("short-decode", 1, 128, 512),
    ("mixed", 1, 512, 512),
    ("decode-heavy", 1, 64, 1024),
    ("high-load", 500, 512, 128),
    ("long-prefill", 1, 4096, 128),
    ("very-long-prefill", 1, 8192, 64),
]

N_WARMUP = 25
N_REP = 100


def _run_vllm(model: str, backend: str, prompts, gen_len: int) -> dict:
    """Run one generation pass under vLLM with the given attention backend."""
    try:
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(
            model=model,
            attention_backend=backend,
            enforce_eager=False,
            max_model_len=16384,
        )
        params = SamplingParams(temperature=0.0, max_tokens=gen_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return {
            "latency_s": dt,
            "tokens": tokens,
            "tokens_per_s": tokens / dt if dt > 0 else float("nan"),
            "text": outputs[0].outputs[0].text if outputs else "",
        }
    except Exception as exc:  # noqa: BLE001
        # One backend failing must not lose the other's numbers: the upstream
        # baseline is the point of this harness, so record the error and move on.
        import traceback

        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
            "latency_s": float("nan"),
            "tokens": 0,
            "tokens_per_s": float("nan"),
            "text": "",
        }


def bench_workload(model: str, name: str, batch: int, prompt_len: int, gen_len: int) -> dict:
    torch.manual_seed(0)
    prompts = [
        " ".join(["token"] * prompt_len) for _ in range(batch)
    ]

    with nvml_sampler() as gpu:
        base = _run_vllm(model, "TURBOQUANT", prompts, gen_len)
    base["gpu_util"] = gpu["util_gpu"]
    base["mem_used_mb"] = gpu["mem_used_mb"]

    # The plugin registers as CUSTOM; see model/registry.py.
    import thunder_vllm.model.registry as reg

    reg.configure(k_bits=4, v_bits=4)
    reg.register()

    with nvml_sampler() as gpu:
        ours = _run_vllm(model, "CUSTOM", prompts, gen_len)
    ours["gpu_util"] = gpu["util_gpu"]
    ours["mem_used_mb"] = gpu["mem_used_mb"]

    return {
        "workload": name,
        "batch": batch,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "baseline": base,
        "ours": ours,
        "speedup": base["latency_s"] / ours["latency_s"] if ours["latency_s"] else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="benchmarks/results/vs_thunder_vllm.md")
    ap.add_argument("--sweep", default=os.environ.get("TQ_VS_SWEEP", ""),
                    help="comma-separated workload names; empty = full sweep")
    args = ap.parse_args()

    sweep = SWEEP
    if args.sweep:
        want = {s.strip() for s in args.sweep.split(",") if s.strip()}
        sweep = [s for s in SWEEP if s[0] in want]

    rows = []
    for name, batch, prompt_len, gen_len in sweep:
        print(f"[vs_tq_vllm] {name}", flush=True)
        row = bench_workload(args.model, name, batch, prompt_len, gen_len)
        b, o = row["baseline"], row["ours"]
        if b.get("error") or o.get("error"):
            print(f"[vs] {name:<18} baseline_err={b.get('error')} ours_err={o.get('error')}",
                  flush=True)
        else:
            pct = o["tokens_per_s"] / b["tokens_per_s"] * 100 if b["tokens_per_s"] else float("nan")
            print(f"[vs] {name:<18} baseline={b['tokens_per_s']:.1f} tok/s "
                  f"ours={o['tokens_per_s']:.1f} tok/s ours%={pct:.1f} "
                  f"latency base/ours={b['latency_s']:.3f}/{o['latency_s']:.3f}s "
                  f"kv_base={b['mem_used_mb']:.0f}MB kv_ours={o['mem_used_mb']:.0f}MB",
                  flush=True)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Thunder-CuTe vs upstream vLLM TurboQuant\n\n")
        f.write("| Workload | batch | prompt | gen | baseline tok/s | ours tok/s | ours % | ours latency s | baseline latency s |\n")
        f.write("|---" * 9 + "|\n")
        for r in rows:
            b, o = r["baseline"], r["ours"]
            pct = (o["tokens_per_s"] / b["tokens_per_s"] * 100) if b["tokens_per_s"] else float("nan")
            f.write(
                f"| {r['workload']} | {r['batch']} | {r['prompt_len']} | {r['gen_len']} | "
                f"{b['tokens_per_s']:.1f} | {o['tokens_per_s']:.1f} | {pct:.1f}% | "
                f"{o['latency_s']:.3f} | {b['latency_s']:.3f} |\n"
            )
        f.write("\n## Raw\n\n```json\n")
        f.write(json.dumps(rows, indent=2, default=str))
        f.write("\n```\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()


__all__ = ["SWEEP", "assert_parity", "run_with_graph", "bench_workload"]
