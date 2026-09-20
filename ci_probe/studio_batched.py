"""Run the P3 batched worker (from modal_probe_batched.py) directly on a studio.

Reuses the exact worker source so studio and Modal runs are the same measurement.
Usage: python ci_probe/studio_batched.py "<pairs>" "<cfg,cfg>"
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ENV_OVER = {
    "fp16": {"THUNDER_8B_INDIRECT": "0", "THUNDER_STORE3": "0"},
    "ours-eager": {"THUNDER_8B_INDIRECT": "1", "THUNDER_STORE3": "1"},
    "ours-graph": {"THUNDER_8B_INDIRECT": "0", "THUNDER_STORE3": "1"},
}


def _consts(path: str) -> dict:
    out = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                try:
                    out[tgt.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def main() -> None:
    pairs = sys.argv[1] if len(sys.argv) > 1 else "1:4096,16:4096,1:16384,16:16384"
    cfgs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fp16", "ours-eager"]
    worker = _consts(os.path.join(HERE, "modal_probe_batched.py"))["_WORKER"]
    # Write the worker next to the project root, NOT /tmp: sys.path[0] becomes the
    # script's directory, and a vLLM source clone under /tmp would shadow the
    # installed package as a namespace package ("cannot import name 'LLM' from
    # 'vllm' (unknown location)").
    wpath = os.path.join(ROOT, "_pb_worker.py")
    with open(wpath, "w") as f:
        f.write(worker)
    for cfg in cfgs:
        print(f"\n{'=' * 70}\n== {cfg}\n{'=' * 70}", flush=True)
        env = dict(os.environ)
        env.update({"PYTHONPATH": ROOT, "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                    "PB_CFG": cfg, "PB_PAIRS": pairs})
        env.update(ENV_OVER[cfg])
        p = subprocess.Popen([sys.executable, wpath], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             cwd=ROOT, env=env)
        try:
            for ln in p.stdout:
                if ln.startswith("[PB") or "Error" in ln or "illegal" in ln:
                    print(ln, end="", flush=True)
            p.wait(timeout=3000)
        except Exception:
            p.kill()
            print(f"[driver] {cfg} TIMED OUT", flush=True)
        print(f"[driver] {cfg} rc={p.returncode}", flush=True)


if __name__ == "__main__":
    main()
