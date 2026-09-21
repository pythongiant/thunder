"""Run the tracked P3 batched worker directly on a studio.

Usage: python ci_probe/studio_batched.py "<pairs>" "<cfg,cfg>"
"""
from __future__ import annotations

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


def main() -> None:
    pairs = sys.argv[1] if len(sys.argv) > 1 else "1:4096,16:4096,1:16384,16:16384"
    cfgs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fp16", "ours-eager"]
    # Run the tracked worker in place. Do not copy it to /tmp: sys.path[0]
    # becomes the script's directory, and a vLLM source clone under /tmp would
    # shadow the installed package as a namespace package.
    wpath = os.path.join(HERE, "batched_worker.py")
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
