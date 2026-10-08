"""Watch what vLLM's CUDA-graph capture does when this backend is in the graph path.

The capture either fails fast with a CUDA error (readable from the worker's stderr)
or HANGS -- and a hang is the hard case: the e2e worker buffers the engine's output,
so nothing is visible until the process is killed. This runs the engine itself,
unbuffered and at DEBUG verbosity, prints its log as it arrives, and samples the
GPU's utilisation when the log goes quiet:

    busy  -> the capture is spinning (a compile / allocator retry loop)
    idle  -> it is waiting on something that will never arrive (a deadlock)

That distinction decides the fix. It also dumps `/proc/<pid>/wchan` for the engine
process, i.e. what the kernel says it is blocked on.

Run (inside the container):  CTX=4096 WATCH_SECONDS=150 python ci_probe/probe_capture_watch.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

CTX = int(os.environ.get("CTX", "4096"))
MAXLEN = int(os.environ.get("MAXLEN", "0")) or (CTX + 64)
WATCH = float(os.environ.get("WATCH_SECONDS", "150"))
K_BITS = os.environ.get("K_BITS", "3")
V_BITS = os.environ.get("V_BITS", "4")

WORKER = f'''
import torch
from vllm import LLM, SamplingParams

from thunder_vllm.model.registry import configure, register

configure(k_bits={K_BITS}, v_bits={V_BITS})
register()

print("[watch] building", flush=True)
llm = LLM(model="Qwen/Qwen3-8B", max_model_len={MAXLEN}, dtype="float16",
          gpu_memory_utilization=0.85, enable_prefix_caching=False,
          attention_config={{"backend": "CUSTOM"}},
          compilation_config={{"cudagraph_mode": "FULL_AND_PIECEWISE",
                              "cudagraph_capture_sizes": [1],
                              "max_cudagraph_capture_size": 1}})
print("[watch] built -- capture phase is over", flush=True)
prompt = " ".join(["token"] * {CTX})
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))
print("[watch] served:", out[0].outputs[0].text[:24], flush=True)
'''


def _proc_state() -> str:
    """What the engine process is blocked on, per the kernel."""
    try:
        pids = subprocess.run(["pgrep", "-f", "EngineCore"], capture_output=True,
                              text=True).stdout.split()
    except Exception:  # noqa: BLE001
        return "pgrep failed"
    if not pids:
        return "no EngineCore process"
    out = []
    for pid in pids[:2]:
        try:
            with open(f"/proc/{pid}/wchan") as fh:
                wchan = fh.read().strip()
        except Exception:  # noqa: BLE001
            wchan = "?"
        out.append(f"pid={pid} wchan={wchan}")
    return "; ".join(out)


def main() -> int:
    path = "/tmp/watch_worker.py"
    with open(path, "w") as fh:
        fh.write(WORKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = "/opt/thunder_vllm"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    proc = subprocess.Popen(
        [sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd="/opt/thunder_vllm", env=env,
    )
    deadline = time.time() + WATCH
    quiet_since = time.time()
    while proc.poll() is None and time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            print(f"E| {line.rstrip()}", flush=True)
            quiet_since = time.time()
            continue
        if time.time() - quiet_since > 15:
            quiet_since = time.time()
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader"], capture_output=True, text=True,
            )
            print(f"[watch] quiet; gpu={smi.stdout.strip()!r} | {_proc_state()}",
                  flush=True)
    if proc.poll() is None:
        print(f"[watch] STILL RUNNING after {WATCH:.0f}s -> hung", flush=True)
        print(f"[watch] state: {_proc_state()}", flush=True)
        proc.kill()
        return 3
    print(f"[watch] finished rc={proc.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
