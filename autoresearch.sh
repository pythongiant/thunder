#!/usr/bin/env bash
# Autoresearch entrypoint: OUR KERNEL vs the upstream backbone (FA4), WITHOUT vLLM.
#
# Why kernel-only: the thing we publish is a faster *kernel*, and the served e2e
# path is confounded (eager vs graphs, multiprocessing on/off, broken capture).
# Here both sides get identical q/k/v, identical CUDA-event timing (25 warmup +
# 100 timed reps, median reported), on the engine's Qwen3-8B shapes
# (head_dim 128, GQA 4:1, causal) -- i.e. our kernel against the kernel upstream
# TurboQuant runs on. No engine, no scheduler, no sampler.
#
# Primary metric: speedup_vs_fa4 = geomean(fa4_ms / ours_ms) over the selected
# cells (higher is better; >1.0 means we beat upstream's kernel).
set -euo pipefail
cd "$(dirname "$0")"
# Fast pre-check (~1s): a syntax error fails here, not after a GPU round trip.
python3 -m compileall -q thunder_vllm >/dev/null
CELLS="hd128-gqa4-causal-decode-B16-S32768-sp16,hd128-gqa4-causal-decode-B16-S16384-sp16,hd128-gqa4-causal-decode-B16-S4096-sp16,hd128-gqa4-causal-decode-B1-S32768-sp64,hd128-gqa4-causal-prefill-B1-S4096-sp1"
log=$(mktemp)
if ! modal run ci/modal_app.py --mode shell \
      --shell-cmd "cd /opt/thunder_vllm && python -m benchmarks.fa4_matrix --cells '$CELLS' --metrics" \
      >"$log" 2>&1; then
  # STDOUT, not stderr: the run capture reads stdout, so a stderr-only failure
  # produced an empty log and an undiagnosable exit 1.
  echo "=== benchmark command failed ==="
  tail -n 100 "$log"
  exit 1
fi
grep -E '^METRIC ' "$log" | sort
# Provenance for the served comparison (docs/BENCHMARKS.md, AGENTS.md). These are
# NOT this metric: they are what upstream TurboQuant serves, for orientation.
cat <<'BASELINE'
[baseline] upstream TurboQuant (FA4-based KV path), served, batch 1 (vLLM 0.25.1):
[baseline]   @4096  109.5 tok/s  ITL 7.03 ms  TTFT 74.3 ms
[baseline]   @32768  18.2 tok/s  ITL 12.23 ms  TTFT 1380.2 ms
[baseline] above (METRIC) is a KERNEL launch, not a served step.
BASELINE
if ! grep -q '^METRIC speedup_vs_fa4=' "$log"; then
  echo "=== primary metric missing; log tail ==="
  tail -n 100 "$log"
  echo "autoresearch.sh: primary metric speedup_vs_fa4 missing"
  exit 1
fi
