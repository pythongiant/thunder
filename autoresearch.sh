#!/usr/bin/env bash
# Autoresearch entrypoint: OUR KERNEL vs the upstream backbone (FA4), WITHOUT vLLM.
#
# Why kernel-only: what we publish is a faster *kernel*, and the served e2e path is
# confounded (eager vs graphs, multiprocessing on/off, a capture path that is
# broken). Here both kernels see identical tensors and identical CUDA-event timing
# (25 warmup + 50 timed reps) on the engine's Qwen3-8B geometry (head_dim 128,
# 32 Q heads / 8 KV heads = GQA 4:1, causal) at the engine's own tile, split-K and
# GQA-packing policy -- i.e. the configuration we ship.
#
# Primary metric: speedup_vs_fa4 = geomean(fa4_ms / ours_ms) over the cells
# (higher is better; >1.0 means our kernel beats the kernel upstream TurboQuant
# runs on).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m compileall -q thunder_vllm >/dev/null
# <shape>|<splits>|<gqa> with the rest left to the engine policy.
CELLS="decode-b16-32k|policy|policy;decode-b16-4k|policy|policy;decode-long|policy|policy;prefill|policy|policy;prefill-16k|policy|policy"
log=$(mktemp)
if ! modal run ci/modal_app.py --mode grid --grid "$CELLS" >"$log" 2>&1; then
  echo "=== benchmark command failed ==="
  tail -n 100 "$log"
  exit 1
fi
grep -E '^METRIC ' "$log" | sort
cat <<'BASELINE'
[baseline] upstream TurboQuant (the upstreamed TURBOQUANT backend) runs on FA4's
[baseline] SM100 forward, so `speedup_vs_fa4` compares our kernel with exactly
[baseline] that kernel on identical tensors and identical timing, with no vLLM.
BASELINE
if ! grep -q '^METRIC speedup_vs_fa4=' "$log"; then
  echo "=== primary metric missing; log tail ==="
  tail -n 100 "$log"
  echo "autoresearch.sh: primary metric speedup_vs_fa4 missing"
  exit 1
fi
