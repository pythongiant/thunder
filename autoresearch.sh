#!/usr/bin/env bash
# Autoresearch entrypoint: kernel latency at the shapes the engine runs, on
# Modal B200 (no local CUDA).
#
# Primary metric: decode_b16_32k_ms — batch 16 at 32k context, the serving shape
# and the one with the most headroom left (the frozen FA4 matrix has ours ~70x
# off FA4 there, against ~2x at B=1). Secondaries cover the B=1 decode shapes
# and prefill so a change that trades one for another is visible.
set -euo pipefail
cd "$(dirname "$0")"

# Fast pre-check (~1s): a syntax error fails here, not after a GPU round trip.
python3 -m compileall -q thunder_vllm >/dev/null

log=$(mktemp)
if ! modal run ci/modal_app.py --mode loop >"$log" 2>&1; then
    tail -n 60 "$log" >&2
    exit 1
fi

grep -E '^METRIC ' "$log" | sort

if ! grep -q '^METRIC decode_b16_32k_ms=' "$log"; then
    tail -n 60 "$log" >&2
    echo "autoresearch.sh: primary metric decode_b16_32k_ms missing" >&2
    exit 1
fi
