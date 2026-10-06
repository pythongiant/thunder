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

# The baseline every run is reported against (AGENTS.md): upstream TurboQuant KV,
# measured on its own pin. The loop measures kernel launches, which are not the
# same quantity as upstream's served steps -- so both are printed, labelled.
cat <<'BASELINE'

[baseline] upstream turboquant_3bit_nc (vLLM 0.25.1, no plugin), served, batch 1:
[baseline]   @4096   109.5 tok/s   ITL 7.03 ms   TTFT 74.3 ms   KV 3.4 MiB
[baseline]   @32768   18.2 tok/s   ITL 12.23 ms  TTFT 1380.2 ms  KV 27.0 MiB
[baseline] ours, served, batch 1 (ci/modal_app.py --mode e2e): request 7.3-7.5 tok/s,
[baseline]   decode 58 tok/s (ITL 17.4 ms), TTFT 1.9-4.1 s; @32768 blocked.
[baseline] ours above (METRIC) is an attention launch, not a served step.
BASELINE

if ! grep -q '^METRIC decode_b16_32k_ms=' "$log"; then
    tail -n 60 "$log" >&2
    echo "autoresearch.sh: primary metric decode_b16_32k_ms missing" >&2
    exit 1
fi
