#!/usr/bin/env bash
# Run every benchmark and materialise benchmarks/results/*.md.
#
# Correctness gate: no speed number is reported if the correctness run fails.
#
# Env:
#   TURBOQUANT_KERNEL_ENABLE=1   required to actually launch the CuTe kernel
#   TURBOQUANT_MODEL             HF model id for the vs-vLLM sweep
#   PROFILE=1                    also run ncu and dump CSVs

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
RESULTS="$HERE/results"
mkdir -p "$RESULTS"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TURBOQUANT_KERNEL_ENABLE="${TURBOQUANT_KERNEL_ENABLE:-0}"
MODEL="${TURBOQUANT_MODEL:-Qwen/Qwen3-4B}"

echo "== correctness gate =="
python -m pytest "$ROOT/tests" -q

echo "== vs FA4 (kernel level) =="
python -m benchmarks.bench_vs_fa4 --out "$RESULTS/vs_fa4.md"

echo "== vs upstream vLLM TurboQuant =="
python -m benchmarks.bench_vs_turboquant_vllm \
  --model "$MODEL" --out "$RESULTS/vs_turboquant_vllm.md"

echo "== ablation =="
python -m pytest "$ROOT/tests/test_ablation.py" -s

if [[ "${PROFILE:-0}" == "1" ]]; then
  echo "== ncu profiling =="
  python -m benchmarks.profile --out-dir "$RESULTS"
  python -m benchmarks.ncu_summary --results-dir "$RESULTS"
fi

echo "results in $RESULTS"
