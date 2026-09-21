# Runbook

Single operational reference for GPU work. `learnings.md` remains local-only; this file is the tracked procedure.

## Validated stack

- CUDA 12.8.1, torch 2.8.0 cu128, `nvidia-cutlass-dsl==4.7.1`, Triton.
- vLLM pinned at `dffbb714e4e8e4b95ccc888df98d47c7d2cef78d`.
- Primary model: `Qwen/Qwen3-8B`, 32 Q heads / 8 KV heads / head_dim 128, GQA 4:1.
- Backend selection: `attention_config={"backend": "CUSTOM"}`; upstream `TURBOQUANT` is intentionally left intact.

## Lightning B200 studio recipe

`thunder-studio` is configured for a 1×B200 machine. The SDK may reject the machine name while REST accepts the configured machine:

- Empty-body REST `/start` provisions the studio’s configured machine.
- If the configured machine was reset to CPU, use `Studio.switch_machine("nb-b200-1gpu-20vcpu-224gb")`.
- Install vLLM with `VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation` from the pinned commit.
- After vLLM install, upgrade studio `scipy`/`scikit-learn`: the install pulls NumPy 2.x and breaks old SciPy with `cannot import name 'Inf' from 'numpy'`.
- Do not run the worker from `/tmp` if a vLLM source clone lives at `/tmp/vllm`; `/tmp` on `sys.path` shadows the installed package.

## Mandatory measurement hygiene

Before every e2e run:

1. Kill leftover engine processes: `pkill -9 -f EngineCore`.
2. Verify no compute processes remain with `nvidia-smi --query-compute-apps`.
3. Verify enough free memory for the planned workload.
4. Run one config at a time; never overlap fp16/ours workers.
5. Record pairs, configs, env flags, commit hash, and machine.

Known contamination trap: a killed `_pb_worker` can leave `VLLM::EngineCore` alive holding most of the GPU. Later runs then start with little free memory and produce absurd timings. Those numbers are invalid; do not optimize against them.

## Canonical checks

- Correctness: `PYTHONPATH=. python3 ci_probe/correctness_check.py`
- Kernel A/B: `PYTHONPATH=. python3 ci_probe/kernel_bench.py`
- Stage probe: `PYTHONPATH=. python3 ci_probe/kernel_stage_probe.py`
- Batched e2e: `PYTHONPATH=. python3 ci_probe/studio_batched.py "<pairs>" "<cfgs>"`
- CPU tests: `python -m pytest tests/ -q`

## Current access reality

- Modal workspace may be disabled by spend limit.
- Some GPU SKUs may exceed account limits; only use provisioned/entitled machines.
- If a B200 start fails with insufficient balance, stop the studio and do CPU-safe work instead.
