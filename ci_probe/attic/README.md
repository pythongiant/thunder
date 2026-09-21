# Attic

Preserved one-off analysis scripts. Nothing here is canonical and nothing here
runs in CI.

Canonical probes live in `ci_probe/`:

- `correctness_check.py`
- `kernel_bench.py`
- `kernel_stage_probe.py`
- `cpasync_probe.py`
- `batched_worker.py` + `studio_batched.py`

These attic scripts analyzed a single saved store tensor during the V-stride
root-cause work. They are kept so the reasoning is not lost, but they must not
be treated as supported tooling.
