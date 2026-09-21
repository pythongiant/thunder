"""Attention package: backend, metadata, layout, gather, scratch, CuTe kernel.

Submodules that need vLLM/CUTLASS/CUDA are imported lazily by their consumers;
``import thunder_vllm.attention`` stays cheap and CPU-safe.
"""

__all__ = [
    "backend",
    "cache_layout",
    "cute_kernel",
    "metadata",
    "paged_kv",
    "scratch",
]
