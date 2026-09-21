"""thunder-vllm: fused TurboQuant attention backend for vLLM.

This package is import-safe on CPU-only machines: nothing in the top-level
import path pulls in CUDA, CUTLASS/CuTeDSL, or vLLM. Those are imported lazily
by the modules that need them (``thunder_vllm.attention.*``).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
