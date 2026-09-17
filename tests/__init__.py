"""TurboQuant-CuTe test suite.

CPU tests exercise the cache layout, packing, codebooks, quantizer round-trip
and paged-KV gather. Kernel and CUDA-graph tests are SM100-only and are skipped
elsewhere (see ``tests/conftest.py``).
"""
