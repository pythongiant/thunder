"""Vendored CUTLASS/FA tcgen05 helpers (Apache-2.0).

``mma_sm100_desc`` builds tcgen05 instruction and SMEM descriptors;
``blackwell_helpers`` issues ``tcgen05.mma`` with explicit TMEM offsets. Both are
copied from Dao-AILab/flash-attention because CUTLASS-DSL 4.7.1's high-level
``cute.gemm`` cannot drive a tcgen05 SMEMxSMEM MMA (see
``thunder_vllm/attention/cute_kernel_tcgen05.py``).

Nothing is imported eagerly here: both modules depend on ``cutlass``, and this
package must stay importable on CPU-only hosts.
"""

__all__ = ["blackwell_helpers", "mma_sm100_desc"]
