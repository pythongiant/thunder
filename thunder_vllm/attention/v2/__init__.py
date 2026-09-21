# v2 pipeline: FA4-derived execution skeleton + TurboQuant data path.
#
# ``fa4/`` is a verbatim vendor of the flash-attn-4 (flash_attn.cute) package,
# version 4.0.0b31 (BSD-1-Clause, Dao-AILab/flash-attention), with exactly one
# mechanical rewrite: ``flash_attn.cute`` -> ``thunder_vllm.attention.v2.fa4``
# in all imports. It is frozen on purpose -- upstream betas break APIs (b31
# already changed FlashAttentionForwardSm100.__call__ to a raw cute-tensor
# signature). External deps (cutlass, quack, torch) are NOT vendored.
#
# Validation ladder (docs/PIPELINE_V2.md): S2 proves the vendored, unmodified
# forward reproduces installed-FA4 numbers via TQ_FA4_MODULE=vendored in
# benchmarks/fa4_matrix.py; S3 swaps the KV load producer for the TurboQuant
# packed path. Do not hand-edit fa4/ except through the producer swap.
