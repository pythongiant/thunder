"""Per-layer online TurboQuant quantizer (the KV-cache write path).

Pipeline for a token's K (V uses a separate codebook):

    k  --(R)-->  k_rot  --(norm)-->  k_norm  --(Lloyd-Max)-->  idx  --(pack)--> bytes

and the norm is stored separately so the attention kernel can apply it after
the QK MMA (``S *= k_norm``) instead of baking a per-token scale into the LUT.
For V the norm is folded into the softmax probabilities before the PV MMA.

Everything here is device-agnostic torch and runs on CPU, which is what the
CPU test-suite exercises. The GPU cache-write path lives in
``attention/cache_layout.py::reshape_and_cache_*`` and reproduces exactly this
arithmetic inside a Triton kernel (rotation included, so no rotated fp16 K/V is
ever written to HBM).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thunder_vllm.quant.hadamard import HadamardRotation
from thunder_vllm.quant.lloyd_max import Codebook, build_lut
from thunder_vllm.quant.packing import pack_indices, unpack_indices


@dataclass
class QuantizedKV:
    """Result of quantizing one token chunk."""

    k_packed: torch.Tensor  # (N, Hk, k_packed_bytes) uint8
    v_packed: torch.Tensor  # (N, Hk, v_packed_bytes) uint8
    k_norm: torch.Tensor  # (N, Hk) fp16
    v_norm: torch.Tensor  # (N, Hk) fp16


class ThunderQuantizer:
    """Owns the rotation and both codebooks for one layer.

    Args:
        head_dim: Model head dimension (unpadded).
        k_bits / v_bits: K/V codebook bit widths.
        head_dim_padded: LUT row width (rounded up to a multiple of 16).
        device / dtype: Where the rotation and LUTs live.
        random_signs: Passed through to :class:`HadamardRotation`; requires a
            per-layer ``seed`` to be meaningful.
        seed: Per-layer seed for the optional sign flips.
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int,
        v_bits: int,
        *,
        head_dim_padded: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
        random_signs: bool = False,
        seed: int = 0,
    ) -> None:
        if head_dim % 16 != 0 and head_dim_padded is None:
            raise ValueError(
                "head_dim must be a multiple of 16 or head_dim_padded must be given"
            )
        self.head_dim = int(head_dim)
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.head_dim_padded = int(head_dim_padded or ((head_dim + 15) // 16) * 16)
        self.device = torch.device(device)
        self.dtype = dtype

        self.rotation = HadamardRotation(
            self.head_dim, dtype=torch.float32, device=self.device,
            random_signs=random_signs, seed=seed,
        )
        self.k_codebook: Codebook = self._make_codebook(self.k_bits, seed)
        self.v_codebook: Codebook = self._make_codebook(self.v_bits, seed + 1)
        self.k_lut = build_lut(self.k_codebook, self.head_dim_padded).to(
            device=self.device, dtype=self.dtype
        ).contiguous()
        self.v_lut = build_lut(self.v_codebook, self.head_dim_padded).to(
            device=self.device, dtype=self.dtype
        ).contiguous()

    def _make_codebook(self, bits: int, seed: int) -> Codebook:
        from thunder_vllm.quant.lloyd_max import build_lloyd_max_codebook

        return build_lloyd_max_codebook(bits, head_dim=self.head_dim, seed=seed,
                                        device=self.device)

    # ------------------------------------------------------------------ #
    # Quantize / dequantize
    # ------------------------------------------------------------------ #
    def _quantize_one(
        self, x: torch.Tensor, codebook: Codebook, bits: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate, normalise, quantize, pack. Returns (packed, norm, indices)."""
        x_f = x.to(torch.float32)
        rot = self.rotation.rotate(x_f)
        norm = torch.linalg.vector_norm(rot, dim=-1)
        safe = torch.where(norm > 0, norm, torch.ones_like(norm))
        unit = rot / safe.unsqueeze(-1)
        idx = codebook.quantize(unit)
        packed = pack_indices(idx, bits, self.head_dim)
        return packed, norm.to(self.dtype), idx

    def quantize_k(self, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``key`` of shape ``(..., head_dim)`` -> (packed, k_norm, indices)."""
        return self._quantize_one(key, self.k_codebook, self.k_bits)

    def quantize_v(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._quantize_one(value, self.v_codebook, self.v_bits)

    def quantize(self, key: torch.Tensor, value: torch.Tensor) -> QuantizedKV:
        """Quantize ``(N, Hk, D)`` key/value into the cache byte format."""
        k_packed, k_norm, _ = self.quantize_k(key)
        v_packed, v_norm, _ = self.quantize_v(value)
        return QuantizedKV(
            k_packed=k_packed,
            v_packed=v_packed,
            k_norm=k_norm,
            v_norm=v_norm,
        )

    def dequantize_k(self, packed: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Reference dequant: packed bytes + norm -> original-basis fp32 K.

        The kernel never does this (it consumes the LUT directly and applies the
        norm inside the softmax), but tests and the dequant-to-fp16 baseline
        need it.
        """
        rot = self.dequantize_k_rotated(packed, norm)
        return self.rotation.inverse(rot)

    def dequantize_v(self, packed: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        rot = self.dequantize_v_rotated(packed, norm)
        return self.rotation.inverse(rot)

    def dequantize_k_rotated(
        self, packed: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        """Rotated-basis dequant: ``norm * unit_hat(R k)``.

        This is what the attention kernel consumes. Because the Hadamard matrix
        is orthonormal, ``<q @ R, k @ R> == <q, k>``, so scores are exact in the
        rotated basis; V, however, comes out rotated, so the caller must apply
        ``R^T`` to the attention output (the plugin does this as the allowed
        output projection GEMM).
        """
        idx = unpack_indices(packed, self.k_bits, self.head_dim)
        unit = self.k_codebook.dequantize(idx).to(torch.float32)
        return unit * norm.to(torch.float32).unsqueeze(-1)

    def dequantize_v_rotated(
        self, packed: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        idx = unpack_indices(packed, self.v_bits, self.head_dim)
        unit = self.v_codebook.dequantize(idx).to(torch.float32)
        return unit * norm.to(torch.float32).unsqueeze(-1)

    def dequantize(self, kv: QuantizedKV) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dequantize_k(kv.k_packed, kv.k_norm), self.dequantize_v(
            kv.v_packed, kv.v_norm
        )
