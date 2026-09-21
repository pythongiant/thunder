"""TurboQuant KV-cache layout, allocation, and the cache-write path.

Allocation
----------
Two tensors per layer:

    kv_cache : uint8, (num_blocks, block_size, kv_slot_bytes)
    kv_scales: fp16,  (num_blocks, block_size, num_kv_heads, 2)

``kv_slot_bytes`` is one combined slot per (block, position):

    [ K codes: Hk * k_packed_bytes | V codes: Hk * v_packed_bytes ]

Zero-copy views ``k_codes()`` / ``v_codes()`` expose exactly the kernel
contract, and ``k_norm()`` / ``v_norm()`` slice the fp16 scales:

    packed K : uint8, (num_blocks, block_size, num_kv_heads, k_packed_bytes)
    packed V : uint8, (num_blocks, block_size, num_kv_heads, v_packed_bytes)
    k_norm   : fp16,  (num_blocks, block_size, num_kv_heads)
    v_norm   : fp16,  (num_blocks, block_size, num_kv_heads)

Keeping K and V in one allocation (rather than two) means the gather layer
issues one strided read per slot instead of two cache lines, and the block
table resolution is identical for both.

Slot resolution
---------------
``slot = block_id * block_size + offset`` with no extra indirection, so
``slot_mapping`` from the scheduler maps a token directly to a byte row. The
block table is only needed by the gather layer, which reads
``block_table[req, n_block]`` to find ``block_id``.
"""

from __future__ import annotations

import logging as _logging
import os as _os
from dataclasses import dataclass
from typing import NamedTuple

import torch

_log = _logging.getLogger("thunder_vllm.cache_layout")

from thunder_vllm.utils.logging import env_flag, get_logger

logger = get_logger("attention.cache_layout")

try:  # pragma: no cover - import guarded for CPU-only machines
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def packed_bytes(head_dim: int, bits: int) -> int:
    return (head_dim * bits + 7) // 8


class KVSlotView(NamedTuple):
    """Views onto an allocated cache, matching the kernel's tensor contract."""

    k_codes: torch.Tensor
    v_codes: torch.Tensor
    k_norm: torch.Tensor
    v_norm: torch.Tensor


@dataclass(frozen=True)
class ThunderCacheLayout:
    """Static description of the combined packed-KV slot.

    ``head_dim_padded`` rounds up to a multiple of 16: the packed bytes are
    computed against the true ``head_dim`` (compression ratio is unaffected),
    while the dequantized code tiles and the LUT rows are padded to
    ``head_dim_padded``.
    """

    num_kv_heads: int
    head_dim: int
    k_bits: int
    v_bits: int
    block_size: int = 16
    scale_dtype: torch.dtype = torch.float16
    head_dim_padded: int = 0

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.k_bits < 1 or self.k_bits > 8:
            raise ValueError(f"k_bits must be in [1, 8], got {self.k_bits}")
        if self.v_bits < 1 or self.v_bits > 8:
            raise ValueError(f"v_bits must be in [1, 8], got {self.v_bits}")
        padded = self.head_dim_padded or round_up(self.head_dim, 16)
        if padded < self.head_dim:
            raise ValueError("head_dim_padded must be >= head_dim")
        object.__setattr__(self, "head_dim_padded", padded)

    # -- sizes -----------------------------------------------------------
    @property
    def k_packed_bytes(self) -> int:
        return packed_bytes(self.head_dim, self.k_bits)

    @property
    def v_packed_bytes(self) -> int:
        return packed_bytes(self.head_dim, self.v_bits)

    @property
    def k_region_bytes(self) -> int:
        return self.num_kv_heads * self.k_packed_bytes

    @property
    def v_region_bytes(self) -> int:
        return self.num_kv_heads * self.v_packed_bytes

    @property
    def head_slot_bytes(self) -> int:
        """Bytes per (block, head, position) covering that head's K and V codes.

        This is the innermost dimension of the cache in vLLM's canonical layout
        ``(num_blocks, num_kv_heads, block_size, head_slot_bytes)``.
        """
        return self.k_packed_bytes + self.v_packed_bytes

    @property
    def kv_slot_bytes(self) -> int:
        """Bytes per (block, position) covering every head's K and V codes."""
        return self.k_region_bytes + self.v_region_bytes

    @property
    def n_scales(self) -> int:
        return self.num_kv_heads * 2

    # -- shapes ----------------------------------------------------------
    def get_kv_cache_shape(self, num_blocks: int) -> tuple[int, ...]:
        """vLLM canonical KV-cache layout, combined K+V, no leading K/V dim.

        ``(num_blocks, num_kv_heads, block_size, head_slot_bytes)``

        V1 views the raw allocation into exactly this shape, so declaring the old
        3-D byte-slot form made the engine hand us a rank-4 tensor and
        ``k_codes`` fail with "too many values to unpack (expected 3)" during the
        KV-cache warm-up.
        """
        return (num_blocks, self.num_kv_heads, self.block_size, self.head_slot_bytes)

    def get_scales_shape(self, num_blocks: int) -> tuple[int, ...]:
        return (num_blocks, self.num_kv_heads, self.block_size, 2)

    def get_kernel_shapes(self, num_blocks: int) -> dict[str, tuple[int, ...]]:
        return {
            "k_packed": (num_blocks, self.block_size, self.num_kv_heads, self.k_packed_bytes),
            "v_packed": (num_blocks, self.block_size, self.num_kv_heads, self.v_packed_bytes),
            "k_norm": (num_blocks, self.block_size, self.num_kv_heads),
            "v_norm": (num_blocks, self.block_size, self.num_kv_heads),
        }

    # -- slot arithmetic -------------------------------------------------
    def slot_index(self, block_id: int, offset: int) -> int:
        return block_id * self.block_size + offset

    def split_slot(self, slot: int) -> tuple[int, int]:
        return divmod(slot, self.block_size)

    # -- views -----------------------------------------------------------
    def views(self, kv_cache: torch.Tensor, kv_scales: torch.Tensor) -> KVSlotView:
        """Zero-copy views of a packed cache in the kernel's layout."""
        nb, hk, bs, slot = kv_cache.shape
        if slot != self.head_slot_bytes:
            raise ValueError(
                f"kv_cache head slot {slot} != layout head slot {self.head_slot_bytes}"
            )
        return KVSlotView(
            k_codes=self.k_codes(kv_cache),
            v_codes=self.v_codes(kv_cache),
            k_norm=self.k_norm(kv_scales),
            v_norm=self.v_norm(kv_scales),
        )

    def k_norm(self, kv_scales: torch.Tensor) -> torch.Tensor:
        """(nb, bs, Hk) fp16 K norms, kernel order."""
        return kv_scales[..., 0].permute(0, 2, 1)

    def v_norm(self, kv_scales: torch.Tensor) -> torch.Tensor:
        """(nb, bs, Hk) fp16 V norms, kernel order."""
        return kv_scales[..., 1].permute(0, 2, 1)

    def k_codes(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """K codes in the kernel's ``(nb, bs, Hk, k_packed_bytes)`` order.

        The cache is ``(nb, Hk, bs, head_slot)``; this is the single transpose
        boundary between vLLM's convention and the kernels'.
        """
        if env_flag("THUNDER_DEBUG_LAYOUT"):
            _log.info(
                "k_codes received shape=%s dtype=%s stride=%s contiguous=%s",
                tuple(kv_cache.shape), kv_cache.dtype, tuple(kv_cache.stride()),
                kv_cache.is_contiguous(),
            )
        return kv_cache[..., : self.k_packed_bytes].permute(0, 2, 1, 3)

    def v_codes(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """V codes in the kernel's ``(nb, bs, Hk, v_packed_bytes)`` order."""
        return kv_cache[..., self.k_packed_bytes :].permute(0, 2, 1, 3)


def allocate_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    K_BITS: int,
    V_BITS: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate ``(kv_cache, kv_scales)`` for the combined TurboQuant slot.

    ``kv_cache`` is always ``uint8``; ``dtype`` controls the fp16 norm storage.
    """
    layout = ThunderCacheLayout(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        k_bits=K_BITS,
        v_bits=V_BITS,
        block_size=block_size,
        scale_dtype=dtype,
    )
    kv_cache = torch.zeros(
        layout.get_kv_cache_shape(num_blocks), dtype=torch.uint8, device=device
    )
    kv_scales = torch.zeros(
        layout.get_scales_shape(num_blocks), dtype=dtype, device=device
    )
    logger.info(
        "allocated TurboQuant cache: blocks=%d block_size=%d Hk=%d D=%d "
        "K_BITS=%d V_BITS=%d slot_bytes=%d codes=%.1f MiB norms=%.1f MiB",
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        K_BITS,
        V_BITS,
        layout.kv_slot_bytes,
        kv_cache.numel() / 2**20,
        kv_scales.numel() * kv_scales.element_size() / 2**20,
    )
    return kv_cache, kv_scales


# ---------------------------------------------------------------------------
# Cache write (reshape_and_cache)
# ---------------------------------------------------------------------------


def reshape_and_cache_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scales: torch.Tensor,
    quantizer,
    layout: ThunderCacheLayout,
) -> None:
    """Reference (pure-torch) cache write. Also the CPU test path.

    ``key`` / ``value`` are ``(N, Hk, head_dim)`` in the *original* basis,
    fp16/bf16. Rotation happens here (``quantizer``), so no rotated fp16 K/V is
    ever stored -- only packed codes and fp16 norms are written.
    """
    n = key.shape[0]
    if n == 0:
        return
    block_size = layout.block_size

    kv = quantizer.quantize(key, value)

    slot = slot_mapping.to(torch.int64)
    # PAD_SLOT_ID (-1) marks a position that must not be written. Negative
    # indices would wrap in torch and corrupt the last block.
    keep = slot >= 0
    if not bool(keep.any()):
        return
    slot = slot[keep]
    block_id = torch.div(slot, block_size, rounding_mode="floor")
    offset = slot % block_size

    k_view = layout.k_codes(kv_cache)
    v_view = layout.v_codes(kv_cache)
    # (N, Hk, bytes) -> (num_blocks, block_size, Hk, bytes) scatter, through the
    # permuted views (one write per (block, position), all heads at once).
    k_view[block_id, offset] = kv.k_packed[keep]
    v_view[block_id, offset] = kv.v_packed[keep]
    kv_scales[block_id, :, offset, 0] = kv.k_norm[keep]
    kv_scales[block_id, :, offset, 1] = kv.v_norm[keep]


if _HAS_TRITON:

    @triton.jit
    def _reshape_and_cache_kernel(
        key_ptr,
        value_ptr,
        slot_ptr,
        cache_ptr,
        scales_ptr,
        rot_ptr,           # (D, D) fp16 rotation matrix
        k_bounds_ptr,      # (2**k_bits - 1,) fp32 ascending boundaries
        v_bounds_ptr,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_vd,
        n_rows,
        stride_cache_block,
        stride_cache_head,
        stride_cache_pos,
        stride_scales_block,
        stride_scales_pos,
        stride_scales_head,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        k_packed_bytes: tl.constexpr,
        v_packed_bytes: tl.constexpr,
        k_region_bytes: tl.constexpr,
        kv_slot_bytes: tl.constexpr,
        K_BITS: tl.constexpr,
        V_BITS: tl.constexpr,
        K_LEVELS: tl.constexpr,
        V_LEVELS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        """One program handles ``ROWS`` tokens for all ``num_kv_heads``.

        Rotation is a dot with the constant rotation matrix; quantization walks
        the boundary table with a branch-free binary search (``O(log levels)``);
        packing reproduces ``quant/packing.py`` little-endian layout inline for
        the widths the kernel supports (2/3/4/8).
        """
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        row_mask = rows < n_rows

        d = tl.arange(0, head_dim)
        # (D, D) rotation, loaded once per program.
        rot = tl.load(rot_ptr + d[:, None] * head_dim + d[None, :])

        for h in tl.static_range(num_kv_heads):
            base_k = rows * stride_kn + h * stride_kh
            base_v = rows * stride_vn + h * stride_vh
            k = tl.load(key_ptr + base_k[:, None] + d[None, :] * 1, mask=row_mask[:, None], other=0.0)
            v = tl.load(value_ptr + base_v[:, None] + d[None, :] * stride_vd, mask=row_mask[:, None], other=0.0)
            # Match Codebook.quantize exactly: the reference rotates in fp32
            # (``x_f @ matrix.to(fp32)``) and normalizes with
            # ``safe = where(norm > 0, norm, 1)`` (no epsilon). An fp16 tl.dot
            # with an fp16-cast rotation matrix was the divergence source.
            k32 = k.to(tl.float32)
            v32 = v.to(tl.float32)
            k_r = tl.dot(k32, rot, out_dtype=tl.float32, input_precision="ieee")
            v_r = tl.dot(v32, rot, out_dtype=tl.float32, input_precision="ieee")
            k_norm = tl.sqrt(tl.sum(k_r * k_r, axis=1))
            v_norm = tl.sqrt(tl.sum(v_r * v_r, axis=1))
            k_safe = tl.where(k_norm > 0.0, k_norm, 1.0)
            v_safe = tl.where(v_norm > 0.0, v_norm, 1.0)
            k_u = k_r / k_safe[:, None]
            v_u = v_r / v_safe[:, None]

            k_idx = _searchsorted(k_bounds_ptr, k_u, K_LEVELS, head_dim)
            v_idx = _searchsorted(v_bounds_ptr, v_u, V_LEVELS, head_dim)

            if K_BITS == 3:
                k_b0, k_b1, k_b2 = _pack3(k_idx, ROWS, head_dim)
            else:
                k_bytes = _pack(k_idx, ROWS, K_BITS, head_dim, k_packed_bytes)
            if V_BITS == 3:
                v_b0, v_b1, v_b2 = _pack3(v_idx, ROWS, head_dim)
            else:
                v_bytes = _pack(v_idx, ROWS, V_BITS, head_dim, v_packed_bytes)

            slot = tl.load(slot_ptr + rows, mask=row_mask, other=0)
            pos = slot % BLOCK_SIZE
            blk = slot // BLOCK_SIZE
            # vLLM's KV-cache warm-up hands an all-PAD slot map (PAD_SLOT_ID ==
            # -1): a negative slot makes blk negative and the store address lands
            # before the cache -> illegal memory access. Skip those rows.
            slot_ok = row_mask & (slot >= 0)
            out = cache_ptr + blk[:, None] * stride_cache_block + h * stride_cache_head + pos[:, None] * stride_cache_pos
            # K codes: little-endian column c goes to byte c*K_BITS//8.
            if K_BITS == 3:
                g3 = (3 * tl.arange(0, head_dim // 8))[None, :]
                tl.store(out + g3, k_b0, mask=slot_ok[:, None])
                tl.store(out + g3 + 1, k_b1, mask=slot_ok[:, None])
                tl.store(out + g3 + 2, k_b2, mask=slot_ok[:, None])
            else:
                kc = tl.arange(0, k_packed_bytes)
                tl.store(out + kc[None, :], k_bytes, mask=slot_ok[:, None])
            vo = cache_ptr + blk[:, None] * stride_cache_block + h * stride_cache_head + pos[:, None] * stride_cache_pos + k_packed_bytes
            if V_BITS == 3:
                g3v = (3 * tl.arange(0, head_dim // 8))[None, :]
                tl.store(vo + g3v, v_b0, mask=slot_ok[:, None])
                tl.store(vo + g3v + 1, v_b1, mask=slot_ok[:, None])
                tl.store(vo + g3v + 2, v_b2, mask=slot_ok[:, None])
            else:
                vc = tl.arange(0, v_packed_bytes)
                tl.store(vo + vc[None, :], v_bytes, mask=slot_ok[:, None])

            tl.store(
                scales_ptr + blk * stride_scales_block + h * stride_scales_head + pos * stride_scales_pos + 0,
                k_norm.to(tl.float16),
                mask=slot_ok,
            )
            tl.store(
                scales_ptr + blk * stride_scales_block + h * stride_scales_head + pos * stride_scales_pos + 1,
                v_norm.to(tl.float16),
                mask=slot_ok,
            )

    @triton.jit
    def _searchsorted(bounds_ptr, x, n_levels: tl.constexpr, head_dim: tl.constexpr):
        """Branch-free binary search: returns the bucket index of each element."""
        lo = tl.zeros(x.shape, dtype=tl.int32)
        hi = tl.full(x.shape, n_levels - 1, dtype=tl.int32)
        for _ in tl.static_range(8):  # ceil(log2(256)) == 8
            active = lo < hi
            mid = (lo + hi) // 2
            b = tl.load(bounds_ptr + mid, mask=active, other=float("inf"))
            go_right = x > b
            lo = tl.where(active & go_right, mid + 1, lo)
            hi = tl.where(active & ~go_right, mid, hi)
        return lo

    @triton.jit
    def _pack3(idx, ROWS: tl.constexpr, head_dim: tl.constexpr):
        """3-bit pack: 8 columns -> 24-bit little-endian word -> three bytes.

        Fields are disjoint, so a sum of left shifts equals OR-ing; matches
        ``quant.packing.pack_indices`` for head_dim=128 (48 bytes). Returned as
        three (ROWS, head_dim//8) byte planes for strided stores (a single
        (ROWS, 48) reshape is impossible: 48 is not a power of two).
        """
        ng: tl.constexpr = head_dim // 8
        s = tl.reshape(idx, (ROWS, ng, 8)).to(tl.int32)
        w = 1 << (3 * tl.arange(0, 8))
        word = tl.sum(s * w[None, None, :], axis=2)
        b0 = (word & 0xFF).to(tl.uint8)
        b1 = ((word >> 8) & 0xFF).to(tl.uint8)
        b2 = ((word >> 16) & 0xFF).to(tl.uint8)
        return b0, b1, b2

    @triton.jit
    def _pack(
        idx,
        ROWS: tl.constexpr,
        bits: tl.constexpr,
        head_dim: tl.constexpr,
        n_bytes: tl.constexpr,
    ):
        """Little-endian bit packing identical to ``quant/packing.py``.

        3-bit uses the 8-columns->24-bit word path; other widths divide 8: byte ``b`` holds
        the ``8 // bits`` indices starting at column ``b * (8 // bits)``, each
        shifted by ``k * bits``. The shifts are disjoint, so summing the
        weighted indices is the same as OR-ing them.
        """
        n_per: tl.constexpr = 8 // bits
        reshaped = tl.reshape(idx, (ROWS, n_bytes, n_per))
        weights = tl.zeros((n_per,), dtype=tl.int32)
        for k in tl.static_range(n_per):
            weights = tl.where(tl.arange(0, n_per) == k, 1 << (k * bits), weights)
        packed = tl.sum(reshaped.to(tl.int32) * weights[None, None, :], axis=2)
        return packed.to(tl.uint8)

    def reshape_and_cache_kernel(
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_scales: torch.Tensor,
        quantizer,
        layout: ThunderCacheLayout,
        block_rows: int = 16,
        value_strides: "tuple[int, int, int] | None" = None,
    ) -> None:
        """GPU cache write. Falls back to :func:`reshape_and_cache_ref` for
        bit widths the vectorised packer does not cover.

        NOTE: the kernel now rotates in fp32 (``input_precision="ieee"``) with an
        exact normalize, matching ``Codebook.quantize``. 3-bit K/4-bit V is
        unit-bit-exact (N=2048, non-contiguous, ``unpack_maxdiff == 0``), but
        enabling it for 3-bit STILL diverges the engine, so the store is left on
        the exact torch ref path. The unit parity evidently does not cover the
        engine's tensor contract (see next-experiment note). 4-bit K (16 levels)
        can differ by one level from the norm reduction order.
        """
        _allow3 = _os.environ.get("THUNDER_STORE3", "0").strip().lower() not in (
            "", "0", "false", "no", "off")
        _ok_bits = (1, 2, 3, 4, 8) if _allow3 else (1, 2, 4, 8)
        if layout.k_bits not in _ok_bits or layout.v_bits not in _ok_bits:
            reshape_and_cache_ref(
                key, value, slot_mapping, kv_cache, kv_scales, quantizer, layout
            )
            return
        n = key.shape[0]
        if n == 0:
            return
        if env_flag("THUNDER_DEBUG_KV") and not (
            slot_mapping.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            # NOTE: this block reads slot min/max back to the host. That is a D2H
            # sync and is ILLEGAL while a CUDA graph is being captured (it
            # invalidates the capture stream: "operation failed due to a
            # previous error during capture"), so it is skipped under capture
            # even when the debug flag is on.
            sm = slot_mapping
            print(
                "[TQ-CACHE] n=%d Hk=%d D=%d cache=%s cache_stride=%s "
                "scales=%s scales_stride=%s slots_min=%s slots_max=%s "
                "nb=%d bs=%d head_slot=%d"
                % (
                    n, layout.num_kv_heads, layout.head_dim,
                    tuple(kv_cache.shape), tuple(kv_cache.stride()),
                    tuple(kv_scales.shape), tuple(kv_scales.stride()),
                    int(sm.min().item()) if sm.numel() else "empty",
                    int(sm.max().item()) if sm.numel() else "empty",
                    kv_cache.shape[0], layout.block_size, layout.head_slot_bytes,
                ),
                flush=True,
            )
        block_rows = max(16, int(block_rows))
        rows = key.reshape(n, layout.num_kv_heads, layout.head_dim)
        vals = value.reshape(n, layout.num_kv_heads, layout.head_dim)
        grid = (triton.cdiv(n, block_rows),)
        _reshape_and_cache_kernel[grid](
            rows,
            vals,
            slot_mapping,
            kv_cache,
            kv_scales,
            quantizer.rotation.matrix.to(torch.float32),
            quantizer.k_codebook.boundaries.to(torch.float32),
            quantizer.v_codebook.boundaries.to(torch.float32),
            rows.stride(0),
            rows.stride(1),
            *(value_strides if value_strides is not None else vals.stride()),
            n,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_scales.stride(0),
            kv_scales.stride(2),
            kv_scales.stride(1),
            num_kv_heads=layout.num_kv_heads,
            head_dim=layout.head_dim,
            k_packed_bytes=layout.k_packed_bytes,
            v_packed_bytes=layout.v_packed_bytes,
            k_region_bytes=layout.k_region_bytes,
            kv_slot_bytes=layout.kv_slot_bytes,
            K_BITS=layout.k_bits,
            V_BITS=layout.v_bits,
            K_LEVELS=1 << layout.k_bits,
            V_LEVELS=1 << layout.v_bits,
            BLOCK_SIZE=layout.block_size,
            ROWS=block_rows,
        )

else:  # pragma: no cover

    def reshape_and_cache_kernel(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Triton is not available; use reshape_and_cache_ref on CPU."
        )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scales: torch.Tensor,
    quantizer,
    layout: ThunderCacheLayout,
) -> None:
    """Dispatch to the Triton path on CUDA, the reference path otherwise."""
    if key.is_cuda and _HAS_TRITON:
        reshape_and_cache_kernel(
            key, value, slot_mapping, kv_cache, kv_scales, quantizer, layout
        )
    else:
        reshape_and_cache_ref(
            key, value, slot_mapping, kv_cache, kv_scales, quantizer, layout
        )
