"""Little-endian bit packing shared by the Triton store kernel and the CPU
reference paths.

The attention kernel's dequant reads a packed row as follows (see
``attention/cute_kernel.py::_dequantize_tile``): for output column ``c`` the
index occupies bit positions ``[c*bits, c*bits + bits)`` of a little-endian
bit stream. Concretely, for every bit width the kernel does

    4-bit:  byte = packed[c // 2];              idx = (byte >> ((c % 2) * 4)) & 0xF
    2-bit:  byte = packed[c // 4];              idx = (byte >> ((c % 4) * 2)) & 0x3
    8-bit:  byte = packed[c];                   idx = byte
    3-bit:  word = b0 | (b1 << 8);              idx = (word >> (3*c % 8)) & 0x7

which is exactly little-endian packing. These helpers are the *reference*
implementation; the GPU store path reproduces the same layout inline so the
two never drift.
"""

from __future__ import annotations

from functools import lru_cache

import torch


def packed_bytes(head_dim: int, bits: int) -> int:
    return (head_dim * bits + 7) // 8


@lru_cache(maxsize=128)
def _contributions(bits: int, head_dim: int) -> tuple:
    """Per-output-byte list of ``(col, src_shift, dst_shift, mask)``.

    A byte can be fed by at most two columns (``bits <= 8``), and the encoding
    is fully determined by ``(bits, head_dim)``, so this is computed once.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n_bytes = packed_bytes(head_dim, bits)
    table: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_bytes)]
    for c in range(head_dim):
        pos = c * bits
        first = pos // 8
        last = (pos + bits - 1) // 8
        for j in range(first, last + 1):
            dst = pos - j * 8
            if dst >= 0:
                src = 0
                width = min(bits, 8 - dst)
            else:
                src = -dst
                dst = 0
                width = bits - src
            mask = (1 << width) - 1
            table[j].append((c, src, dst, mask))
    return tuple(tuple(row) for row in table)


def pack_indices(indices: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    """Pack ``(..., head_dim)`` integer indices into ``(..., packed_bytes)`` uint8.

    ``indices`` values must fit in ``bits`` bits.
    """
    if indices.shape[-1] != head_dim:
        raise ValueError(
            f"last dim {indices.shape[-1]} != head_dim {head_dim}"
        )
    n_bytes = packed_bytes(head_dim, bits)
    idx = indices.to(torch.int32)
    out = torch.zeros(*indices.shape[:-1], n_bytes, dtype=torch.uint8, device=indices.device)
    table = _contributions(bits, head_dim)
    for j, contribs in enumerate(table):
        acc = torch.zeros(indices.shape[:-1], dtype=torch.int32, device=indices.device)
        for c, src, dst, mask in contribs:
            acc = acc | (((idx[..., c] >> src) & mask) << dst)
        out[..., j] = acc.to(torch.uint8)
    return out


def unpack_indices(packed: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    """Inverse of :func:`pack_indices`. Returns ``int64`` indices."""
    n_bytes = packed_bytes(head_dim, bits)
    if packed.shape[-1] != n_bytes:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} != expected {n_bytes}"
        )
    out = torch.zeros(*packed.shape[:-1], head_dim, dtype=torch.int64, device=packed.device)
    table = _contributions(bits, head_dim)
    for j, contribs in enumerate(table):
        byte = packed[..., j].to(torch.int64)
        for c, src, dst, mask in contribs:
            out[..., c] = out[..., c] | (((byte >> dst) & mask) << src)
    return out
