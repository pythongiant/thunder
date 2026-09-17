"""CUDA-graph-safe scratch pools.

The attention kernel needs transient fp32 buffers (combined split-K output,
per-split max / lse accumulators). Allocating them inside ``forward`` is a
correctness hazard under CUDA graphs: the allocator may hand back a *new*
address on replay, so the captured kernel writes to a stale pointer.

The fix is the STAR-KV pattern: reserve the pools to their worst-case size
*before* capture, then only ever slice into them. :func:`reserve_scratch`
grows (never shrinks) an idempotent dict of pools; :func:`alloc_scratch`
returns zero-copy slices of the reserved backing storage.

Rules enforced here:
* ``reserve_scratch`` must run before ``torch.cuda.graphs.CUDAGraph`` capture.
* ``alloc_scratch`` on the captured path must request sizes <= the reserved
  worst case, else it raises (rather than silently allocating).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from thunder_vllm.utils.logging import get_logger

logger = get_logger("attention.scratch")

# Canonical pool names. Keyed so that future split variants can add pools
# without touching call sites.
POOL_COMBINED = "comb_pool"  # (num_tokens, num_heads, head_dim) fp32
POOL_SPLIT_ACC = "split_acc_pool"  # (num_splits, num_tokens, num_heads, head_dim) fp32
POOL_SPLIT_M = "split_m_pool"  # (num_splits, num_tokens, num_heads) fp32
POOL_SPLIT_L = "split_l_pool"  # (num_splits, num_tokens, num_heads) fp32

_ALL_POOLS = (POOL_COMBINED, POOL_SPLIT_ACC, POOL_SPLIT_M, POOL_SPLIT_L)


@dataclass
class ScratchState:
    """Mutable bookkeeping for one device's scratch pools."""

    pools: dict[str, torch.Tensor] = field(default_factory=dict)
    caps: dict[str, torch.Size] = field(default_factory=dict)
    reserved_tokens: int = 0
    num_heads: int = 0
    head_dim: int = 0
    num_splits: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def key(self) -> tuple[int, int, int, int, str]:
        return (
            self.reserved_tokens,
            self.num_heads,
            self.head_dim,
            self.num_splits,
            str(self.device),
        )


def _shape_for(pool: str, tokens: int, heads: int, dim: int, splits: int) -> torch.Size:
    if pool == POOL_COMBINED:
        return torch.Size((tokens, heads, dim))
    if pool == POOL_SPLIT_ACC:
        return torch.Size((splits, tokens, heads, dim))
    if pool in (POOL_SPLIT_M, POOL_SPLIT_L):
        return torch.Size((splits, tokens, heads))
    raise KeyError(f"unknown scratch pool {pool!r}")


def reserve_scratch(
    max_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | str,
    scratch: ScratchState,
    *,
    num_splits: int = 1,
) -> ScratchState:
    """Grow ``scratch`` so any captured batch up to ``max_tokens`` fits.

    Must be called before CUDA-graph capture. Idempotent: re-reserving the same
    or smaller shapes is a no-op; larger shapes reallocate (call this during
    warmup, never during replay).
    """
    dev = torch.device(device)
    tokens = int(max_tokens)
    heads = int(num_heads)
    dim = int(head_dim)
    splits = int(num_splits)
    if tokens <= 0 or heads <= 0 or dim <= 0:
        raise ValueError("reserve_scratch needs positive tokens/heads/head_dim")

    wanted_key = (tokens, heads, dim, splits, str(dev))
    if scratch.key == wanted_key and all(p in scratch.pools for p in _ALL_POOLS):
        return scratch

    for pool in _ALL_POOLS:
        shape = _shape_for(pool, tokens, heads, dim, splits)
        cur = scratch.pools.get(pool)
        if cur is None or tuple(cur.shape) != tuple(shape) or cur.device != dev:
            scratch.pools[pool] = torch.zeros(shape, dtype=torch.float32, device=dev)
            scratch.caps[pool] = shape

    scratch.reserved_tokens = tokens
    scratch.num_heads = heads
    scratch.head_dim = dim
    scratch.num_splits = splits
    scratch.device = dev
    logger.info(
        "reserved scratch: tokens=%d heads=%d dim=%d splits=%d device=%s "
        "(combined %.1f MiB, split_acc %.1f MiB)",
        tokens,
        heads,
        dim,
        splits,
        scratch.pools[POOL_COMBINED].numel() * 4 / 2**20,
        scratch.pools[POOL_SPLIT_ACC].numel() * 4 / 2**20,
    )
    return scratch


def alloc_scratch(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | str,
    scratch: ScratchState,
    *,
    num_splits: int = 1,
    pool: str = POOL_COMBINED,
) -> torch.Tensor:
    """Zero-copy slice of a reserved pool for the requested shape.

    Raises if the request exceeds the reservation: that means ``reserve_scratch``
    was called with a smaller worst case than an actual captured batch, which
    would have produced a silent out-of-bounds write inside the graph.
    """
    dev = torch.device(device)
    if pool not in scratch.pools or dev != scratch.device:
        raise RuntimeError(
            "scratch not reserved for this device; call reserve_scratch during "
            "warmup and before CUDA-graph capture"
        )
    need = _shape_for(pool, int(num_tokens), int(num_heads), int(head_dim), int(num_splits))
    have = scratch.pools[pool].shape
    if tuple(need) > tuple(have):
        raise RuntimeError(
            f"scratch pool {pool} overflow: need {tuple(need)} > reserved "
            f"{tuple(have)}; reserve for the largest captured batch"
        )
    backing = scratch.pools[pool]
    if pool == POOL_COMBINED:
        return backing[: int(num_tokens), : int(num_heads), : int(head_dim)]
    if pool in (POOL_SPLIT_M, POOL_SPLIT_L):
        return backing[: int(num_splits), : int(num_tokens), : int(num_heads)]
    return backing[
        : int(num_splits), : int(num_tokens), : int(num_heads), : int(head_dim)
    ]


def new_scratch() -> ScratchState:
    return ScratchState()


def scratch_dict(scratch: ScratchState) -> dict[str, torch.Tensor]:
    """Backing pools, for code that wants to pass them around explicitly."""
    return scratch.pools


__all__ = [
    "POOL_COMBINED",
    "POOL_SPLIT_ACC",
    "POOL_SPLIT_L",
    "POOL_SPLIT_M",
    "ScratchState",
    "alloc_scratch",
    "new_scratch",
    "reserve_scratch",
    "scratch_dict",
]
