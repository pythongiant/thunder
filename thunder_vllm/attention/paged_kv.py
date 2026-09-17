"""Paged-KV gather layer.

vLLM hands the attention layer a ``block_table`` and a ``slot_mapping``, not a
contiguous KV tensor. The CuTeDSL kernel wants a contiguous
``(num_page_rows, block_size, num_kv_heads, packed_bytes)`` view whose page row
``r`` maps to block ``block_table[req, n_block]``.

v0.1 strategy (see README "Known limitations")
---------------------------------------------
Gather once into a pre-allocated, fixed-shape buffer that the CUDA graph can
record:

    gather_packed_tiles(block_table, kv_cache, kv_scales, seq_lens)

The buffers are reserved at warmup (:meth:`PagedKVManager.reserve`) and never
re-allocated on the captured path, so the graph always sees the same pointers.

The follow-up (and the reason this module also exposes
:meth:`block_table_row_map`) is to delete the gather entirely and have the
kernel's TMA atoms walk the block table in-kernel, the way FA4's
``flash_attn/cute/paged_kv.py::PagedKVManager`` does. Everything downstream of
this module already consumes the kernel layout, so the swap is local.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import os as _os

import torch

from thunder_vllm.attention.cache_layout import ThunderCacheLayout
from thunder_vllm.utils.logging import get_logger

logger = get_logger("attention.paged_kv")


@dataclass
class GatheredKV:
    """Contiguous gathered cache, matching the kernel's argument layout."""

    k_packed: torch.Tensor  # (page_rows, block_size, Hk, k_packed_bytes)
    v_packed: torch.Tensor  # (page_rows, block_size, Hk, v_packed_bytes)
    k_norm: torch.Tensor  # (page_rows, block_size, Hk)
    v_norm: torch.Tensor  # (page_rows, block_size, Hk)

    @property
    def n_page_rows(self) -> int:
        return self.k_packed.shape[0]


class PagedKVManager:
    """Owns the pre-allocated gather buffers for one attention layer.

    One manager is shared by all layers of a given shape (the gathered cache is
    identical regardless of which layer reads it), but the plugin keeps one per
    impl so that pointer identity is trivially stable under capture.
    """

    def __init__(
        self,
        layout: ThunderCacheLayout,
        *,
        max_num_reqs: int,
        max_blocks_per_req: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.layout = layout
        self.max_num_reqs = int(max_num_reqs)
        self.max_blocks_per_req = int(max_blocks_per_req)
        self.device = torch.device(device)
        self._buffers: GatheredKV | None = None

    # ------------------------------------------------------------------ #
    # Shape / reservation
    # ------------------------------------------------------------------ #
    @property
    def max_page_rows(self) -> int:
        return self.max_num_reqs * self.max_blocks_per_req

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        bs = self.layout.block_size
        hk = self.layout.num_kv_heads
        rows = self.max_page_rows
        return {
            "k_packed": (rows, bs, hk, self.layout.k_packed_bytes),
            "v_packed": (rows, bs, hk, self.layout.v_packed_bytes),
            "k_norm": (rows, bs, hk),
            "v_norm": (rows, bs, hk),
        }

    def reserve(self) -> GatheredKV:
        """Allocate (or return) the worst-case gather buffers.

        Must be called before CUDA-graph capture. Cheap and idempotent.
        """
        if self._buffers is None:
            shapes = self.shape
            self._buffers = GatheredKV(
                k_packed=torch.empty(
                    shapes["k_packed"], dtype=torch.uint8, device=self.device
                ),
                v_packed=torch.empty(
                    shapes["v_packed"], dtype=torch.uint8, device=self.device
                ),
                k_norm=torch.empty(
                    shapes["k_norm"], dtype=torch.float16, device=self.device
                ),
                v_norm=torch.empty(
                    shapes["v_norm"], dtype=torch.float16, device=self.device
                ),
            )
            logger.info(
                "reserved gather buffers: page_rows=%d block_size=%d Hk=%d "
                "(K %d B, V %d B per row)",
                self.max_page_rows,
                self.layout.block_size,
                self.layout.num_kv_heads,
                self.layout.k_packed_bytes,
                self.layout.v_packed_bytes,
            )
        return self._buffers

    # ------------------------------------------------------------------ #
    # Block-table helpers
    # ------------------------------------------------------------------ #
    def block_table_row_map(self, block_table: torch.Tensor) -> torch.Tensor:
        """Map every gathered page row to its source cache block.

        Returns ``int32`` of shape ``(max_page_rows,)``;
        ``row = block_table[i // B, i % B]``. Exposed because the in-kernel TMA
        rewrite consumes exactly this mapping.
        """
        bt = block_table.to(torch.int32)
        b = self.max_blocks_per_req
        # (R, B) -> (R*B,)
        return bt[:, :b].reshape(-1).contiguous()

    def seq_row_counts(
        self, seq_lens: torch.Tensor, block_size: int | None = None
    ) -> torch.Tensor:
        """Number of *valid* page rows per request, i.e. ceil(seq_len / bs)."""
        bs = block_size or self.layout.block_size
        return torch.div(seq_lens + bs - 1, bs, rounding_mode="floor")

    # ------------------------------------------------------------------ #
    # Gather
    # ------------------------------------------------------------------ #
    def gather_packed_tiles(
        self,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_scales: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        live_blocks: int | None = None,
    ) -> GatheredKV:
        """Gather the paged cache into the reserved buffers.

        Writes in place; returns the same buffer object every call so the
        pointers baked into a captured CUDA graph stay valid.

        Only the LIVE region is gathered: ``r = block_table.shape[0]`` requests and
        ``b_live`` blocks per request. The reservation is the worst case, so the
        rows past the live region are left stale and are never read (``seq_lens``
        masks them, and the kernel's row bound is ``kv_len``). The old code
        gathered the full ``max_num_reqs * max_blocks_per_req`` table every call,
        which made a 1-request decode rebuild ~524288 rows (~536 MB) per layer per
        token. Request-major row order (stride ``max_blocks_per_req``) is
        preserved so the launcher's ``kv_row_stride`` is unchanged.

        ``b_live`` must be a HOST value known before this call; under CUDA-graph
        capture there can be no device-to-host sync. Callers pass ``live_blocks``
        computed from a CPU-resident mirror (vLLM's ``seq_lens_cpu``); when it is
        absent, ``seq_lens`` is only reduced on the host when it is safe to do so,
        and capture falls back to the full table (correct for any replay length,
        just not trimmed).
        """
        out = self.reserve()
        bt = block_table.to(torch.int64)
        if bt.shape[0] > self.max_num_reqs or bt.shape[1] > self.max_blocks_per_req:
            raise ValueError(
                f"block_table {tuple(bt.shape)} exceeds reserved "
                f"({self.max_num_reqs}, {self.max_blocks_per_req})"
            )
        bs = self.layout.block_size
        hk = self.layout.num_kv_heads
        k_pb = self.layout.k_packed_bytes
        v_pb = self.layout.v_packed_bytes
        r = int(bt.shape[0])
        b = int(bt.shape[1])

        capturing = (
            bt.is_cuda and torch.cuda.is_current_stream_capturing()  # type: ignore[attr-defined]
        )
        if live_blocks is not None:
            b_live = int(live_blocks)
        elif capturing:
            # No D2H sync is allowed while capturing; the full table is correct
            # for every replay length. Trim by passing `live_blocks` instead.
            b_live = self.max_blocks_per_req
        elif seq_lens is not None and seq_lens.numel() > 0:
            if seq_lens.device.type == "cpu":
                sl = seq_lens[:r].to(torch.int64)
            else:
                sl = seq_lens[:r].to(torch.int64).cpu()
            need = int(
                torch.div(sl + bs - 1, bs, rounding_mode="floor").max().item()
            )
            b_live = max(1, min(b, need))
        else:
            b_live = b
        b_live = max(1, min(b_live, self.max_blocks_per_req))
        bt_live = bt[:r, :b_live].contiguous()

        k_codes = self.layout.k_codes(kv_cache)
        v_codes = self.layout.v_codes(kv_cache)
        kn = self.layout.k_norm(kv_scales)
        vn = self.layout.v_norm(kv_scales)

        # (r, b_live, bs, Hk, pb) in request-major order.
        k = k_codes[bt_live].reshape(r, b_live, bs, hk, k_pb)
        v = v_codes[bt_live].reshape(r, b_live, bs, hk, v_pb)

        kv_view = out.k_packed.view(
            self.max_num_reqs, self.max_blocks_per_req, bs, hk, k_pb
        )
        vv_view = out.v_packed.view(
            self.max_num_reqs, self.max_blocks_per_req, bs, hk, v_pb
        )
        kn_view = out.k_norm.view(self.max_num_reqs, self.max_blocks_per_req, bs, hk)
        vn_view = out.v_norm.view(self.max_num_reqs, self.max_blocks_per_req, bs, hk)

        kv_view[:r, :b_live].copy_(k)
        vv_view[:r, :b_live].copy_(v)
        kn_view[:r, :b_live].copy_(kn[bt_live])
        vn_view[:r, :b_live].copy_(vn[bt_live])
        if _os.environ.get("THUNDER_DEBUG_LAYOUT"):
            print(
                f"[TQ-GATHER-TRIM] r={r} b={b} b_live={b_live} "
                f"live_page_rows={r * b_live} reserved_page_rows={self.max_page_rows}",
                flush=True,
            )
        return out

    def gather_packed_tiles_ref(
        self,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_scales: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> GatheredKV:
        """Pure-torch gather producing freshly allocated output.

        This is both the CPU reference used by ``tests/test_paged_kv.py`` and
        the current implementation of the GPU path (the copy in
        :meth:`gather_packed_tiles` turns it into an in-place update of the
        pre-reserved buffers).
        """
        bt = block_table.to(torch.int64)
        if bt.shape[0] > self.max_num_reqs or bt.shape[1] > self.max_blocks_per_req:
            raise ValueError(
                f"block_table {tuple(bt.shape)} exceeds reserved "
                f"({self.max_num_reqs}, {self.max_blocks_per_req})"
            )
        if bt.shape[1] < self.max_blocks_per_req:
            pad = torch.zeros(
                (bt.shape[0], self.max_blocks_per_req - bt.shape[1]),
                dtype=bt.dtype,
                device=bt.device,
            )
            bt = torch.cat([bt, pad], dim=1)
        if bt.shape[0] < self.max_num_reqs:
            # Pad requests with block 0 (never read: masked by seq_lens).
            pad = torch.zeros(
                (self.max_num_reqs - bt.shape[0], self.max_blocks_per_req),
                dtype=bt.dtype,
                device=bt.device,
            )
            bt = torch.cat([bt, pad], dim=0)

        # Derive the page-row count from the (now padded) block table rather than
        # from self.max_page_rows. vLLM's KV-cache warm-up hands us a table whose
        # reserved dimensions do not have to equal the ones the manager was built
        # with, and a fixed-target reshape then fails with
        #   shape '[32768, 16, 8, 64]' is invalid for input of size 1879048192
        page_rows = int(bt.shape[0]) * int(bt.shape[1])
        if _os.environ.get("THUNDER_DEBUG_LAYOUT"):
            print(
                f"[TQ-GATHER] bt={tuple(bt.shape)} reserved=({self.max_num_reqs},"
            f"{self.max_blocks_per_req}) max_page_rows={self.max_page_rows} "
            f"page_rows={page_rows} cache={tuple(kv_cache.shape)} "
            f"cache_stride={tuple(kv_cache.stride())} bs(layout)={self.layout.block_size} "
            f"k_packed={self.layout.k_packed_bytes} v_packed={self.layout.v_packed_bytes} "
                f"Hk={self.layout.num_kv_heads}",
                flush=True,
            )

        k_codes = self.layout.k_codes(kv_cache)  # (nb, bs, Hk, pb)
        v_codes = self.layout.v_codes(kv_cache)

        # (R, B, bs, Hk, pb) -> (R*B*bs, Hk, pb)
        k = k_codes[bt].reshape(-1, self.layout.num_kv_heads, self.layout.k_packed_bytes)
        v = v_codes[bt].reshape(-1, self.layout.num_kv_heads, self.layout.v_packed_bytes)
        if _os.environ.get("THUNDER_DEBUG_LAYOUT"):
            print(
                f"[TQ-GATHER] k_codes={tuple(k_codes.shape)} v_codes={tuple(v_codes.shape)} "
                f"k={tuple(k.shape)} v={tuple(v.shape)}",
                flush=True,
            )
        kn = self.layout.k_norm(kv_scales)[bt].reshape(
            -1, self.layout.block_size, self.layout.num_kv_heads
        )
        vn = self.layout.v_norm(kv_scales)[bt].reshape(
            -1, self.layout.block_size, self.layout.num_kv_heads
        )
        return GatheredKV(
            k_packed=k.reshape(
                page_rows, self.layout.block_size,
                self.layout.num_kv_heads, self.layout.k_packed_bytes,
            ),
            v_packed=v.reshape(
                page_rows, self.layout.block_size,
                self.layout.num_kv_heads, self.layout.v_packed_bytes,
            ),
            k_norm=kn,
            v_norm=vn,
        )


def make_paged_kv_manager(
    layout: ThunderCacheLayout,
    *,
    max_num_reqs: int,
    max_model_len: int,
    device: torch.device | str = "cuda",
) -> PagedKVManager:
    """Convenience constructor deriving the worst-case block count."""
    blocks = (int(max_model_len) + layout.block_size - 1) // layout.block_size
    return PagedKVManager(
        layout,
        max_num_reqs=max_num_reqs,
        max_blocks_per_req=blocks,
        device=device,
    )
