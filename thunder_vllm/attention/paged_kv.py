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
from thunder_vllm.utils.logging import env_flag, get_logger

logger = get_logger("attention.paged_kv")

# ---- the one hard limit on a gathered buffer -----------------------------
# CuTeDSL addresses the buffers it is handed with 32-bit offsets. A gathered
# tensor past this many bytes therefore WRAPS: the kernel reads a wrapped
# address, which is `cudaErrorIllegalAddress` when it lands outside the
# allocation and silently wrong data when it lands back inside it (the buffer is
# much larger than the wrap window, so most CTAs are in the second case -- this
# is why the fault looked flaky and why the engine's 16k decode steps "ran").
#
# The request-major layout is the one that gets big: its reservation is
# `max_num_reqs * max_blocks_per_req` block-rows because the row stride is the
# engine's block-table width. Measured on Qwen3-8B slots (block_size 16, Hk 8,
# k 3-bit -> 48 B, v 4-bit -> 64 B per (token, head)):
#
#   4k  (ctx 4160): 4.33M token rows -> K 1.66 GB, V 2.22 GB -> clean
#   8k  (ctx 8256): 8.52M token rows -> K 3.27 GB, V 4.36 GB
#   16k (ctx 16448): 16.9M token rows -> K 6.49 GB, V 8.65 GB -> faults
#
# The dense (CSR) gather reserves by the PHYSICAL block count instead, which is
# bounded by the cache: at 16k that is 1.71 / 2.28 GB and it runs (`THUNDER_8B_INDIRECT=1`
# completes init and generation). So the boundary lies between 2.22 GB (clean)
# and 6.49 GB (faulting) and this cap sits inside that gap: 4k and the loop's
# batch-scaled shapes stay on request-major, 8k/16k switch.
ADDRESSABLE_GATHER_BYTES = 4 << 30


def gathered_tensor_bytes(rows: int, layout: ThunderCacheLayout,
                          num_kv_heads: int) -> int:
    """Bytes of the largest tensor the kernel addresses for ``rows`` block-rows.

    The kernel sees K as ``(rows * block_size, Hk, k_packed_bytes)`` uint8, so
    the unit is the (token, kv-head) row: bytes == elements for uint8, and V's
    wider packing is what makes it the larger of the two.
    """
    per_token = int(layout.block_size) * int(num_kv_heads)
    return per_token * max(int(layout.k_packed_bytes), int(layout.v_packed_bytes)) * int(rows)


def check_addressable(rows: int, layout: ThunderCacheLayout, num_kv_heads: int,
                      where: str) -> int:
    """Refuse a reservation whose K/V tensor CuTeDSL cannot address.

    A loud failure here is the whole point: the alternative is an illegal address
    (or silent corruption) thousands of kernel launches later, with nothing in
    the traceback pointing at the reservation.

    ``THUNDER_ALLOW_UNADDRESSABLE=1`` skips the check, the same way
    ``THUNDER_8B_INDIRECT`` bypasses the path policy: it is how the 32-bit wrap
    itself is reproduced on demand.
    """
    nbytes = gathered_tensor_bytes(rows, layout, num_kv_heads)
    if env_flag("THUNDER_ALLOW_UNADDRESSABLE"):
        return nbytes
    if nbytes > ADDRESSABLE_GATHER_BYTES:
        raise RuntimeError(
            f"{where}: the gathered K/V tensor for {int(rows)} block-rows would be "
            f"{nbytes / 2**30:.2f} GiB, past the {ADDRESSABLE_GATHER_BYTES / 2**30:.0f} GiB "
            f"a 32-bit CuTeDSL tensor index can address (docs/FAILURE_MODES.md 15). "
            f"Use the dense gather (THUNDER_8B_INDIRECT=1 / a smaller batch "
            f"reservation): it reserves by physical blocks, not by the block-table width."
        )
    return nbytes

# Narrow debug counters (branch): how often CSR metadata is rebuilt/uploaded.
CSR_COUNTS = {
    "gather_calls": 0,      # CSR METADATA builds (indptr + page index): 1/step
    "uploads": 0,           # indptr H2D: 1/step
    "payload_gathers": 0,   # per-layer KV select: N_layers/step (per-layer cache)
    "index_hits": 0,
    "index_misses": 0,
    "reserve_calls": 0,
}


@dataclass
class CsrIndex:
    """Per-step, layer-independent CSR metadata (shared by every layer)."""

    indptr: torch.Tensor   # (r+1,) int32, token base per request
    sel: torch.Tensor      # (nrows,) int64, physical page id in CSR order
    nrows: int
    num_blocks: int
    key: tuple = ()


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
        self._rows: int | None = None
        self._csr: "CsrIndex | None" = None
        self._csr_md: object | None = None
        # Persistent, capture-safe CSR scratch (allocated with the buffers).
        self._csr_bufs: dict | None = None

    # ------------------------------------------------------------------ #
    # Shape / reservation
    # ------------------------------------------------------------------ #
    @property
    def max_page_rows(self) -> int:
        return self.max_num_reqs * self.max_blocks_per_req

    @property
    def page_rows(self) -> int:
        return self._rows if self._rows is not None else self.max_page_rows

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        bs = self.layout.block_size
        hk = self.layout.num_kv_heads
        rows = self.page_rows
        return {
            "k_packed": (rows, bs, hk, self.layout.k_packed_bytes),
            "v_packed": (rows, bs, hk, self.layout.v_packed_bytes),
            "k_norm": (rows, bs, hk),
            "v_norm": (rows, bs, hk),
        }

    def reserve(self, cap_rows: int | None = None) -> GatheredKV:
        """Allocate (or return) the gather buffers.

        ``cap_rows`` (physical block count) bounds the reservation so it does not
        scale with ``max_model_len``. The request-major path calls this with no
        cap and is unchanged.
        """
        if self._buffers is not None and cap_rows is not None and self._rows is not None:
            # The two gather paths share one reservation, and whoever reserves
            # first fixes its size. Growing later would write past the allocated
            # rows (a silent OOB), so refuse loudly instead: the path choice must
            # be stable for the life of the manager.
            if int(cap_rows) > self._rows:
                raise ValueError(
                    f"gather reservation is {self._rows} block-rows but this call "
                    f"needs {int(cap_rows)}; the gather path must not change size "
                    f"after the first reserve"
                )
        if self._buffers is None:
            CSR_COUNTS["reserve_calls"] += 1
            if cap_rows is not None:
                self._rows = min(self.max_page_rows, max(int(cap_rows), 1))
            check_addressable(
                self._rows if self._rows is not None else self.max_page_rows,
                self.layout, self.layout.num_kv_heads,
                "PagedKVManager.reserve",
            )
            # CSR metadata scratch, sized to the FULL table so no step can exceed
            # it (dest indices use max_blocks_per_req); address-stable for capture.
            cap = self.max_num_reqs * self.max_blocks_per_req
            self._csr_bufs = {
                "indptr": torch.zeros(self.max_num_reqs + 1, dtype=torch.int32,
                                      device=self.device),
                "bpr": torch.zeros(self.max_num_reqs, dtype=torch.int64,
                                   device=self.device),
                "sel": torch.zeros(cap, dtype=torch.int64, device=self.device),
                # int32 to match the block table: scatter_ requires equal dtypes,
                # and a cast would allocate inside the captured region.
                "padded": torch.zeros(cap + 1, dtype=torch.int32, device=self.device),
                # Build scratch. The device-side CSR build runs inside the captured
                # region, and an allocation there has to come from the graph pool:
                # it was building ~65 MB of temporaries per layer, which made the
                # pool grow mid-capture and invalidated the 32k capture. These are
                # persistent and reused, so the build allocates nothing.
                "ar": torch.arange(self.max_blocks_per_req, dtype=torch.int64,
                                   device=self.device),
                "dest": torch.zeros(cap, dtype=torch.int64, device=self.device),
                "valid": torch.zeros(cap, dtype=torch.bool, device=self.device),
                "off": torch.zeros(self.max_num_reqs, dtype=torch.int64,
                                   device=self.device),
            }
            self._csr = None
            self._csr_md = None
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
        out = self.reserve(self.max_page_rows)
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

        k_codes = self.layout.k_codes(kv_cache)
        v_codes = self.layout.v_codes(kv_cache)
        kn = self.layout.k_norm(kv_scales)
        vn = self.layout.v_norm(kv_scales)

        # Clamp block ids into every tensor we are about to index. vLLM's
        # post-capture kernel warm-up hands a block table whose entries can
        # exceed the allocated block count; a raw index then trips a CUDA device
        # assert ("index out of bounds"/"scatter gather kernel index out of
        # bounds"). Those positions are padding -- masked by seq_lens and never
        # read by the kernel -- so clamping is correct. Use the minimum of the
        # four source sizes: the norm buffers (`kv_scales`) are a separate
        # tensor from the cache and need not have the same block count.
        nb = min(
            int(k_codes.shape[0]), int(v_codes.shape[0]),
            int(kn.shape[0]), int(vn.shape[0]),
        )
        bt_live = bt[:r, :b_live].clamp(0, max(nb - 1, 0)).contiguous()
        bt_flat = bt_live.reshape(-1)

        if env_flag("THUNDER_DEBUG_GATHER") and not capturing:
            print(
                f"[TQ-GATHER-CHK] nb={nb} r={r} b_live={b_live} "
                f"bt_min={int(bt_flat.min())} bt_max={int(bt_flat.max())} "
                f"k_codes0={int(k_codes.shape[0])} kn0={int(kn.shape[0])}",
                flush=True,
            )

        # ``torch.index_select`` on the cache's block axis instead of advanced
        # indexing ``k_codes[bt_live]``: the advanced-index kernel runs inside a
        # CUDA graph capture here and poisons the context (the failure surfaces
        # later at the next cuBLAS op). index_select on a static-shape index is
        # the capture-safe equivalent.
        # (r, b_live, bs, Hk, pb) in request-major order.
        k = torch.index_select(k_codes, 0, bt_flat).reshape(r, b_live, bs, hk, k_pb)
        v = torch.index_select(v_codes, 0, bt_flat).reshape(r, b_live, bs, hk, v_pb)
        kn_sel = torch.index_select(kn, 0, bt_flat).reshape(r, b_live, bs, hk)
        vn_sel = torch.index_select(vn, 0, bt_flat).reshape(r, b_live, bs, hk)

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
        kn_view[:r, :b_live].copy_(kn_sel)
        vn_view[:r, :b_live].copy_(vn_sel)
        if env_flag("THUNDER_DEBUG_LAYOUT"):
            print(
                f"[TQ-GATHER-TRIM] r={r} b={b} b_live={b_live} "
                f"live_page_rows={r * b_live} reserved_page_rows={self.max_page_rows}",
                flush=True,
            )
        return out

    # -- Phase A: shared per-step CSR metadata ---------------------------
    def _step_key(self, metadata, num_blocks: int):
        """Key that is identical across the layers of one step and differs across
        steps. vLLM hands the SAME metadata object to every layer of a step and
        builds a fresh one per step (supports_update_block_table=False), so
        ``id(metadata)`` is stable-within-step; ``seq_lens_cpu`` content is the
        per-step discriminator (block_table/seq_lens pointers are persistent
        sliced buffers). Returns ``None`` when no host mirror exists -> never
        memoize (correctness over speed, no device sync)."""
        bt = metadata.block_table
        sl = getattr(metadata, "seq_lens_cpu", None)
        if not (torch.is_tensor(sl) and sl.numel() > 0 and not sl.is_cuda):
            return None
        return (
            id(metadata), bt.data_ptr(), tuple(bt.shape),
            int(getattr(metadata, "num_actual_tokens", 0) or 0),
            int(getattr(metadata, "max_query_len", 0) or 0),
            bool(getattr(metadata, "is_prefill", False)),
            int(num_blocks),
            (sl.data_ptr(), tuple(sl.shape), tuple(int(x) for x in sl.tolist())),
        )

    def _blocks_per_req(self, metadata) -> list[int]:
        r = int(metadata.block_table.shape[0])
        bs = self.layout.block_size
        sl = getattr(metadata, "seq_lens_cpu", None)
        if not (torch.is_tensor(sl) and sl.numel() > 0 and not sl.is_cuda):
            sl = metadata.seq_lens[:r].to(torch.int64).cpu()
        return [max(0, int((int(s) + bs - 1) // bs)) for s in sl.tolist()]

    def _build_csr_index(self, block_table, blocks_per_req, num_blocks) -> "CsrIndex":
        CSR_COUNTS["gather_calls"] += 1
        bs = self.layout.block_size
        bt = block_table.to(torch.int64)
        r = len(blocks_per_req)
        b = int(bt.shape[1])
        dev = bt.device
        idx_parts = []
        base = 0
        indptr = [0]
        for bn in blocks_per_req:
            bn = max(0, min(int(bn), b))
            if bn:
                idx_parts.append(torch.arange(base, base + bn, device=dev, dtype=torch.int64))
                base += b
            indptr.append(indptr[-1] + bn * bs)
        flat_idx = (
            torch.cat(idx_parts) if idx_parts
            else torch.empty(0, dtype=torch.int64, device=dev)
        )
        nrows = int(flat_idx.numel())
        if nrows:
            flat_idx = flat_idx.clamp_(0, max(num_blocks - 1, 0))
        self._indptr = torch.tensor(indptr, dtype=torch.int32, device=dev)
        CSR_COUNTS["uploads"] += 1
        sel = bt.reshape(-1)[flat_idx] if nrows else torch.empty(0, dtype=torch.int64, device=dev)
        return CsrIndex(self._indptr, sel, nrows, int(num_blocks))

    def build_csr_device(self, metadata, num_blocks: int,
                         compute_nrows: bool = True) -> "CsrIndex":
        """Capture-safe CSR build: persistent buffers, device ops only, no sync.

        Sizes come from HOST-STATIC shapes (block_table, max blocks), never from
        ``seq_lens`` values, so capture metadata (seq_lens=1) records the same
        ops. ``sel``/``indptr`` addresses are stable; only values change per
        replay.
        """
        CSR_COUNTS["gather_calls"] += 1
        bufs = self._csr_bufs
        if bufs is None:
            self.reserve(int(num_blocks))
            bufs = self._csr_bufs
        bs = self.layout.block_size
        b = int(metadata.block_table.shape[1])
        r = int(metadata.block_table.shape[0])
        cap = self.max_num_reqs * self.max_blocks_per_req
        dev = metadata.block_table.device
        indptr = bufs["indptr"]
        bpr = bufs["bpr"]
        sel = bufs["sel"]
        padded = bufs["padded"]
        indptr.zero_()
        if r > 0:
            bpr.zero_()
            ar = bufs["ar"][:b].unsqueeze(0)
            n = r * b
            dead = bufs["valid"][:n].view(r, b)
            dest = bufs["dest"][:n].view(r, b)
            off = bufs["off"][:r]
            # Every step writes only into persistent buffers: an allocation inside
            # a captured region has to come from the graph pool, and this build
            # runs inside the capture.
            off.copy_(metadata.seq_lens[:r])
            off.add_(bs - 1)
            off.floor_divide_(bs)
            off.clamp_(0, b)
            bpr[:r].copy_(off)
            torch.ge(ar, bpr[:r].unsqueeze(1), out=dead)
            torch.cumsum(bpr[:r], 0, out=off)
            off.sub_(bpr[:r])
            torch.add(off.unsqueeze(1), ar, out=dest)
            dest.clamp_(0, cap - 1)
            # Dead slots point at the sentinel slot of `padded` (one longer than
            # cap), so the scatter writes them somewhere harmless.
            dest.masked_fill_(dead, cap)
            padded.zero_()
            padded.scatter_(
                0, dest.reshape(-1), metadata.block_table[:r].reshape(-1)
            )
            sel.copy_(padded[:cap])
            sel.clamp_(0, max(int(num_blocks) - 1, 0))
            # indptr[i] is request i's token base, i.e. the EXCLUSIVE prefix sum --
            # which is the inclusive sum shifted one slot, since indptr[0] = 0.
            off.add_(bpr[:r])
            indptr[1:r + 1].copy_(off)
            indptr[1:r + 1].mul_(bs)
        nrows = int(bpr[:r].sum().item()) if (compute_nrows and r > 0) else -1
        self._indptr = indptr
        CSR_COUNTS["uploads"] += 1
        return CsrIndex(indptr, sel, nrows, int(num_blocks))

    def csr_for_step(self, metadata, num_blocks: int) -> "CsrIndex":
        """Memoized CSR metadata: one build/upload per step, shared by all layers."""
        key = self._step_key(metadata, num_blocks)
        if (
            key is not None
            and self._csr is not None
            and self._csr_md is metadata
            and self._csr.key == key
        ):
            CSR_COUNTS["index_hits"] += 1
            return self._csr
        CSR_COUNTS["index_misses"] += 1
        idx = self._build_csr_index(
            metadata.block_table, self._blocks_per_req(metadata), num_blocks
        )
        idx.key = key
        self._csr, self._csr_md = idx, metadata
        return idx

    def gather_csr_payload(self, index: "CsrIndex", kv_cache, kv_scales,
                           nrows: int | None = None) -> GatheredKV:
        """Per-layer KV select driven by the shared CSR index (36x/step)."""
        CSR_COUNTS["payload_gathers"] += 1
        out = self.reserve(int(kv_cache.shape[0]))
        bs = self.layout.block_size
        hk = self.layout.num_kv_heads
        k_pb = self.layout.k_packed_bytes
        v_pb = self.layout.v_packed_bytes
        if nrows is None:
            nrows = index.nrows
        if nrows and nrows > 0:
            # The capture-safe device index keeps the FULL persistent `sel`
            # buffer; the eager index is already length nrows. Slice so
            # index_select only visits the live rows.
            #
            # Select straight INTO the reservation: the destination rows are
            # already (nrows, bs, hk, pb), so the intermediate tensor and the
            # second copy both go away. That matters beyond the saved copy -- an
            # allocation inside a captured region has to come from the graph pool,
            # and at 32k the large-batch capture graphs want 3 GiB per layer,
            # which invalidated the capture (cudaErrorStreamCaptureInvalidated)
            # when this path built temporaries per layer.
            # Select straight into the reservation's rows: at long context the
            # intermediate is ~1.4 GB per tensor, and a capture cannot grow the
            # pool (cudaErrorStreamCaptureUnsupported), so this path must not
            # allocate at all.
            # Select straight into the reservation's rows: at long context the
            # intermediate is ~1.4 GB per tensor, and a capture cannot grow the
            # pool, so this path must not allocate at all.
            sel = index.sel[:nrows]
            torch.index_select(self.layout.k_codes(kv_cache), 0, sel,
                               out=out.k_packed.view(-1, bs, hk, k_pb)[:nrows])
            torch.index_select(self.layout.v_codes(kv_cache), 0, sel,
                               out=out.v_packed.view(-1, bs, hk, v_pb)[:nrows])
            torch.index_select(self.layout.k_norm(kv_scales), 0, sel,
                               out=out.k_norm.view(-1, bs, hk)[:nrows])
            torch.index_select(self.layout.v_norm(kv_scales), 0, sel,
                               out=out.v_norm.view(-1, bs, hk)[:nrows])
        if env_flag("THUNDER_DEBUG_GATHER"):
            print(f"[TQ-CSR] nrows={nrows} capacity={self.page_rows}", flush=True)
        return out

    def gather_csr(
        self,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_scales: torch.Tensor,
        blocks_per_req: list[int],
    ) -> GatheredKV:
        """Non-memoized CSR gather (tests/probes): build index then select."""
        k_codes = self.layout.k_codes(kv_cache)
        v_codes = self.layout.v_codes(kv_cache)
        kn = self.layout.k_norm(kv_scales)
        vn = self.layout.v_norm(kv_scales)
        nb = min(int(k_codes.shape[0]), int(v_codes.shape[0]),
                 int(kn.shape[0]), int(vn.shape[0]))
        idx = self._build_csr_index(block_table, blocks_per_req, nb)
        return self.gather_csr_payload(idx, kv_cache, kv_scales)

    @property
    def indptr(self) -> torch.Tensor | None:
        return getattr(self, "_indptr", None)

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
        # Same clamp as the in-place path: padding / post-capture warm-up block
        # ids can exceed the cache and would otherwise fault the advanced index.
        bt = bt.clamp(0, max(int(kv_cache.shape[0]) - 1, 0))
        if env_flag("THUNDER_DEBUG_LAYOUT"):
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
        if env_flag("THUNDER_DEBUG_LAYOUT"):
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
    max_blocks_per_req: int | None = None,
    device: torch.device | str = "cuda",
) -> PagedKVManager:
    """Convenience constructor deriving the worst-case block count.

    ``max_blocks_per_req`` overrides the derived ``ceil(max_model_len /
    block_size)``. Pass it from the engine's own block-table width: vLLM pads
    that table (e.g. 260 -> 264 blocks for a 4160-token limit), and a manager
    sized only from ``max_model_len`` then rejects the table it is handed.
    """
    if max_blocks_per_req is not None:
        blocks = int(max_blocks_per_req)
    else:
        # vLLM pads its block table to a multiple of 8 (260 blocks -> 264 columns
        # for a 4160-token limit), so round the reservation up to that
        # granularity: sizing it from the token limit alone makes the manager
        # reject the table the engine hands over.
        blocks = (int(max_model_len) + layout.block_size - 1) // layout.block_size
        blocks = -(-blocks // 8) * 8
    return PagedKVManager(
        layout,
        max_num_reqs=max_num_reqs,
        max_blocks_per_req=blocks,
        device=device,
    )


# ---------------------------------------------------------------------------
# Option A (direct paged KV) addressing reference.
#
# The kernel will read the paged cache through the block table instead of a
# gathered copy: for request ``req`` and token ``t`` the physical cache row is
# ``block_table[req, t // block_size] * block_size + (t % block_size)``. A
# ``tile_n`` tile therefore spans ``tile_n // block_size`` physical blocks. This
# reference is what the CPU test compares against the CSR gather's ``sel``; the
# in-kernel version must reproduce it exactly.
# ---------------------------------------------------------------------------
def direct_paged_tile_rows(
    block_table: torch.Tensor,
    req: int,
    nt: int,
    tile_n: int,
    block_size: int,
) -> torch.Tensor:
    """Physical cache rows (block*bs + offset) for tile ``nt`` of ``req``."""
    tokens = nt * int(tile_n) + torch.arange(int(tile_n), device=block_table.device)
    blk = block_table[int(req), tokens // int(block_size)].to(torch.int64)
    off = tokens % int(block_size)
    return blk * int(block_size) + off


def _dump_csr_counts() -> None:
    """Emit the CSR/gather counters with the system fingerprint (THUNDER_DIAG)."""
    if env_flag("THUNDER_DIAG"):
        from thunder_vllm.utils.telemetry import emit as _emit_telemetry
        _emit_telemetry("csr", {"counters": dict(CSR_COUNTS)})


import atexit as _atexit

_atexit.register(_dump_csr_counts)
