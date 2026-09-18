"""Attention metadata for the Thunder-CuTe backend.

vLLM ``main`` replaced the old ``get_metadata_cls()`` + classmethod ``build()``
contract with a *builder* class: ``AttentionBackend.get_builder_cls()`` returns
an :class:`vllm.v1.attention.backend.AttentionMetadataBuilder`, whose ``build``
turns ``CommonAttentionMetadata`` into the backend's metadata dataclass.

The dataclass is importable without vLLM so CPU tests and the benchmark
harness can construct it directly; the builder only materialises when vLLM is
present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from thunder_vllm.utils.logging import get_logger

logger = get_logger("attention.metadata")

try:  # pragma: no cover - CPU-only machines have no vLLM
    from vllm.v1.attention.backend import (  # type: ignore
        AttentionCGSupport,
        AttentionMetadata,
        AttentionMetadataBuilder,
        CommonAttentionMetadata,
    )

    _HAS_VLLM = True
except Exception:  # noqa: BLE001
    _HAS_VLLM = False

    class AttentionCGSupport:  # type: ignore[no-redef]
        ALWAYS = 3
        UNIFORM_BATCH = 2
        UNIFORM_SINGLE_TOKEN_DECODE = 1
        NEVER = 0

    class AttentionMetadata:  # type: ignore[no-redef]
        pass

    class AttentionMetadataBuilder:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **k: Any) -> None:
            self.kv_cache_spec = a[0] if a else None

        def __class_getitem__(cls, item: Any) -> Any:  # noqa: N805
            return cls

    class CommonAttentionMetadata:  # type: ignore[no-redef]
        pass


@dataclass
class ThunderMetadata(AttentionMetadata):
    """Everything the CuTe kernel needs to resolve its KV tiles.

    Mirrors ``FlashAttentionMetadata`` in vLLM: a flat token axis
    (``query_start_loc``) plus a paged-KV descriptor (``block_table``,
    ``slot_mapping``). The kernel itself consumes the *gathered* contiguous
    buffers; this dataclass is what the gather layer and the launcher read.
    """

    # Request-level
    seq_lens: torch.Tensor  # (num_reqs,) total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) cache slot per token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) cu_seqlens for queries

    # Shapes
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_prefill_seq_len: int = 0
    max_decode_seq_len: int = 0
    max_seq_len: int = 0

    # Engine capacities (not the current batch). The gather buffers must be
    # reserved for these, because CUDA-graph capture can run the first forward
    # at batch size 1 and a later forward at full batch: sizing to the current
    # batch reserves (1, blocks) and the next call fails with "block_table
    # (1024, 32) exceeds reserved (1, 32)".
    max_num_reqs_capacity: int = 0
    max_model_len_capacity: int = 0

    # Split/batch structure
    num_decodes: int = 0  # decode requests occupy the first num_decodes slots
    is_prefill: bool = False

    # CUDA-graph
    use_cuda_graph: bool = False
    cudagraph_support: ClassVar[int] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    # CPU-resident mirrors (avoid per-step D2H syncs in the launcher).
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None

    @property
    def num_reqs(self) -> int:
        return int(self.query_start_loc.shape[0]) - 1

    @property
    def max_blocks_per_req(self) -> int:
        return int(self.block_table.shape[1])

    def token_slice(self, req: int) -> slice:
        lo = int(self.query_start_loc_cpu[req]) if self.query_start_loc_cpu is not None else int(
            self.query_start_loc[req]
        )
        hi = (
            int(self.query_start_loc_cpu[req + 1])
            if self.query_start_loc_cpu is not None
            else int(self.query_start_loc[req + 1])
        )
        return slice(lo, hi)


class ThunderMetadataBuilder(AttentionMetadataBuilder[ThunderMetadata]):
    """Builds :class:`ThunderMetadata` from ``CommonAttentionMetadata``.

    ``_cudagraph_support`` is ``UNIFORM_SINGLE_TOKEN_DECODE``: vLLM's FULL graphs
    capture single-token decode (attention included), which is what the gather
    buffers are sized for. PIECEWISE (prefill) always breaks at attention, so
    prefill attention runs eager regardless of this value -- see the TTFT note in
    the session log: the CuTeDSL per-launch host cost is the prefill bottleneck.
    """

    _cudagraph_support: ClassVar[int] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
    ) -> None:
        if _HAS_VLLM:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = list(layer_names)
        self.vllm_config = vllm_config
        self.device = device
        # Engine capacities for the gather reservation (see ThunderMetadata).
        self._cap_num_reqs = 0
        self._cap_model_len = 0
        try:
            self._cap_num_reqs = int(
                getattr(vllm_config.scheduler_config, "max_num_seqs", 0) or 0
            )
            self._cap_model_len = int(
                getattr(vllm_config.model_config, "max_model_len", 0) or 0
            )
        except Exception:  # noqa: BLE001
            pass
        # Decodes first, threshold 1 token == a decode.
        init = getattr(self, "_init_reorder_batch_threshold", None)
        if init is not None:
            init(1, supports_spec_as_decode=False)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> ThunderMetadata:
        cam = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            _split_decodes_and_prefills(cam)
        )
        max_query_len = int(getattr(cam, "max_query_len", 0) or 0)
        max_seq_len = int(getattr(cam, "max_seq_len", 0) or 0)
        seq_lens = getattr(cam, "seq_lens", None)
        return ThunderMetadata(
            seq_lens=seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_decodes=num_decodes,
            num_actual_tokens=int(getattr(cam, "num_actual_tokens", 0) or 0),
            max_query_len=max_query_len,
            max_prefill_seq_len=max_seq_len if num_prefills else 0,
            max_decode_seq_len=max_seq_len if num_decodes else 0,
            max_seq_len=max_seq_len,
            max_num_reqs_capacity=self._cap_num_reqs,
            max_model_len_capacity=self._cap_model_len,
            is_prefill=max_query_len > 1,
            use_cuda_graph=False,
            query_start_loc_cpu=getattr(cam, "query_start_loc_cpu", None),
            seq_lens_cpu=getattr(cam, "seq_lens_cpu_upper_bound", None),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata: Any) -> ThunderMetadata:
        """Capture-time metadata with placeholder seq_lens.

        The graph is replayed with real ``seq_lens`` written into the same
        buffers, so capture can use length 1 for every request and stay fast.
        """
        md = self.build(0, common_attn_metadata)
        md.use_cuda_graph = True
        if md.seq_lens is not None:
            md.seq_lens.fill_(1)
        return md


def _split_decodes_and_prefills(cam: Any) -> tuple[int, int, int, int]:
    """Best-effort decode/prefill partition without a GPU sync."""
    if _HAS_VLLM:
        try:
            from vllm.v1.attention.backends.utils import (  # type: ignore
                split_decodes_and_prefills,
            )

            out = split_decodes_and_prefills(cam, decode_threshold=1)
            if isinstance(out, tuple) and len(out) == 4:
                return int(out[0]), int(out[1]), int(out[2]), int(out[3])
        except Exception:  # noqa: BLE001
            pass
    max_query_len = int(getattr(cam, "max_query_len", 0) or 0)
    if max_query_len <= 1:
        n = int(getattr(cam, "num_reqs", getattr(cam, "num_actual_tokens", 0)) or 0)
        return n, 0, int(getattr(cam, "num_actual_tokens", 0) or 0), 0
    num_actual = int(getattr(cam, "num_actual_tokens", 0) or 0)
    return 0, num_actual, 0, num_actual


def get_builder_cls() -> type[ThunderMetadataBuilder]:
    return ThunderMetadataBuilder


__all__ = [
    "ThunderMetadata",
    "ThunderMetadataBuilder",
    "get_builder_cls",
]
