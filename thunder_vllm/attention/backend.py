"""vLLM attention backend + impl for the fused TurboQuant CuTeDSL kernel.

Registered as ``THUNDER_CUTE`` through ``AttentionBackendEnum.CUSTOM`` (see
``model/registry.py``), deliberately *not* as ``TURBOQUANT``: upstream vLLM
already ships a ``ThunderAttentionBackend`` under that name, and we need it
intact to benchmark against.

Only the plumbing lives here; the kernel itself is in ``cute_kernel.py`` and is
imported lazily so that this module is safe to import on a CPU-only host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from thunder_vllm.attention.cache_layout import (
    ThunderCacheLayout,
    allocate_kv_cache,
    reshape_and_cache,
)
from thunder_vllm.attention.metadata import (
    ThunderMetadata,
    ThunderMetadataBuilder,
)
from thunder_vllm.attention.paged_kv import PagedKVManager, make_paged_kv_manager
from thunder_vllm.attention.scratch import ScratchState, new_scratch, reserve_scratch
from thunder_vllm.utils.logging import env_flag, get_logger, log_once

logger = get_logger("attention.backend")

_ENGINE_HOOK = {"done": False}
_PAGED_CACHE: dict = {}
_CSR_DEBUG = {"done": False}

# Capture audit: count forward invocations by (capturing?, max_query_len). If the
# decode step is captured, forward is NOT called per decode step during replay.
import collections as _collections
import atexit as _atexit

_FWD_COUNT: "_collections.Counter" = _collections.Counter()


def _dump_fwd_count() -> None:
    if env_flag("THUNDER_COUNT"):
        print(f"[TQ-COUNT] {dict(_FWD_COUNT)}", flush=True)


_atexit.register(_dump_fwd_count)

try:  # pragma: no cover - CPU-only machines have no vLLM
    from vllm.v1.attention.backend import (  # type: ignore
        AttentionBackend,
        AttentionImplBase,
        AttentionType,
    )

    _HAS_VLLM = True
except Exception:  # noqa: BLE001
    _HAS_VLLM = False

    class AttentionBackend:  # type: ignore[no-redef]
        """Fallback shim so this module imports without vLLM."""

        accept_output_buffer: bool = True

        def __class_getitem__(cls, item: Any) -> Any:  # noqa: N805
            return cls

    class AttentionImplBase:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> Any:  # noqa: N805
            return cls

    class AttentionType:  # type: ignore[no-redef]
        DECODER = "decoder"


BACKEND_NAME = "THUNDER_CUTE"

# Name that must be returned by ``get_name()``. vLLM resolves the backend name
# through ``AttentionBackendEnum[name]``, and ``register_backend`` keys the
# override by the ENUM MEMBER, so the resolvable name is "CUSTOM" -- our display
# name "THUNDER_CUTE" is not an enum member and makes the engine fail with
#   ValueError: Unknown attention backend: 'THUNDER_CUTE'.
REGISTRY_NAME = "CUSTOM"
SUPPORTED_HEAD_SIZES = (64, 128, 256)
SUPPORTED_BLOCK_SIZES = (16, 32, 64, 128)


@dataclass(frozen=True)
class ThunderCuteConfig:
    """Plugin-level knobs that vLLM has no slot for.

    Upstream vLLM encodes the TurboQuant preset in ``kv_cache_dtype`` (a
    ``Literal`` that a plugin cannot extend). This plugin therefore reads its
    bit widths from the environment, which keeps it installable and testable
    without patching vLLM's type stubs, and still lets a deployment pin them.
    """

    k_bits: int = 4
    v_bits: int = 4
    num_stages: int = 2
    num_dequant_stages: int = 2
    num_threads: int = 128
    m_block_size: int = 64
    n_block_size: int = 64
    q_stage: int = 2
    use_2cta_instrs: bool = False
    # Paged-cache block size. Distinct from n_block_size, which is the kernel's
    # KV tile width.
    cache_block_size: int = 16
    # Kernel fast paths validated by the prefill A/Bs. Off by default: each is an
    # isolated, correctness-checked optimization that should be turned on
    # together only after the end-to-end regression suite is green.
    #   onepass      single-pass online softmax (removes PASS1's K dequant + QK)
    #   reg_rescale  register-local accumulator rescale (drops 2 SMEM trips/tile)
    #   causal_bound skip fully-masked causal KV tiles (requires is_causal)
    onepass: bool = False
    reg_rescale: bool = False
    causal_bound: bool = False

    @classmethod
    def from_env(cls, kv_cache_dtype: str | None = None) -> ThunderCuteConfig:
        k_bits, v_bits = _bits_from_kv_cache_dtype(kv_cache_dtype)
        env = os.environ

        def _flag(name: str, default: bool = False) -> bool:
            val = env.get(name)
            if val is None:
                return default
            return val.strip().lower() not in ("", "0", "false", "no", "off")

        return cls(
            k_bits=int(env.get("THUNDER_K_BITS", k_bits)),
            v_bits=int(env.get("THUNDER_V_BITS", v_bits)),
            num_stages=int(env.get("THUNDER_NUM_STAGES", 2)),
            num_dequant_stages=int(env.get("THUNDER_NUM_DEQUANT_STAGES", 2)),
            num_threads=int(env.get("THUNDER_NUM_THREADS", 128)),
            m_block_size=int(env.get("THUNDER_M_BLOCK", 64)),
            n_block_size=int(env.get("THUNDER_N_BLOCK", 64)),
            q_stage=int(env.get("THUNDER_Q_STAGE", 2)),
            use_2cta_instrs=_flag("THUNDER_USE_2CTA"),
            onepass=_flag("THUNDER_ONEPASS"),
            reg_rescale=_flag("THUNDER_REG_RESCALE"),
            causal_bound=_flag("THUNDER_CAUSAL_BOUND"),
        )

    def kernel_key(self, head_dim: int, num_kv_heads: int, is_causal: bool) -> tuple:
        return (
            head_dim,
            self.k_bits,
            self.v_bits,
            num_kv_heads,
            is_causal,
            self.m_block_size,
            self.n_block_size,
            self.num_stages,
            self.num_dequant_stages,
            self.num_threads,
            self.use_2cta_instrs,
            self.q_stage,
            self.cache_block_size,
            self.onepass,
            self.reg_rescale,
            self.causal_bound,
        )


def _kernel_module():
    """Pick the kernel schedule: ``mma`` (default) or ``tcgen05``.

    Selected with ``THUNDER_SCHEDULE``. The tcgen05 module is a drop-in for
    the same call contract, so only the import site changes.
    """
    which = os.environ.get("THUNDER_SCHEDULE", "mma").lower()
    if which in ("tcgen05", "tmem"):
        raise NotImplementedError(
            "the tcgen05 schedule is not finished (path A in progress). It needs "
            "FA's precomputed-descriptor plumbing (declare_ptx_smem_desc + "
            "gemm_ptx_precomputed_varname) to drive the SMEMxSMEM MMA; the "
            "vendored helpers are in place and the TMEM/mbarrier/readback parts "
            "are API-correct. See the module docstring of "
            "thunder_vllm/attention/cute_kernel_tcgen05.py and "
            "ci_probe/results/probe_summary.md. Use THUNDER_SCHEDULE=mma."
        )
    from thunder_vllm.attention import cute_kernel as mod
    return mod


def _bits_from_kv_cache_dtype(kv_cache_dtype: str | None) -> tuple[int, int]:
    """Best-effort parse of a ``thunder_*`` cache-dtype string."""
    if not kv_cache_dtype:
        return 4, 4
    s = kv_cache_dtype.lower()
    if "k3v4" in s:
        return 3, 4
    if "k8v4" in s:
        return 8, 4
    if "3bit" in s:
        return 3, 3
    if "4bit" in s:
        return 4, 4
    return 4, 4


class ThunderAttentionBackend(AttentionBackend):
    """Backend registration surface for ``THUNDER_CUTE``."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "thunder_cute",
        "thunder_k8v4",
        "thunder_k3v4_nc",
        "thunder_4bit_nc",
        "thunder_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return REGISTRY_NAME

    @staticmethod
    def get_impl_cls():
        return ThunderAttentionImpl

    @staticmethod
    def get_builder_cls():
        return ThunderMetadataBuilder

    # -- capability queries ------------------------------------------------
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return list(SUPPORTED_HEAD_SIZES)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return list(SUPPORTED_BLOCK_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        # vLLM asks with None to mean "backend decides" (e.g. a CUSTOM backend
        # with no preset), and `None in (16,32,64,128)` is False, which made the
        # engine reject the backend outright with
        #   ValueError: ... not valid for this configuration. Reason:
        #   ['block_size not supported']
        # Accept None and let the kernel's own cache_block_size govern.
        result = True if block_size is None else block_size in SUPPORTED_BLOCK_SIZES
        if env_flag("THUNDER_DEBUG_BLOCK"):
            logger.info(
                "supports_block_size(%r) -> %s (supported=%s)",
                block_size, result, SUPPORTED_BLOCK_SIZES,
            )
        return result

    @classmethod
    def supports_compute_capability(cls, capability: Any) -> bool:
        """SM100 (``cc[0] == 10``) or SM110 (``cc[0] == 11``)."""
        try:
            major = int(capability[0])
        except Exception:  # noqa: BLE001
            major = int(getattr(capability, "major", -1))
        return major in (10, 11)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in SUPPORTED_HEAD_SIZES

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in cls.supported_dtypes

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: Any) -> bool:
        if kv_cache_dtype is None:
            return True
        s = str(kv_cache_dtype)
        return s == "auto" or s.startswith("thunder")

    @classmethod
    def supports_attn_type(cls, attn_type: Any) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    # -- cache layout ------------------------------------------------------
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Combined packed-KV slot shape: ``(blocks, block_size, slot_bytes)``."""
        cfg = ThunderCuteConfig.from_env()
        layout = ThunderCacheLayout(
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            k_bits=cfg.k_bits,
            v_bits=cfg.v_bits,
            block_size=block_size,
        )
        shape = layout.get_kv_cache_shape(num_blocks)
        if env_flag("THUNDER_DEBUG_LAYOUT"):
            logger.info(
                "get_kv_cache_shape(num_blocks=%d, block_size=%d, num_kv_heads=%d, "
                "head_size=%d) -> %s",
                num_blocks, block_size, num_kv_heads, head_size, tuple(shape),
            )
        return shape

    @classmethod
    def customize_spec(cls, spec: Any) -> Any:
        """Tell vLLM how many bytes one cached position occupies.

        Upstream TurboQuant does the same thing for its ``state_content_bytes``;
        without it the KV-cache manager sizes blocks for fp16 K+V and the
        packed slot would overrun the block.
        """
        # MUST override, not defer. vLLM pre-populates state_content_bytes with
        # the standard fp16 K+V size (2 * head_size * 2 per position), and
        # returning early there leaves the packed slot unsized: the engine then
        # allocates a standard cache, e.g. observed
        #   cache=(30582, 8, 16, 512)      # 512 B per (block, head, position)
        # while our layout expects (num_blocks, 8, 16, 128). v_codes then read
        # 512-64 = 448 bytes instead of 64 -- a 7x over-read -- which is exactly
        # the factor the gather reshape failed on.
        prev = getattr(spec, "state_content_bytes", None)
        try:
            cfg = ThunderCuteConfig.from_env()
            layout = ThunderCacheLayout(
                num_kv_heads=spec.num_kv_heads,
                head_dim=spec.head_size,
                k_bits=cfg.k_bits,
                v_bits=cfg.v_bits,
                block_size=spec.block_size,
            )
            # vLLM's per-position inner dim = state_content_bytes / dtype.itemsize.
            # The cache holds packed BYTES, so declare a uint8 spec dtype and let
            # state_content_bytes be the byte slot directly (the old 2x only made
            # sense for an fp16 cache). Without this vLLM allocates fp16 and its
            # store does index_put Half<-Byte.
            packed = layout.head_slot_bytes
            updates: dict = {"state_content_bytes": packed}
            if hasattr(spec, "dtype") and spec.dtype != torch.uint8:
                updates["dtype"] = torch.uint8
            if prev is not None and int(prev) != packed:
                logger.info(
                    "customize_spec: state_content_bytes %s -> %s, dtype -> uint8",
                    prev, packed,
                )
            return replace(spec, **updates)
        except Exception as exc:  # noqa: BLE001
            log_once(logger, "customize_spec skipped: %s", exc)
            return spec

    # -- block management --------------------------------------------------
    @staticmethod
    def swap_blocks(
        src_kv_cache: Any,
        dst_kv_cache: Any,
        src_to_dst: Any,
    ) -> None:
        """Swap whole cache blocks. Operates on the combined slot, so both K
        and V codes and both norms move together with one copy per block."""
        # Works for the single-tensor combined slot produced by
        # `allocate_kv_cache`. Indices are block ids.
        if isinstance(src_kv_cache, (tuple, list)):
            for src, dst in zip(src_kv_cache, dst_kv_cache, strict=False):
                _swap_blocks_one(src, dst, src_to_dst)
        else:
            _swap_blocks_one(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(kv_caches: Any, src_to_dists: Any) -> None:
        for src, dst in src_to_dists.items() if isinstance(src_to_dists, dict) else []:
            _copy_blocks_one(kv_caches, src, dst)


def _swap_blocks_one(src: torch.Tensor, dst: torch.Tensor, src_to_dst: Any) -> None:
    if src.numel() == 0:
        return
    if hasattr(src_to_dst, "shape") and src_to_dst.numel():
        pairs = src_to_dst.reshape(-1, 2)
        for s, d in pairs.tolist():
            dst[int(d)].copy_(src[int(s)])
    else:
        # dict {src: dst}
        for s, d in src_to_dst.items():
            dst[int(d)].copy_(src[int(s)]) if hasattr(dst, "__setitem__") else None


def _copy_blocks_one(kv_caches: Any, src: Any, dst: Any) -> None:
    for cache in kv_caches if isinstance(kv_caches, (tuple, list)) else [kv_caches]:
        cache[dst].copy_(cache[src])


class ThunderAttentionImpl(AttentionImplBase):
    """Owns the compiled kernel, the gather buffers, and the scratch pools."""

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: Any = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        K_BITS: int | None = None,
        V_BITS: int | None = None,
        **kwargs: Any,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("THUNDER_CUTE does not support alibi")
        if logits_soft_cap:
            raise NotImplementedError("THUNDER_CUTE does not support logits_soft_cap")

        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads if num_kv_heads is not None else num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.attn_type = attn_type

        cfg = ThunderCuteConfig.from_env(kv_cache_dtype)
        if K_BITS is not None or V_BITS is not None:
            cfg = replace(
                cfg,
                k_bits=int(K_BITS if K_BITS is not None else cfg.k_bits),
                v_bits=int(V_BITS if V_BITS is not None else cfg.v_bits),
            )
        self.cfg = cfg

        self.layout = ThunderCacheLayout(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            k_bits=cfg.k_bits,
            v_bits=cfg.v_bits,
            block_size=cfg.cache_block_size,
        )

        self._kernels: dict[tuple, Any] = {}
        self._quantizer: Any = None
        self._paged: PagedKVManager | None = None
        self._scratch: ScratchState = new_scratch()
        self._max_capture_tokens = 0
        self._buffers_registered = False

    # ------------------------------------------------------------------ #
    # Lazy construction (must happen before graph capture)
    # ------------------------------------------------------------------ #
    def _ensure_quantizer(self, device: torch.device) -> Any:
        if self._quantizer is None:
            from thunder_vllm.quant.quantizer import ThunderQuantizer

            self._quantizer = ThunderQuantizer(
                self.head_size,
                self.cfg.k_bits,
                self.cfg.v_bits,
                head_dim_padded=self.layout.head_dim_padded,
                device=device,
            )
        return self._quantizer

    def _ensure_paged(
        self,
        device: torch.device,
        *,
        max_num_reqs: int,
        max_model_len: int,
    ) -> PagedKVManager:
        dev = torch.device(device)
        key = (dev.type, dev.index, self.layout.block_size, self.layout.num_kv_heads,
               self.cfg.k_bits, self.cfg.v_bits, int(max_num_reqs), int(max_model_len))
        mgr = _PAGED_CACHE.get(key)
        if mgr is None:
            mgr = make_paged_kv_manager(
                self.layout, max_num_reqs=max_num_reqs,
                max_model_len=max_model_len, device=device,
            )
            # Allocation deferred to the first gather so the indirect path can
            # cap it by the physical block count.
            _PAGED_CACHE[key] = mgr
            _CSR_DEBUG["manager_create"] = _CSR_DEBUG.get("manager_create", 0) + 1
        self._paged = mgr
        return self._paged

    def get_kernel(self, head_dim: int, is_causal: bool) -> Any:
        """Compile (once) and cache the kernel for this shape."""
        key = self.cfg.kernel_key(head_dim, self.num_kv_heads, is_causal)
        kernel = self._kernels.get(key)
        if kernel is None:
            mod = _kernel_module()
            kernel = mod.ThunderAttentionForward(
                head_dim=head_dim,
                K_BITS=self.cfg.k_bits,
                V_BITS=self.cfg.v_bits,
                qhead_per_kvhead=self.num_kv_groups,
                is_causal=is_causal,
                m_block_size=self.cfg.m_block_size,
                n_block_size=self.cfg.n_block_size,
                num_stages=self.cfg.num_stages,
                num_dequant_stages=self.cfg.num_dequant_stages,
                num_threads=self.cfg.num_threads,
                use_2cta_instrs=self.cfg.use_2cta_instrs,
                q_stage=self.cfg.q_stage,
            )
            self._kernels[key] = kernel
        return kernel

    def warmup(
        self,
        device: torch.device | str,
        *,
        max_num_reqs: int,
        max_model_len: int,
        max_tokens: int,
        is_causal: bool = True,
    ) -> None:
        """Pre-compile the kernel, reserve gather buffers and scratch.

        Call before ``torch.cuda.graphs.CUDAGraph`` capture: it is the only
        place allowed to allocate.
        """
        dev = torch.device(device)
        self._ensure_quantizer(dev)
        self._ensure_paged(
            dev, max_num_reqs=max_num_reqs, max_model_len=max_model_len
        )
        reserve_scratch(
            max_tokens=max_tokens,
            num_heads=self.num_heads,
            head_dim=self.layout.head_dim_padded,
            device=dev,
            scratch=self._scratch,
        )
        self._max_capture_tokens = max(self._max_capture_tokens, int(max_tokens))
        # Touch the kernel so compilation happens now, not during capture.
        self.get_kernel(self.head_size, is_causal)

    # ------------------------------------------------------------------ #
    # Executable surface
    # ------------------------------------------------------------------ #
    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ThunderMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if env_flag("THUNDER_COUNT") and attn_metadata is not None:
            _cap = bool(torch.cuda.is_current_stream_capturing())
            _mql = int(getattr(attn_metadata, "max_query_len", 0) or 0)
            _FWD_COUNT[(f"cap={int(_cap)}", f"mql={_mql}")] += 1
        if output is None:
            output = torch.empty(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            return output.zero_()

        n = attn_metadata.num_actual_tokens or num_tokens
        if n <= 0:
            return output.zero_()

        # Diagnostic only: skip the entire TurboQuant forward (no gather, no
        # cache write, no launch) so a capture-path fault can be attributed to
        # this backend vs vLLM. Set THUNDER_SKIP_BACKEND=1.
        if os.environ.get("THUNDER_SKIP_BACKEND", "0").strip().lower() not in (
            "", "0", "false", "no", "off"
        ):
            return output.zero_()

        # Validate what the launch path dereferences. Without this, a missing or
        # oddly-shaped field during vLLM's warm-up surfaces as
        # CUDA_ERROR_ILLEGAL_ADDRESS (700) inside run_compiled_program instead of
        # a clear message.
        for _name in ("seq_lens", "query_start_loc", "block_table", "slot_mapping"):
            _v = getattr(attn_metadata, _name, None)
            if _v is None:
                raise RuntimeError(
                    f"ThunderAttentionImpl.forward: attn_metadata.{_name} is None; "
                    f"metadata type={type(attn_metadata).__name__} "
                    f"num_actual_tokens={getattr(attn_metadata, 'num_actual_tokens', None)} "
                    f"num_reqs={getattr(attn_metadata, 'num_reqs', None)}"
                )
            if not torch.is_tensor(_v):
                raise RuntimeError(
                    f"ThunderAttentionImpl.forward: attn_metadata.{_name} is "
                    f"{type(_v).__name__}, expected a tensor"
                )
            if not _v.is_cuda:
                raise RuntimeError(
                    f"ThunderAttentionImpl.forward: attn_metadata.{_name} is on "
                    f"{_v.device}, expected CUDA"
                )
        if env_flag("THUNDER_DEBUG_LAUNCH"):
            print(
                f"[TQ-LAUNCH] q={tuple(query.shape)}/{query.dtype} n={n} "
                f"kv={tuple(kv_cache.shape)} "
                f"bt={tuple(attn_metadata.block_table.shape)} "
                f"sl={tuple(attn_metadata.seq_lens.shape)} "
                f"qsl={tuple(attn_metadata.query_start_loc.shape)} "
                f"slot={tuple(attn_metadata.slot_mapping.shape)} "
                f"num_reqs={attn_metadata.num_reqs} "
                f"max_blocks_per_req={attn_metadata.max_blocks_per_req} "
                f"is_prefill={attn_metadata.is_prefill} "
                f"max_query_len={attn_metadata.max_query_len} "
                f"num_heads={self.num_heads} Hk={self.num_kv_heads} "
                f"head_size={self.head_size}",
                flush=True,
            )

        is_causal = attn_metadata.is_prefill or attn_metadata.max_query_len > 1
        kernel = self.get_kernel(self.head_size, is_causal)

        # The KV-cache write is a separate op in vLLM main; if the runner has
        # not done it yet, do it here.
        if not getattr(layer, "_tq_cache_updated", False):
            quantizer = self._ensure_quantizer(query.device)
            reshape_and_cache(
                key[:n].reshape(n, self.num_kv_heads, self.head_size),
                value[:n].reshape(n, self.num_kv_heads, self.head_size),
                attn_metadata.slot_mapping[:n],
                kv_cache,
                self._scales_for(kv_cache),
                quantizer,
                self.layout,
            )

        # Reserve for the ENGINE capacity, not this batch: under CUDA-graph
        # capture the first forward can be batch 1 and a later one full batch,
        # and the reservation must already cover the latter (a resize after
        # capture would invalidate the captured pointers).
        _cap_reqs = int(getattr(attn_metadata, "max_num_reqs_capacity", 0) or 0)
        _cap_len = int(getattr(attn_metadata, "max_model_len_capacity", 0) or 0)
        if _cap_reqs <= 0:
            _cap_reqs = int(attn_metadata.num_reqs)
        if _cap_len <= 0:
            _cap_len = int(attn_metadata.max_blocks_per_req) * self.layout.block_size
        # Engine-path profiling/capture: only on the first true single-request
        # decode step (max_query_len == 1, num_reqs == 1), where the oracle is
        # cheap and unambiguous. Times each stage once.
        _cap = (
            env_flag("THUNDER_ENGINE_HOOK")
            and not _ENGINE_HOOK["done"]
            and int(getattr(attn_metadata, "max_query_len", 0) or 0) == 1
            and int(getattr(attn_metadata, "num_reqs", 0) or 0) == 1
        )
        if _cap:
            _ev0 = torch.cuda.Event(enable_timing=True)
            _ev0.record()
        paged = self._ensure_paged(
            query.device,
            max_num_reqs=max(_cap_reqs, 1),
            max_model_len=_cap_len,
        )
        # Live-block count for the gather must be known on the HOST: under
        # CUDA-graph capture a `.item()` on the device seq_lens is a D2H sync and
        # is illegal. vLLM keeps a CPU mirror; use it when present (it may be an
        # upper bound, which only means a slightly larger gather, never a wrong
        # one), otherwise the gather falls back to the full table while
        # capturing and trims only in eager mode.
        live_blocks = None
        _sl_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if torch.is_tensor(_sl_cpu) and _sl_cpu.numel() > 0 and not _sl_cpu.is_cuda:
            _r = max(int(attn_metadata.num_reqs), 1)
            _bs = self.layout.block_size
            live_blocks = max(
                1,
                int(((_sl_cpu[:_r].to(torch.int64) + _bs - 1) // _bs).max().item()),
            )
        _indirect = (
            os.environ.get("THUNDER_8B_INDIRECT", "0").strip().lower()
            not in ("", "0", "false", "no", "off")
            and not torch.cuda.is_current_stream_capturing()
        )
        _indptr = None
        if os.environ.get("THUNDER_SKIP_GATHER", "0").strip().lower() not in (
            "", "0", "false", "no", "off"
        ):
            # Diagnostic: use the reserved buffers without the torch gather.
            gathered = paged.reserve()
        elif _indirect:
            # CSR packing: no request-major payload, capacity bounded by the
            # physical block count. Eager only for now.
            _sl = getattr(attn_metadata, "seq_lens_cpu", None)
            if not (torch.is_tensor(_sl) and _sl.numel() > 0 and not _sl.is_cuda):
                _sl = attn_metadata.seq_lens[: attn_metadata.block_table.shape[0]].cpu()
            _bs = self.layout.block_size
            _bpr = [
                max(0, int((int(s) + _bs - 1) // _bs)) for s in _sl.tolist()
            ]
            gathered = paged.gather_csr(
                attn_metadata.block_table, kv_cache, self._scales_for(kv_cache), _bpr
            )
            _indptr = paged.indptr
            if (
                os.environ.get("THUNDER_CSR_DUMP", "0").strip().lower()
                not in ("", "0", "false", "no", "off")
                and not _CSR_DEBUG["done"]
            ):
                try:
                    _bt = attn_metadata.block_table
                    _r0 = int(_bt.shape[0])
                    _nb = max(1, int((int(_sl[0]) + _bs - 1) // _bs))
                    _kc = self.layout.k_codes(kv_cache)
                    _phys = _bt[0, :_nb].to(torch.int64).clamp_(0, max(int(_kc.shape[0]) - 1, 0))
                    torch.save(
                        {
                            "r": _r0,
                            "blocks_req0": _nb,
                            "indptr": paged.indptr.detach().cpu(),
                            "gathered_k": gathered.k_packed[:_nb].detach().cpu(),
                            "cache_k_bt0": _kc[_phys].detach().cpu(),
                            "bs": _bs, "hk": int(self.num_kv_heads),
                        },
                        os.environ.get("THUNDER_CSR_DUMP_PATH", "/tmp/thunder_csr.pt"),
                    )
                    _CSR_DEBUG["done"] = True
                except Exception:  # noqa: BLE001
                    logger.exception("CSR dump failed")
        else:
            gathered = paged.gather_packed_tiles(
                attn_metadata.block_table,
                kv_cache,
                self._scales_for(kv_cache),
                attn_metadata.seq_lens,
                live_blocks=live_blocks,
            )

        if _cap:
            _ev1 = torch.cuda.Event(enable_timing=True)
            _ev1.record()
        q = query[:n].reshape(n, self.num_heads, self.head_size)
        o = output[:n].reshape(n, self.num_heads, self.head_size)

        from thunder_vllm.attention.cute_kernel import launch_thunder_attention

        quantizer = self._ensure_quantizer(query.device)

        # Rotation contract, applied HERE so the plugin is self-contained and the
        # model does not have to be patched. The kernel consumes Q in the rotated
        # basis (q_rot = q @ R) and produces O in that basis; the inverse is
        # applied to the output below. Without this, a stock vLLM model yields
        # scores in a different basis and results are silently wrong.
        _skip_rot = os.environ.get("THUNDER_SKIP_ROT", "0").strip().lower() not in (
            "", "0", "false", "no", "off")
        if not _skip_rot:
            q = (q.float() @ quantizer.rotation.matrix).to(q.dtype)
        if _cap:
            _ev2 = torch.cuda.Event(enable_timing=True)
            _ev2.record()
        launch_thunder_attention(
            kernel,
            q,
            gathered,
            o,
            attn_metadata,
            self.scale,
            quantizer=quantizer,
            num_splits=self._decode_split_count(attn_metadata, is_causal),
            onepass=self.cfg.onepass,
            reg_rescale=self.cfg.reg_rescale,
            causal_bound=self.cfg.causal_bound,
            indptr=_indptr,
            indirect=bool(_indirect),
        )
        if _cap:
            _ev3 = torch.cuda.Event(enable_timing=True)
            _ev3.record()
            _o_pre = o.detach().clone()

        # The kernel accumulates ``O_rot = P @ (R V) = R (P @ V)``: scores are
        # rotation invariant but the value contribution is not. Undo the
        # rotation once, on the flattened head axis. This is the plugin's
        # "final weight-absorbed output projection" GEMM -- a single linear
        # projection over the head dimension, not a second attention pass.
        if not _skip_rot:
            o.copy_(quantizer.rotation.inverse(o.float()).to(o.dtype))
        if _cap:
            _ev4 = torch.cuda.Event(enable_timing=True)
            _ev4.record()
            torch.cuda.synchronize()
            _times = {
                "gather_ms": _ev0.elapsed_time(_ev1),
                "qrot_ms": _ev1.elapsed_time(_ev2),
                "launch_ms": _ev2.elapsed_time(_ev3),
                "inverse_ms": _ev3.elapsed_time(_ev4),
                "total_ms": _ev0.elapsed_time(_ev4),
            }
            _ENGINE_HOOK["done"] = True
            try:
                torch.save(
                    {
                        "q_rot": q.detach().float().cpu(),
                        "k": gathered.k_packed.detach().cpu(),
                        "v": gathered.v_packed.detach().cpu(),
                        "kn": gathered.k_norm.detach().float().cpu(),
                        "vn": gathered.v_norm.detach().float().cpu(),
                        "o_pre": _o_pre.detach().float().cpu(),
                        "o_final": o.detach().float().cpu(),
                        "seq_lens": attn_metadata.seq_lens.detach().cpu(),
                        "block_table": attn_metadata.block_table.detach().cpu(),
                        "max_blocks_per_req": int(attn_metadata.max_blocks_per_req),
                        "hq": self.num_heads, "hk": self.num_kv_heads,
                        "hd": self.head_size, "bs": self.layout.block_size,
                        "is_causal": bool(is_causal),
                        "times": _times,
                    },
                    os.environ.get("THUNDER_HOOK_PATH", "/tmp/thunder_hook.pt"),
                )
            except Exception:  # noqa: BLE001
                logger.exception("THUNDER_ENGINE_HOOK dump failed")
        return output

    def _decode_split_count(self, attn_metadata: Any, is_causal: bool) -> int:
        """Split-K count for a decode step, or 1 when splitting would not help.

        Decode only (``is_causal`` is False and there is one query token per
        request); prefill keeps the baseline schedule. The count must be known on
        the host before the launch -- under CUDA-graph capture a device read is
        illegal -- so it comes from vLLM's CPU seq-len mirror when present, or a
        non-capturing sync, and otherwise falls back to no split. Override with
        ``THUNDER_SPLITS`` for experiments.
        """
        import os

        forced = os.environ.get("THUNDER_SPLITS")
        if forced:
            return max(1, int(forced))
        if is_causal or self.num_kv_groups <= 1:
            return 1
        n_reqs = max(int(attn_metadata.num_reqs), 1)
        sl_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if torch.is_tensor(sl_cpu) and sl_cpu.numel() > 0 and not sl_cpu.is_cuda:
            seq_len = int(sl_cpu[:n_reqs].max().item())
        elif not torch.cuda.is_current_stream_capturing():
            seq_len = int(attn_metadata.seq_lens[:n_reqs].max().item())
        else:
            return 1
        from thunder_vllm.attention.splits import choose_split_count

        return choose_split_count(
            seq_len, n_reqs, self.num_heads, tile_n=self.cfg.n_block_size
        )

    def do_kv_cache_update(self, *args: Any, **kwargs: Any) -> None:
        """vLLM's separate KV-cache write hook.

        vLLM 0.29 calls this via ``torch.ops.vllm.unified_kv_cache_update`` and
        asserts the impl exposes it:

            AssertionError: ThunderAttentionImpl does not support kv cache update

        The write itself is the same ``reshape_and_cache`` the forward path used
        to do inline, so the forward path now only has to handle the case where
        the runner has not called this.

        The positional convention is discovered rather than assumed: identify the
        cache (largest tensor), the slot mapping (1-D integer), and take
        (key, value) as the remaining two fp16 tensors in order. Set
        ``THUNDER_DEBUG_KV=1`` to log exactly what arrives.
        """
        if env_flag("THUNDER_DEBUG_KV"):
            logger.info(
                "do_kv_cache_update args=%s kwargs=%s",
                [(type(a).__name__, tuple(getattr(a, "shape", ()))) for a in args],
                {k: (type(v).__name__, tuple(getattr(v, "shape", ())))
                 for k, v in kwargs.items()},
            )

        kv_cache = kwargs.get("kv_cache")
        slot_mapping = kwargs.get("slot_mapping")
        key = kwargs.get("key")
        value = kwargs.get("value")

        rest = [a for a in args if torch.is_tensor(a)]
        if kv_cache is None and rest:
            # cache has by far the largest footprint
            kv_cache = max(rest, key=lambda t: t.numel())
            rest = [t for t in rest if t is not kv_cache]
        if slot_mapping is None:
            ints = [t for t in rest if not t.is_floating_point()]
            if ints:
                slot_mapping = ints[0]
                rest = [t for t in rest if t is not slot_mapping]
        if key is None or value is None:
            fp = [t for t in rest if t.is_floating_point()]
            if len(fp) >= 2:
                key, value = fp[0], fp[1]

        if kv_cache is None or key is None or value is None or slot_mapping is None:
            raise RuntimeError(
                "do_kv_cache_update could not identify its arguments; set "
                "THUNDER_DEBUG_KV=1 to log them"
            )

        quantizer = self._ensure_quantizer(key.device)
        n = int(key.shape[0])
        reshape_and_cache(
            key[:n].reshape(n, self.num_kv_heads, self.head_size),
            value[:n].reshape(n, self.num_kv_heads, self.head_size),
            slot_mapping[:n],
            kv_cache,
            self._scales_for(kv_cache),
            quantizer,
            self.layout,
        )
        for holder in args:
            if hasattr(holder, "_tq_cache_updated"):
                holder._tq_cache_updated = True

    def _scales_for(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """The norm tensor paired with ``kv_cache``.

        vLLM only threads a single cache tensor through the backend API, so the
        fp16 norms ride alongside it on the impl, keyed by the cache's data
        pointer. Populated by :meth:`bind_scales` at allocation time.
        """
        scales = getattr(self, "_kv_scales", None)
        num_blocks = int(kv_cache.shape[0])
        # Re-derive if the cache changed size. vLLM's KV-cache profile/warmup can
        # call this with a 1-block cache first; caching that buffer left a
        # 1-block norm tensor for a ~122k-block cache, so gathering norms for
        # block ids ran off the end (ScatterGather "index out of bounds") and,
        # once clamped, silently used block 0's norms. Never resize while a graph
        # is being captured -- it would move a pointer already baked in.
        if scales is not None and int(scales.shape[0]) != num_blocks:
            if torch.cuda.is_current_stream_capturing():
                return scales
            scales = None
        if scales is None:
            # vLLM allocates the KV cache itself and threads only that one tensor
            # through the backend API, so nothing calls bind_scales(). Allocate the
            # paired fp16 norms here, sized from the cache's block count, so the
            # engine can start without an explicit allocation hook.
            # NOTE: this side buffer is not counted by vLLM's cache accounting;
            # the tidy fix is a view into the norms region of the packed slot.
            try:
                shape = self.layout.get_scales_shape(num_blocks)
                dtype = getattr(self.layout, "scale_dtype", torch.float16) or torch.float16
                scales = torch.zeros(shape, dtype=dtype, device=kv_cache.device)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "TurboQuant norms buffer not bound and could not be derived "
                    f"from kv_cache of shape {tuple(kv_cache.shape)}: {exc}"
                ) from exc
            self._kv_scales = scales
            logger.info(
                "lazily bound TurboQuant norms buffer %s (dtype=%s) for cache %s",
                tuple(scales.shape), scales.dtype, tuple(kv_cache.shape),
            )
        return scales

    def bind_scales(self, kv_scales: torch.Tensor) -> None:
        self._kv_scales = kv_scales


def allocate(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    k_bits: int = 4,
    v_bits: int = 4,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thin re-export so ``registry``/tests have one import site."""
    return allocate_kv_cache(
        num_blocks, block_size, num_kv_heads, head_dim, k_bits, v_bits, dtype, device
    )


__all__ = [
    "BACKEND_NAME",
    "ThunderAttentionBackend",
    "ThunderAttentionImpl",
    "ThunderCuteConfig",
    "allocate",
]
