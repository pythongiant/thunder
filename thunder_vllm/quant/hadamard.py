"""Hadamard / orthogonal rotation applied to K and V before quantization.

Why rotate at all
-----------------
TurboQuant is a *scalar* quantizer. Its distortion depends on the coordinate
distribution of the vector being quantized. A raw K/V row is highly
non-uniform across coordinates, so per-coordinate Lloyd-Max is badly matched.
Multiplying by an orthonormal matrix spreads energy uniformly across
coordinates, making each coordinate close to Gaussian. Since the rotation is
orthonormal, ``<q, k> == <q @ R, k @ R>`` -- the attention scores are
unchanged in exact arithmetic, so Q is rotated by the same matrix at the model
level and the kernel never needs to undo it.

Contract
--------
* The **same** matrix is used for Q and for K/V of a given layer.
* For K/V the rotation is applied **once per token, before the cache write**.
* Q is rotated **exactly once**, at the model level, after the Q projection.
* The matrix is symmetric and orthonormal (``R == R.T``, ``R @ R.T == I``) for
  the Sylvester construction, so the "inverse" is free -- which matters for
  continuation-prefill paths that need to go back to the original basis.

``T = R @ R.T`` is verified in the tests to within fp32 tolerance.
"""

from __future__ import annotations

import functools
import math

import torch

from thunder_vllm.utils.logging import get_logger

logger = get_logger("quant.hadamard")


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _sylvester(d: int) -> torch.Tensor:
    """Unnormalised Sylvester Hadamard matrix of order ``d`` (``d`` = 2**k)."""
    if not is_power_of_two(d):
        raise ValueError(f"Sylvester construction needs a power of two, got {d}")
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < d:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h


def _random_orthogonal(d: int, seed: int) -> torch.Tensor:
    """Uniformly-random orthogonal matrix via Householder QR of a Gaussian."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Fix signs so the decomposition is deterministic across BLAS backends.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q


@functools.lru_cache(maxsize=64)
def _build_hadamard_cached(
    head_dim: int,
    device_str: str,
    dtype_str: str,
    random_signs: bool,
    seed: int,
) -> torch.Tensor:
    if is_power_of_two(head_dim):
        h = _sylvester(head_dim) / math.sqrt(head_dim)
        if random_signs:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            signs = torch.randint(
                0, 2, (head_dim,), generator=gen, dtype=torch.int64
            )
            signs = signs.to(torch.float64) * 2.0 - 1.0
            # Diagonal sign flips: D @ H is still orthogonal and symmetric.
            h = h * signs.unsqueeze(1) * signs.unsqueeze(0)
    else:
        # Non power-of-two head dims cannot use Sylvester; fall back to a
        # random orthogonal matrix. Requires materialising D x D, which is
        # only acceptable for the small head dims vLLM supports (<= 256).
        logger.warning(
            "head_dim=%d is not a power of two; using a random orthogonal "
            "rotation instead of Hadamard",
            head_dim,
        )
        h = _random_orthogonal(head_dim, seed)

    return h.to(dtype=getattr(torch, dtype_str), device=torch.device(device_str))


def build_hadamard(
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    random_signs: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Return the ``(head_dim, head_dim)`` orthonormal rotation matrix.

    Matches the reference signature ``build_hadamard(head_dim)``; the extra
    kwargs control dtype/device/construction. Results are cached per
    ``(head_dim, device, dtype, random_signs, seed)``.

    Args:
        head_dim: Rotated dimension. Powers of two use the Sylvester
            construction (fast, exactly orthonormal, symmetric).
        dtype: Output dtype. The rotation itself is built in fp64 and cast.
        device: Output device.
        random_signs: Apply a diagonal +-1 similarity transform. Lloyd-Max for
            a symmetric distribution is invariant to sign flips, so this does
            not change quantization quality -- it only de-correlates *which*
            coordinate a given index lands in across layers. Default False,
            matching the upstream vLLM TurboQuant reference.
        seed: Seed for ``random_signs`` / the non-power-of-two fallback.
    """
    device = torch.device(device)
    return _build_hadamard_cached(
        int(head_dim),
        str(device),
        dtype.__str__().split(".")[-1],
        bool(random_signs),
        int(seed),
    )


class HadamardRotation:
    """Bundles a rotation matrix with the two directions it is used in.

    Held per layer by the quantizer/attention impl. ``rotate`` maps a vector
    in the original basis into the quantization basis; ``inverse`` maps back.
    For the (symmetric) Sylvester matrix the two are the same operator.
    """

    __slots__ = ("head_dim", "matrix", "_symmetric")

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        random_signs: bool = False,
        seed: int = 0,
    ) -> None:
        self.head_dim = int(head_dim)
        self.matrix = build_hadamard(
            head_dim, dtype=dtype, device=device, random_signs=random_signs, seed=seed
        )
        # Exact symmetry check is cheap and lets callers skip a transpose.
        self._symmetric = bool(
            torch.equal(self.matrix, self.matrix.transpose(0, 1))
        )

    @property
    def transposed(self) -> torch.Tensor:
        return self.matrix if self._symmetric else self.matrix.transpose(0, 1)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ R`` for ``x`` of shape ``(..., head_dim)``."""
        return x @ self.matrix.to(x.dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ R.T`` for ``x`` of shape ``(..., head_dim)``."""
        return x @ self.transposed.to(x.dtype)
