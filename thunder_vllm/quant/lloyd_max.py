"""Lloyd-Max optimal scalar quantizer for the rotated TurboQuant coordinates.

After an orthonormal rotation, each coordinate of a unit-norm vector is
approximately ``N(0, 1 / head_dim)`` -- variance ``1/head_dim`` (i.e. std
``1 / sqrt(head_dim)``). The spec writes this distribution as
``N(0, 1/sqrt(head_dim))``; interpreted as a *standard deviation* that matches
the rotation exactly, which is what we implement (``std = head_dim ** -0.5``).

The codebook is the fixed-point of Lloyd's algorithm (generalised Lloyd-Max
for a continuous density): alternate

    assignment:  cells    = Voronoi partition of the real line by the centroids
    update:      centroid = conditional mean of the density inside its cell

run to convergence on a large Monte-Carlo sample of the target Gaussian. The
result is returned as centroids plus the assignment boundaries (cell edges),
which the Triton store kernel uses as a fast ``searchsorted``.

The LUT the attention kernel gathers from is a *centroid table* broadcast
across the head dimension: ``lut[k, c] == centroids[k]`` for every ``c``. That
makes the kernel's dequant a plain indexed read with no per-coordinate work.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

from thunder_vllm.utils.logging import get_logger

logger = get_logger("quant.lloyd_max")

# Number of histogram bins used to build closed-form-ish cell means. Lloyd's
# algorithm on a fine quantile grid converges to the same fixed point as the
# sample-mean version and is deterministic across machines.
_QUANTILE_GRID = 1 << 20


@dataclass(frozen=True)
class Codebook:
    """A scalar quantizer over the rotated-coordinate Gaussian."""

    centroids: torch.Tensor  # (2**bits,) fp32
    boundaries: torch.Tensor  # (2**bits - 1,) fp32, ascending midpoints
    bits: int
    head_dim: int
    std: float

    @property
    def n_levels(self) -> int:
        return self.centroids.numel()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid index, shape of ``x`` with an int64 dtype."""
        # searchsorted on boundaries -> bucket index == centroid index.
        return torch.searchsorted(self.boundaries, x.contiguous()).to(torch.int64)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(idx.device)[idx]


def _gaussian_std(head_dim: int) -> float:
    return float(head_dim) ** -0.5


def _init_centroids(n_levels: int, std: float) -> torch.Tensor:
    """Quantile initialisation of the Gaussian, which is already close to the
    Lloyd-Max fixed point and avoids the degenerate all-zero solution."""
    normal = torch.distributions.Normal(0.0, std)
    probs = (torch.arange(n_levels, dtype=torch.float64) + 0.5) / n_levels
    return normal.icdf(probs).to(torch.float32)


def _sample_gaussian(std: float, num_samples: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(num_samples, generator=gen, dtype=torch.float64) * std


def _refine_centroids(
    samples: torch.Tensor, centroids: torch.Tensor, num_iterations: int
) -> torch.Tensor:
    """Lloyd iterations; converges monotonically in MSE."""
    c = centroids.to(torch.float64)
    sorted_samples, _ = torch.sort(samples)
    n = sorted_samples.numel()
    prev = None
    for _ in range(num_iterations):
        # Midpoints between neighbouring centroids are the cell boundaries.
        bounds = (c[:-1] + c[1:]) * 0.5
        # Assignment index of each sorted sample.
        assign = torch.searchsorted(bounds, sorted_samples)
        # Cell means via segment sums (cumsum trick, O(n)).
        # We need, per level l: mean of samples whose assign == l.
        counts = torch.zeros(c.numel(), dtype=torch.float64)
        assign_counts = torch.bincount(assign, minlength=c.numel()).to(torch.float64)
        counts = assign_counts
        csum = torch.cumsum(sorted_samples, dim=0)
        # end index of each level = number of samples with assign <= l
        cum_counts = torch.cumsum(assign_counts, dim=0).to(torch.int64)
        # Starting offset for each level.
        starts = cum_counts - assign_counts.to(torch.int64)
        sums = torch.zeros(c.numel(), dtype=torch.float64)
        total = csum[-1] if n > 0 else torch.zeros((), dtype=torch.float64)
        for lvl in range(c.numel()):
            s, e = int(starts[lvl]), int(cum_counts[lvl])
            if e > s:
                before = csum[s - 1] if s > 0 else torch.zeros((), dtype=torch.float64)
                sums[lvl] = csum[e - 1] - before
        nonempty = counts > 0
        new_c = c.clone()
        new_c[nonempty] = sums[nonempty] / counts[nonempty]
        # Empty cells keep their old centroid (standard guard).
        if prev is not None and torch.allclose(new_c, prev, atol=1e-10):
            c = new_c
            break
        prev = c
        c = new_c
    return c.to(torch.float32)


def _distortion(samples: torch.Tensor, centroids: torch.Tensor) -> float:
    bounds = (centroids[:-1] + centroids[1:]) * 0.5
    idx = torch.searchsorted(bounds, samples)
    recon = centroids.to(torch.float64)[idx]
    return float(torch.mean((samples - recon) ** 2))


@functools.lru_cache(maxsize=64)
def _build_codebook_cached(
    bits: int,
    head_dim: int,
    num_iterations: int,
    num_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n_levels = 1 << bits
    std = _gaussian_std(head_dim)

    centroids = _init_centroids(n_levels, std)
    samples = _sample_gaussian(std, num_samples, seed)
    centroids = _refine_centroids(samples, centroids, num_iterations)

    # Standardise ordering: ascending centroids with ascending boundaries.
    centroids, _ = torch.sort(centroids)
    # The target density is symmetric, so the Lloyd-Max fixed point is exactly
    # antisymmetric. Monte-Carlo noise and the empty-cell guard break that by
    # ~1e-3; project back onto the symmetric solution so the codebook has no
    # arbitrary sign bias between coordinates.
    centroids = (centroids - centroids.flip(0)) * 0.5
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5

    logger.info(
        "Lloyd-Max codebook bits=%d head_dim=%d std=%.6g distortion=%.3e",
        bits,
        head_dim,
        std,
        _distortion(samples, centroids),
    )
    return centroids.contiguous(), boundaries.contiguous()


def build_lloyd_max_codebook(
    bits: int,
    num_iterations: int = 100,
    num_samples: int = 1_000_000,
    *,
    head_dim: int = 128,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Codebook:
    """Lloyd-Max codebook for the rotated-coordinate Gaussian.

    The first three parameters match the reference signature
    ``build_lloyd_max_codebook(bits, num_iterations, num_samples)``. The target
    distribution's std is ``1 / sqrt(head_dim)``, so ``head_dim`` is
    keyword-only but required for a correct codebook.

    Returns a :class:`Codebook` with fp32 ``centroids`` and ``boundaries`` on
    ``device``.
    """
    centroids, boundaries = _build_codebook_cached(
        int(bits), int(head_dim), int(num_iterations), int(num_samples), int(seed)
    )
    dev = torch.device(device)
    return Codebook(
        centroids=centroids.to(dev),
        boundaries=boundaries.to(dev),
        bits=int(bits),
        head_dim=int(head_dim),
        std=_gaussian_std(head_dim),
    )


def build_lut(codebook: Codebook, head_dim_padded: int) -> torch.Tensor:
    """Centroid table for the kernel's indexed gather.

    Returns fp16 ``(2**bits, head_dim_padded)`` where every column holds the
    same centroid vector. The kernel computes ``code[row, c] = lut[idx, c]``,
    so this layout makes the gather unit-stride along ``c``.
    """
    n_levels = codebook.n_levels
    if head_dim_padded < codebook.head_dim:
        raise ValueError(
            f"head_dim_padded ({head_dim_padded}) < head_dim ({codebook.head_dim})"
        )
    col = codebook.centroids.to(torch.float16).unsqueeze(1)
    lut = col.expand(n_levels, head_dim_padded).contiguous()
    return lut


@functools.lru_cache(maxsize=64)
def _centroids_cached(head_dim: int, bits: int) -> torch.Tensor:
    return build_lloyd_max_codebook(bits, head_dim=head_dim).centroids


def get_centroids(head_dim: int, bits: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Cached centroid lookup (mirrors upstream ``get_centroids``)."""
    return _centroids_cached(int(head_dim), int(bits)).to(torch.device(device))


def boundary_table(codebook: Codebook) -> torch.Tensor:
    """Boundaries padded to ``2**bits`` with +inf so a single ``searchsorted``
    in a Triton kernel can bucketise without a separate tail branch."""
    b = codebook.boundaries
    inf = torch.full((1,), float("inf"), dtype=b.dtype, device=b.device)
    return torch.cat([b, inf]).contiguous()
