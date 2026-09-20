"""Test: kernel used KEY strides + last-dim stride 1 for VALUE too."""
import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
v = d["v"].float(); rot = d["rot"].float(); vb = d["vb"].float()
n, Hk, D = v.shape
got = unpack_indices(d["got_v"].reshape(-1, 64), 4, D).reshape(n, Hk, D)

def q(x):
    y = x @ rot
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.searchsorted(vb, y)

# kernel index for (h,d) is h*stride_kh + d*1 with stride_kh = key.stride(1) = D
h = torch.arange(Hk); dd = torch.arange(D)
idx = h[:, None] * D + dd[None, :]                       # (Hk, D) flat offsets

# candidate value layouts: true strides (sN, sH, sD) -> logical v[r,h,d] = mem[r*sN + h*sH + d*sD]
# kernel reads mem[r*sN + idx].  Recover which logical (h',d') has h'*sH + d'*sD == idx.
def misread(sN, sH, sD):
    hh = (idx % sH) if sH else torch.zeros_like(idx)
    rem = idx
    # solve h'*sH + d'*sD = idx  with the actual layout's valid ranges
    h2 = (rem // sH) % Hk
    d2 = (rem // sD) % D
    # brute force correct: for each idx, find (h',d') with h'*sH + d'*sD == idx
    out = torch.zeros(Hk, D, dtype=torch.long)
    for a in range(Hk):
        for b in range(D):
            off = a * sH + b * sD
            mask = idx == off
            if mask.any():
                out[mask] = a * D + b
    return out

for name, (sN, sH, sD) in {
    "contig (Hk,D) strides (D*Hk,D,1)": (D * Hk, D, 1),
    "transposed (D,Hk) -> (D*Hk,1,Hk)": (D * Hk, 1, Hk),
    "swapped last two (Hk,D)->(1,D)": (D * Hk, 1, D),
}.items():
    if (sH, sD) == (D, 1):
        vmis = v
    else:
        m = misread(D * Hk, sH, sD)            # (Hk,D) -> flat index into (Hk*D)
        vmis = v.reshape(n, Hk * D)[:, m.reshape(-1)].reshape(n, Hk, D)
    m_ = float((q(vmis) == got).float().mean())
    print(f"  {name:38s} match={m_:.4f}")
