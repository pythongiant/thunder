"""Local (CPU) repro of the engine V-store mismatch. No GPU."""
import torch
from thunder_vllm.quant.packing import unpack_indices

d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
k, v = d["k"].float(), d["v"].float()
rot, kb, vb = d["rot"].float(), d["kb"].float(), d["vb"].float()
got_v, v_packed = d["got_v"].long(), d["v_packed"].long()
got_k, k_packed = d["got_k"].long(), d["k_packed"].long()
n, Hk, D = v.shape
KB, VB = 3, 4

def idx_of(packed, bits):
    return unpack_indices(packed.reshape(-1, packed.shape[-1]), bits, D)

ref_v = idx_of(v_packed, VB).reshape(n, Hk, D)
kn_v = idx_of(got_v, VB).reshape(n, Hk, D)
ref_k = idx_of(k_packed, KB).reshape(n, Hk, D)
kn_k = idx_of(got_k, KB).reshape(n, Hk, D)
print(f"K match={(kn_k == ref_k).float().mean():.4f}  V match={(kn_v == ref_v).float().mean():.4f}")

def unit(x):
    y = x @ rot
    return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def score(name, idx):
    m = (idx.reshape(-1) == kn_v.reshape(-1)).float().mean().item()
    print(f"  {name:34s} match={m:.4f}")

uv = unit(v)
print("\nhypotheses:")
score("ref (v@rot->vb)", torch.searchsorted(vb, uv))
score("V from K data", torch.searchsorted(vb, unit(k)))
for off in (-2, -1, 1, 2, 3):
    b = vb[off:] if off > 0 else vb[:off]
    score(f"vb shifted {off}", torch.searchsorted(b, uv))
score("vb reversed", torch.searchsorted(vb.flip(0).contiguous(), uv))
score("vb[:7] (8 levels)", torch.searchsorted(vb[:7].contiguous(), uv))
score("vb right=True", torch.searchsorted(vb.contiguous(), uv.contiguous(), right=True))
score("centroids not bounds", torch.searchsorted(vb.roll(1).contiguous(), uv))
# maybe v bytes are the low/high nibbles of a 3-bit-ish pack?
score("v unpacked as 3-bit", idx_of(got_v, 3).reshape(n, Hk, D))
score("v unpacked as 2-bit", idx_of(got_v, 2).reshape(n, Hk, D))
# row-shift: is got_v row i equal to ref row i'?
gr = kn_v.reshape(n, Hk * D)
rr = ref_v.reshape(n, Hk * D)
hits = {}
for i in range(0, 24):
    eq = (gr[i] == rr[i]).float().mean().item()
    hits[i] = round(eq, 3)
print("  per-row match (first 24):", hits)
