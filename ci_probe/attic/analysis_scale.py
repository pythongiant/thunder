import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
v = d["v"].float(); k = d["k"].float(); rot = d["rot"].float(); vb = d["vb"].float()
n, Hk, D = v.shape
got = unpack_indices(d["got_v"].reshape(-1, 64), 4, D).reshape(n, Hk, D)
bad = torch.ones(n, dtype=torch.bool); bad[::12] = False
u = v @ rot
un = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
best = []
for s in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0):
    for off in (-0.05, -0.02, 0.0, 0.02, 0.05):
        idx = torch.searchsorted(vb, un * s + off)
        m = float((idx[bad] == got[bad]).float().mean())
        best.append((round(m, 4), s, off))
best.sort(reverse=True)
print("top scale/offset matches on bad rows:", best[:6])
# per-head normalisation hypothesis
un2 = u / u.norm(dim=-2, keepdim=True).clamp_min(1e-12)   # normalise across heads (wrong axis)
print("normalise-over-heads match:", float((torch.searchsorted(vb, un2)[bad] == got[bad]).float().mean()))
# raw v (unrotated) normalised
ur = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
print("raw v normalised match:", float((torch.searchsorted(vb, ur)[bad] == got[bad]).float().mean()))
# k data rotated and normalised
uk = k @ rot; uk = uk / uk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
print("k data (v-codebook) match:", float((torch.searchsorted(vb, uk)[bad] == got[bad]).float().mean()))
# per-row: is got for a bad row equal to ref of the row at the tile start?
per = []
for i in range(0, 256, 16):
    m = float((got[i] == got[i]).float().mean())
    refs = torch.searchsorted(vb, un)
    share = float((refs[i] == got[i]).float().mean())
    per.append(round(share, 3))
print("row0-of-tile ref-vs-got share (first 8 tiles):", per[:8])
