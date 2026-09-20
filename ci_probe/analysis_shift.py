import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
v = d["v"].float(); k = d["k"].float(); rot = d["rot"].float(); vb = d["vb"].float()
n, Hk, D = v.shape
got = unpack_indices(d["got_v"].reshape(-1, 64), 4, D).reshape(n, Hk, D)
ref = unpack_indices(d["v_packed"].reshape(-1, 64), 4, D).reshape(n, Hk, D)

def q(x):
    y = x @ rot
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.searchsorted(vb, y)

i = 1
gi = got[i, 0]
best = []
for src, name in ((v, "v"), (k, "k")):
    for j in (0, 1, 2, i - 1, i + 1):
        if not (0 <= j < n):
            continue
        for roll in (0, 1, 2, 4, 8, 16, 32, 64, 127):
            cand = torch.roll(src[j, 0], roll)
            m = (q(cand) == gi).float().mean().item()
            if m > 0.5:
                best.append((name, j, roll, round(m, 3)))
print("shift/neighbour hypotheses matching >0.5:", best[:10])
# exact: does q(v[1,0]) == ref[1,0] and != got
print("q(v[1,0]) == ref[1,0]:", bool((q(v[1,0]) == ref[1, 0]).all()))
print("q(v[1,0]) == got[1,0]:", bool((q(v[1,0]) == got[1, 0]).all()))
# maybe got used un-normalized but with a *different* rotation for v? try identity
print("q(v) with k-bounds == got:", round((torch.searchsorted(d["kb"], (v[1,0] @ rot)) == gi).float().mean().item(), 3))
# is got[1,0] equal to ref of a *pair-swapped* dim order?
half = ref[1, 0].reshape(2, 64)
print("pair-swap(ref) == got:", bool(torch.equal(half.flip(0).reshape(-1), gi)))
print("even/odd deinterleave match:", bool(torch.equal(ref[1, 0].reshape(64, 2).T.reshape(-1), gi)))
