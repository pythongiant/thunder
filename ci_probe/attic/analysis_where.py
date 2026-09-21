import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
n, Hk, D = d["v"].shape
def idx(px, bits): return unpack_indices(px.reshape(-1, px.shape[-1]), bits, D).reshape(n, Hk, D)
got = idx(d["got_v"], 4); ref = idx(d["v_packed"], 4)
match = (got == ref)
per_dim = match.float().mean(dim=(0, 1))
bad = (per_dim < 0.99).nonzero().flatten().tolist()
print(f"dims with <99% match: n={len(bad)} first={bad[:12]} last={bad[-6:] if bad else []}")
# contiguous ranges of bad dims
if bad:
    runs = []
    s = bad[0]; p = bad[0]
    for x in bad[1:]:
        if x != p + 1:
            runs.append((s, p)); s = x
        p = x
    runs.append((s, p))
    print("bad dim runs:", runs[:10])
per_row = match.float().mean(dim=(1, 2))
print("per-row match: first 12 =", [round(x,3) for x in per_row[:12].tolist()])
print("fraction rows fully correct:", float((per_row > 0.999).float().mean()))
# per head
print("per-head match:", [round(x,3) for x in match.float().mean(dim=(0,2)).tolist()])
# K per-dim for contrast
def idxk(px): return unpack_indices(px.reshape(-1, px.shape[-1]), 3, D).reshape(n, Hk, D)
kg = idxk(d["got_k"]); kr = idxk(d["k_packed"])
print("K per-dim min match:", float((kg==kr).float().mean(dim=(0,1)).min()))
