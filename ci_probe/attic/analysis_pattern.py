import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
n, Hk, D = d["v"].shape
def idx(px, bits): return unpack_indices(px.reshape(-1, px.shape[-1]), bits, D).reshape(n, Hk, D)
got = idx(d["got_v"], 4); ref = idx(d["v_packed"], 4)
match = (got == ref).float().mean(dim=(1, 2))
good = (match > 0.999).nonzero().flatten().tolist()
print(f"fully-correct rows: n={len(good)} first 20={good[:20]}")
diffs = [good[i+1]-good[i] for i in range(len(good)-1)]
from collections import Counter
print("gaps between correct rows:", Counter(diffs).most_common(5))
# per (row, head) granularity
mrh = (got == ref).float().mean(dim=2)
good_rh = (mrh > 0.999).nonzero()
print("correct (row,head) count:", good_rh.shape[0], "of", n*Hk)
print("first 12 (row,head):", good_rh[:12].tolist())
# does got_v row i equal ref row i for the FIRST 8 dims always?
first8 = (got[:, :, :8] == ref[:, :, :8]).float().mean()
print("dim 0..7 match fraction:", float(first8))
