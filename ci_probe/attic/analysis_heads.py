import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
n, Hk, D = d["v"].shape
got = unpack_indices(d["got_v"].reshape(-1, 64), 4, D).reshape(n, Hk, D)
ref = unpack_indices(d["v_packed"].reshape(-1, 64), 4, D).reshape(n, Hk, D)
# best head-to-head match matrix
print("head match matrix (rows=got h, cols=ref h):")
for h in range(Hk):
    row = [round((got[:, h, :] == ref[:, hp, :]).float().mean().item(), 3) for hp in range(Hk)]
    print(f"  h{h}: {row}")
# also: per (row,head) correct set -> is it a fixed head?
mrh = (got == ref).float().mean(dim=2)
print("heads with any fully-correct (row,head):",
      sorted(set((mrh > 0.999).nonzero()[:, 1].tolist())))
# check whether the correct (row,head) are exactly rows%12==0 and head 0
ok = (mrh > 0.999).nonzero()
subset = [(int(a), int(b)) for a, b in ok[:8]]
print("first correct (row,head):", subset)
print("all correct have head in:", sorted(set(int(b) for _, b in ok)))
