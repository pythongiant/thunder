import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
D = d["v"].shape[-1]; VB = 4
def idx(px): return unpack_indices(px.reshape(-1, px.shape[-1]), VB, D)
got = idx(d["got_v"]); ref = idx(d["v_packed"])
print("got idx histogram:", torch.bincount(got.reshape(-1), minlength=16).tolist())
print("ref idx histogram:", torch.bincount(ref.reshape(-1), minlength=16).tolist())
# per-row: is got a permutation of ref along D?
g1 = got.reshape(-1, D); r1 = ref.reshape(-1, D)
same_multiset = sum(1 for i in range(min(200, g1.shape[0]))
                    if torch.equal(g1[i].sort().values, r1[i].sort().values))
print(f"rows (of 200) where got is a permutation of ref along D: {same_multiset}")
# correlation
print("corr(got,ref):", float(torch.corrcoef(torch.stack([got.float().reshape(-1), ref.float().reshape(-1)]))[0,1]))
# do got indices depend on the *value* at all? compare got row i vs ref row i for head 0
print("got[0,:8]:", got.reshape(-1, D)[0, :8].tolist())
print("ref[0,:8]:", ref.reshape(-1, D)[0, :8].tolist())
print("got[1,:8]:", got.reshape(-1, D)[1, :8].tolist())
print("ref[1,:8]:", ref.reshape(-1, D)[1, :8].tolist())
# is got == quantization of the *key* with the KEY codebook, unpacked as 4-bit?
kk = unpack_indices(d['got_k'].reshape(-1, d['got_k'].shape[-1]), 3, D)
print("got_v vs got_k correlation:", float(torch.corrcoef(torch.stack([got.float().reshape(-1), kk.float().reshape(-1)]))[0,1]))
