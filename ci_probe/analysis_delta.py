import torch
from thunder_vllm.quant.packing import unpack_indices
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
n, Hk, D = d["v"].shape
got = unpack_indices(d["got_v"].reshape(-1, 64), 4, D).reshape(n, Hk, D)
ref = unpack_indices(d["v_packed"].reshape(-1, 64), 4, D).reshape(n, Hk, D)
bad = torch.ones(n, dtype=torch.bool); bad[::12] = False
delta = (got[bad] - ref[bad]).reshape(-1)
print("delta histogram (-8..8):", torch.bincount((delta + 8).clamp(0, 16), minlength=17).tolist())
print("mean|delta| on bad rows:", float(delta.abs().float().mean()))
# is got closer to ref computed with a different rotation scale?
v = d["v"].float(); rot = d["rot"].float(); vb = d["vb"].float()
for s in (0.5, 0.9, 1.0, 1.1, 2.0):
    y = (v @ (rot * s))
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    idx = torch.searchsorted(vb, y)
    print(f"  rot*{s}: overall match={(idx == got).float().mean():.4f} "
          f"bad-row match={(idx[bad] == got[bad]).float().mean():.4f}")
# what about applying rot twice (squared hadamard)?
for name, y0 in (("v@rot@rot", v @ rot @ rot), ("(v@rot)@rot.T", (v @ rot) @ rot.T)):
    y = y0 / y0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    print(f"  {name}: bad-row match={(torch.searchsorted(vb,y)[bad] == got[bad]).float().mean():.4f}")
