import torch
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
slots = d["slots"].long()
n = slots.numel()
print("slots[:24] =", slots[:24].tolist())
print("slots unique:", int(slots.unique().numel()), "min", int(slots.min()), "max", int(slots.max()))
pos = slots % 16; blk = slots // 16
print("pos[:24]      =", pos[:24].tolist())
print("blk[:24]      =", blk[:24].tolist())
good = list(range(0, n, 12))
print("good rows pos =", [int(pos[i]) for i in good[:12]])
print("good rows blk =", [int(blk[i]) for i in good[:12]])
# V region for a bad row vs expected
got_v, v_packed = d["got_v"].long(), d["v_packed"].long()
i = 1
print("row1 got_v[0,:16]:", got_v[i,0,:16].tolist())
print("row1 exp_v[0,:16]:", v_packed[i,0,:16].tolist())
print("row0 got_v[0,:16]:", got_v[0,0,:16].tolist())
print("row0 exp_v[0,:16]:", v_packed[0,0,:16].tolist())
# is got_v row1 == v_packed row0 or row12?
print("row1 got == exp row0 :", bool(torch.equal(got_v[1,0], v_packed[0,0])))
print("row1 got == exp row12:", bool(torch.equal(got_v[1,0], v_packed[12,0])))
print("row13 got == exp row12:", bool(torch.equal(got_v[13,0], v_packed[12,0])))
print("row12 got == exp row12:", bool(torch.equal(got_v[12,0], v_packed[12,0])))
