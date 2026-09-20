"""Emulate the Triton _pack (4-bit) in torch and test reshape-pairing variants."""
import torch
from thunder_vllm.quant.packing import pack_indices, unpack_indices

d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
v = d["v"].float(); rot = d["rot"].float(); vb = d["vb"].float()
n, Hk, D = v.shape
got_v = d["got_v"].long()
got_idx = unpack_indices(got_v.reshape(-1, 64), 4, D).reshape(n, Hk, D)

# reference V indices (deterministic)
u = v @ rot
u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
ref_idx = torch.searchsorted(vb, u)                      # (n,Hk,D)
ref_packed = pack_indices(ref_idx, 4, D)                 # (n,Hk,64)
print("ref recomputed == stored v_packed:",
      bool(torch.equal(ref_packed, d["v_packed"].long())))

ROWS, n_bytes, n_per = 16, 64, 2
def tile_pack(flat_idx, pair_fn):
    out = torch.zeros(flat_idx.shape[0], n_bytes, dtype=torch.int64)
    for pid in range(0, flat_idx.shape[0], ROWS):
        t = flat_idx[pid:pid + ROWS]
        out[pid:pid + t.shape[0]] = pair_fn(t)
    return out

def w_adjacent(t):   # canonical: byte b = idx[2b] | idx[2b+1]<<4
    r = t.reshape(t.shape[0], n_bytes, n_per)
    return (r[:, :, 0] | (r[:, :, 1] << 4)) & 0xFF
def w_halves(t):     # layout-reorder: byte b = idx[b] | idx[64+b]<<4
    return (t[:, :n_bytes] | (t[:, n_bytes:] << 4)) & 0xFF
def w_swap(t):       # nibble swapped within adjacent pair
    r = t.reshape(t.shape[0], n_bytes, n_per)
    return ((r[:, :, 0] << 4) | r[:, :, 1]) & 0xFF

flat_ref = ref_idx.reshape(-1, D)
flat_got = got_idx.reshape(-1, D)
for name, fn in (("adjacent(canonical)", w_adjacent), ("halves", w_halves), ("swap", w_swap)):
    packed = tile_pack(flat_ref, fn)
    print(f"  {name:22s} vs pack_indices match="
          f"{float((packed == ref_packed.reshape(-1, n_bytes)).float().mean()):.4f}  "
          f"vs kernel_bytes match="
          f"{float((packed == got_v.reshape(-1, n_bytes)).float().mean()):.4f}")
# canonical emulation must equal pack_indices
print("canonical emulation == pack_indices:",
      bool(torch.equal(tile_pack(flat_ref, w_adjacent), ref_packed.reshape(-1, n_bytes))))
