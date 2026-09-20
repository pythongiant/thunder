import torch
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
got_v, v_packed = d["got_v"].long(), d["v_packed"].long()
n = got_v.shape[0]
print("shapes", got_v.shape, v_packed.shape)
# for a few rows, find which v_packed row matches (head 0)
for i in (0, 1, 9, 100, 1000):
    g = got_v[i, 0]
    exact = [j for j in range(n) if bool(torch.equal(g, v_packed[j, 0]))]
    print(f"row {i}: exact matches -> {exact[:4]}")
# maybe v is byte-identical to the ref V of the same row but a different head
for i in (0, 1, 9):
    row = [h for h in range(got_v.shape[1])
           if any(bool(torch.equal(got_v[i, h], v_packed[j, hh]))
                  for j in range(min(n, 64)) for hh in range(v_packed.shape[1]))]
    print(f"row {i}: head-match candidates {row[:4]}")
# is got_v just a byte permutation of v_packed globally?
print("same multiset of bytes:", bool(torch.equal(got_v.reshape(-1).sort().values,
                                                  v_packed.reshape(-1).sort().values)))
print("got_v unique bytes:", int(got_v.unique().numel()),
      "v_packed unique bytes:", int(v_packed.unique().numel()))
print("got_k unique:", int(d['got_k'].unique().numel()))
