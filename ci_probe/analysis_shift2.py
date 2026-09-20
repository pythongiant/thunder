import torch
d = torch.load("ci_probe/results/store_ab.pt", map_location="cpu")
got_v = d["got_v"].long(); ref = d["v_packed"].long()
n, Hk, _ = got_v.shape
# best byte-shift alignment between got_v[i,h] and ref[i,h], and vs ref of other heads
def best_shift(a, b):
    r = []
    for sh in range(-63, 64):
        if sh >= 0:
            x, y = a[sh:], b[:len(a)-sh]
        else:
            x, y = a[:len(a)+sh], b[-sh:]
        r.append((float((x == y).float().mean()), sh))
    r.sort(reverse=True)
    return r[0]
for i in (1, 2, 13):
    score, sh = best_shift(got_v[i, 0], ref[i, 0])
    print(f"row{i} head0: best_shift={sh} match={score:.3f}")
    for hp in range(1, 4):
        s, sh2 = best_shift(got_v[i, 0], ref[i, hp])
        print(f"          vs ref head{hp}: best_shift={sh2} match={s:.3f}")
# is got_v[i,0] equal to ref[i,0] with a bit-level rotation (e.g., 4-bit shifted by n nibbles)?
# treat as 128 nibbles and rotate
for i in (1,):
    a = torch.cat([(got_v[i,0] & 0xF), (got_v[i,0] >> 4)])          # low nibbles then high
    b = torch.cat([(ref[i,0] & 0xF), (ref[i,0] >> 4)])
    best = max(((float((torch.roll(b, k) == a).float().mean()), k) for k in range(128)))
    print(f"row{i} nibble-sequence rotate best: {best}")
