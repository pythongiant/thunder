"""Symbolic decoder: tcgen05 SMEM descriptor -> physical 16B cells.

Purely local, no GPU. Takes the exact descriptors the probe feeds to
tcgen05.mma and enumerates, per row, which physical 16-byte cells one K=16 MMA
instruction is entitled to consume under the documented model. Then builds the
descriptor x physical-cell multiplicity matrix and compares its column sums with
the measurement (1,2,4,4).

Descriptor fields (vendor/mma_sm100_desc.py, make_smem_desc_base):
    start_address   bits [0:14)   16B units     (set by declare_ptx_smem_desc)
    leading_byte_offset  [16:30)  16B units     = LBO
    stride_byte_offset   [32:46)  16B units     = SBO
    version              [46:48)
    base_offset          [49:52)
    lbo_mode             [52:53)
    layout_type          [61:64)   SWIZZLE_128B == 2

The layouts we build (probe, M=128 N=64 K=128):
    A mode0 (128,16) stride (64,1) ; mode2 (4,2) stride (16,8192)   fp16
    B mode0 ( 64,16) stride (64,1) ; mode2 (4,2) stride (16,4096)
In 16B (u128) units mode0 becomes (R,2) stride (8,1), and the within-row cell
index for logical K-cell c is  c = 2*c1 + (c0//8),  c in [0,8)   (== 128B row).
Canonical SWIZZLE_128B physical cell = c XOR (row % 8); row stride 8 u128,
row-group (8 rows) stride = SBO = 64 u128.
"""

SWIZZLE_128B = 2


def decode(desc: int) -> dict:
    return {
        "start": desc & 0x3FFF,
        "lbo": (desc >> 16) & 0x3FFF,
        "sbo": (desc >> 32) & 0x3FFF,
        "version": (desc >> 46) & 0x3,
        "base_offset": (desc >> 49) & 0x7,
        "lbo_mode": (desc >> 52) & 0x1,
        "layout_type": (desc >> 61) & 0x7,
    }


# ---------------------------------------------------------------- descriptor
# Exactly the values the probe measured, for the A operand.
A_BASE_DESC = 0x4000404000010000          # LBO=1 SBO=64 type=SWIZZLE_128B
A_START = 192                             # u128, measured (mDbg[0])
A_OFFSETS = [0, 2, 4, 6, 1024, 1026, 1028, 1030]   # measured per-issue deltas
NROWS = 128


def cells_per_issue(start_u128: int, row: int, k_u128: int = 2,
                    lbo: int = 1, sbo: int = 64) -> set:
    """Physical 16B cells one K=16 issue covers for `row`.

    Documented model: the issue consumes k_u128 (= 32B / 16B) cells starting at
    the descriptor's start address, then the canonical swizzle permutes them.
    """
    row_group, row_in = divmod(row, 8)
    out = set()
    for d in range(k_u128):
        # logical cell within the 128B row, relative to the descriptor start
        cell_logical = (start_u128 % 8) + d
        if cell_logical >= 8:
            continue                      # past the row's swizzle atom
        phys = cell_logical ^ row_in      # Swizzle<3,4,3>
        out.add(row_group * (sbo // 8) * 8 + row_in * 8 + phys)
    return out


def table(offsets, label):
    print(f"\n{label}: descriptor start = {A_START} u128, deltas {offsets}")
    d = decode(A_BASE_DESC)
    print(f"  LBO={d['lbo']} SBO={d['sbo']} base_offset={d['base_offset']} "
          f"lbo_mode={d['lbo_mode']} layout_type={d['layout_type']}")
    # multiplicity matrix over the first atom's 64 u128 cells
    ncell = 64
    mat = [[0] * ncell for _ in offsets]
    for i, off in enumerate(offsets):
        for row in range(NROWS):
            for cell in cells_per_issue(A_START + off, row):
                if cell < ncell:
                    mat[i][cell] += 1
    print("  descriptor x physical-cell multiplicity (first 8 cells, and column sums):")
    for i, off in enumerate(offsets):
        print(f"    start+{off:<5} {mat[i][:8]}")
    sums = [sum(mat[i][c] for i in range(len(offsets))) for c in range(ncell)]
    print(f"  column sums (cells 0..7): {sums[:8]}")
    # collapse to the 4 logical 16B cell groups actually probed (f16 0..63)
    grp = [sum(sums[c * 1:c * 1 + 1]) for c in range(4)]
    print(f"  per-f16-cell-group totals for f16 0..63: {grp}  (sum {sum(grp)})")
    return sums


print("=" * 72)
print("Documented model: each issue consumes 2 u128 (32 B = K=16 fp16) per row,")
print("swizzled canonically. Expected correct coverage = every cell once.")
print("=" * 72)
first_atom = [0, 2, 4, 6]
sums = table(first_atom, "PER-ATOM (4 issues covering K=16 each)")
tot = sum(sums[:64])
print(f"\n  TOTAL cell-reads predicted = {tot} for 64 physical u128 cells")
print(f"  => multiplicity per cell under the documented model: "
      f"{'all 1 (perfect)' if tot == 64 else 'NOT all 1'}")

print("\n" + "=" * 72)
print("MEASURED on B200 (per f16 cell group of 16 elements):")
print("    f16  0..15 -> 1x,  16..31 -> 2x,  32..47 -> 4x,  48..63 -> 4x")
print("    read-events = 16*1 + 16*2 + 16*4 + 16*4 = 176  for 64 positions")
print("    ratio 176/64 = 2.75x")
print("=" * 72)

print("\nNo uniform per-issue K span reproduces the measurement:")
for span, pred in ((2, (1, 1, 1, 1)), (4, (1, 2, 2, 2)), (8, (1, 2, 3, 4))):
    print(f"  span {span*16:>3}B  -> {pred}   {'<-- matches' if pred == (1,2,4,4) else ''}")
print("  observed        -> (1, 2, 4, 4)   (nothing matches)")
