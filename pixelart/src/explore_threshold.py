"""Contact sheets of real drawings under several 16x16 preprocessing variants.
usage: python -I src/explore_threshold.py <pixelart_root>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from classes import CLASSES
from pix import load_raw, to_bits, downsample, sheet

root = sys.argv[1]
out = os.path.join(root, "out")
VARIANTS = [("t64", 64, False), ("t96", 96, False), ("t128", 128, False),
            ("t160", 160, False), ("t192+d", 192, True)]
K = 3  # examples per class
rows, labels = [], []
for c in CLASSES:
    x = load_raw(root, c, 200)[[0, 7, 13]]
    r = []
    for k in range(K):
        r.append(x[k])
        for _, t, d in VARIANTS:
            r.append(to_bits(x[k:k + 1], t, d)[0])
    rows.append(r); labels.append(c)
cols = (["28px"] + [v[0] for v in VARIANTS]) * K
sheet(rows, labels, os.path.join(out, "prep_variants.png"), scale=3, gap=3,
      title="real drawings: 28x28 original, then 16x16 box-downsample + threshold (t) / dilation (+d)",
      col_labels=cols)
# ink statistics per variant
for name, t, d in VARIANTS:
    fr = [to_bits(load_raw(root, c, 500), t, d).mean() for c in CLASSES]
    print(f"{name:8s} mean ink fraction {np.mean(fr):.3f}  (min {min(fr):.3f} max {max(fr):.3f})")
ds = downsample(load_raw(root, "cat", 500))
print("downsampled intensity percentiles (nonzero):", np.percentile(ds[ds > 0], [10, 25, 50, 75, 90]).round(1))
