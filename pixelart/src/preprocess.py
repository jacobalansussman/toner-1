"""Build token sequences from the raw Quick, Draw! slices.
28x28 uint8 (ink high) -> exact box downsample to 16x16 -> ink if mean >= THRESH -> 2x2 patches.
Sequence = BOS, CLASS, 64 patch tokens (raster order over the 8x8 patch grid).
Split: per class, a fixed random 5% goes to validation.
usage: python -I src/preprocess.py <pixelart_root>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from classes import CLASSES
from pix import load_raw, to_bits, sequences

THRESH, DILATE, VAL_FRAC, SEED = 96, False, 0.05, 1234

root = sys.argv[1]
proc = os.path.join(root, "data", "proc")
os.makedirs(proc, exist_ok=True)
rng = np.random.default_rng(SEED)
tr, va, stats = [], [], {}
for ci, c in enumerate(CLASSES):
    x = load_raw(root, c)
    b = to_bits(x, THRESH, DILATE)
    s = sequences(b, np.full(len(b), ci))
    perm = rng.permutation(len(s))
    nv = int(round(VAL_FRAC * len(s)))
    va.append(s[perm[:nv]]); tr.append(s[perm[nv:]])
    stats[c] = dict(n=len(s), train=len(s) - nv, val=nv, ink=float(b.mean()),
                    blank=int((b.reshape(len(b), -1).sum(1) == 0).sum()))
tr = np.concatenate(tr).astype(np.uint8); va = np.concatenate(va).astype(np.uint8)
np.save(os.path.join(proc, "train.npy"), tr)
np.save(os.path.join(proc, "val.npy"), va)
json.dump(dict(threshold=THRESH, dilate=DILATE, val_frac=VAL_FRAC, seed=SEED,
               train=len(tr), val=len(va), per_class=stats),
          open(os.path.join(proc, "prep_stats.json"), "w"), indent=1)
print("train", tr.shape, "val", va.shape, "mean ink", np.mean([v["ink"] for v in stats.values()]).round(3),
      "blank images", sum(v["blank"] for v in stats.values()))
