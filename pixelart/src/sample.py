"""Sample doodles from a checkpoint and save contact sheets (8 samples per class + 3 real val images).
usage: python -I src/sample.py <pixelart_root> <run_name> [--temps 1.0 0.8] [--n 8] [--seed 0] [--quant]
--quant samples from the int8-dequantized weights (what the printer runs).
Sampling is restricted to the 16 patch tokens (2..17), as the PostScript sampler should be."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from classes import CLASSES, BOS, CLASS0, PATCH0, NPATCH, VOCAB
from pix import unpatchify, sheet
from quant import load_model, extract, quantize_all, load_into_model

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("name")
ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.8])
ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--quant", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--no_sheet", action="store_true")
a = ap.parse_args()

m, st = load_model(a.root, a.name)
if a.quant:
    load_into_model(m, quantize_all(extract(m))[1])
va = np.load(os.path.join(a.root, "data", "proc", "val.npy"), allow_pickle=False)


@torch.no_grad()
def generate(temp, seed):
    g = torch.Generator().manual_seed(seed)
    C = len(CLASSES)
    idx = torch.tensor([[BOS, CLASS0 + c] for c in range(C) for _ in range(a.n)])
    offmass = []
    for _ in range(NPATCH):
        lg = m(idx)[:, -1, :].double()
        p_all = torch.softmax(lg, -1)
        offmass.append(1 - p_all[:, PATCH0:PATCH0 + 16].sum(-1))
        lp = lg[:, PATCH0:PATCH0 + 16] / temp
        nxt = torch.multinomial(torch.softmax(lp, -1).float(), 1, generator=g) + PATCH0
        idx = torch.cat([idx, nxt], 1)
    pv = (idx[:, 2:] - PATCH0).numpy()
    return unpatchify(pv).reshape(C, a.n, 16, 16), float(torch.stack(offmass).max())


paths = {}
for t in a.temps:
    imgs, off = generate(t, a.seed)
    rows, labs = [], []
    for ci, c in enumerate(CLASSES):
        real = va[va[:, 1] == CLASS0 + ci][:3, 2:] - PATCH0
        rows.append(list(imgs[ci]) + list(unpatchify(real)))
        labs.append(c)
    q = "_int8" if a.quant else ""
    p = os.path.join(a.root, "out", f"samples_{a.name}{q}{a.tag}_t{int(round(t * 10)):02d}.png")
    if not a.no_sheet: sheet(rows, labs, p, scale=4, gap=4,
          title=f"{a.name}{' int8' if a.quant else ''}  temperature {t}  seed {a.seed}: "
                f"{a.n} samples per class, then 3 real validation drawings",
          col_labels=["sample"] * a.n + ["real"] * 3)
    np.save(p.replace(".png", ".npy"), imgs)
    paths[t] = p
    print(f"T={t}: {p}  (max prob mass on non-patch tokens: {off:.2e})")
