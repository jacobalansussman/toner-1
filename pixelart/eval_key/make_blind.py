"""make_blind.py: choose the blind grid's 36 cells (2 doodles per class, shuffled) and its 36 seeds.
Writes, next to this script (eval_key/, which holds everything that reveals the answer):
  blind_order.ps   /BORDER [ class seed  class seed ... ]  in cell order 1..36, read by ps/src/blind_grid_page.ps
  blind_key.json   {"1": "<class>", ..., "36": "<class>"}  cell number -> class
The shuffle and the seeds come from one fixed RNG, chosen once and not re-rolled or cherry-picked.
usage: python -I eval_key/make_blind.py <pixelart_root>"""
import json, os, random, sys

root = sys.argv[1]
here = os.path.join(root, "eval_key")
meta = json.load(open(os.path.join(root, "out", "doodle_meta.json")))
classes = meta["classes"] if "classes" in meta else meta["CLASSES"]
assert len(classes) == 18, classes
rng = random.Random(20261105)
cells = [c for c in range(18) for _ in range(2)]
rng.shuffle(cells)
seeds = rng.sample(range(10000, 100000), 36)          # 36 different seeds, none in the sample sheet's 1..72
with open(os.path.join(here, "blind_order.ps"), "w") as f:
    f.write("% TONER-1-V blind grid: class index and seed for cells 1..36 (THIS IS THE ANSWER KEY)\n/BORDER [\n")
    for i, (c, s) in enumerate(zip(cells, seeds)):
        f.write(f"  {c} {s}\n")
    f.write("] def\n")
json.dump({str(i + 1): classes[c] for i, c in enumerate(cells)},
          open(os.path.join(here, "blind_key.json"), "w"), indent=1)
print("wrote blind_order.ps and blind_key.json:", len(cells), "cells")
