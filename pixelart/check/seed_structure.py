"""seed_structure.py: are doodles from CONSECUTIVE seeds independent samples?
Ghostscript's rand is Park-Miller (x <- 16807 x mod 2^31-1), so the k-th draw after srand(s) is
16807^k s mod m: across consecutive seeds every draw steps by a fixed amount (a lattice), not independently.
Uses the PostScript-exact emulator (sampler_emulate.py, verified identical to Ghostscript) to draw N doodles
per class with (a) consecutive seeds 1001.. and (b) seeds from Python's random, and scores both with the
judge CNN; also reports the mean pairwise Hamming distance between doodles of the same class.
usage: python -I pixelart/check/seed_structure.py <pixelart_root> [n_per_class]"""
import json, os, random, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import numpy as np
root = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
sys.argv = [sys.argv[0], root, os.devnull]          # sampler_emulate reads argv[1..2] at import
import sampler_emulate as se
sys.path.insert(0, os.path.join(root, "src"))
from pix import unpatchify

rng = random.Random(7)
sets = {"consecutive": [[1001 + c * n + i for i in range(n)] for c in range(18)],
        "random": [[rng.randrange(1, 2 ** 31 - 2) for _ in range(n)] for c in range(18)]}
out = {}
for name, seeds in sets.items():
    pv = np.array([[se.doodle(c, s) for s in seeds[c]] for c in range(18)])        # 18 x n x 64
    imgs = unpatchify(pv.reshape(-1, 64)).astype(bool).reshape(18, n, 16, 16)
    npy = os.path.join(here, f"seed_structure_{name}_n{n}.npy"); np.save(npy, imgs)
    flat = imgs.reshape(18, n, -1)
    ham = np.mean([np.mean([(flat[c, i] != flat[c, j]).sum() for i in range(n) for j in range(i + 1, n)]) for c in range(18)])
    out[name] = dict(npy=npy, mean_pairwise_hamming_within_class=float(ham))
    print(name, out[name], flush=True)
    r = subprocess.run([sys.executable, "-I", os.path.join(root, "src", "classifier.py"), root, "score", npy],
                       capture_output=True, text=True)
    print(r.stdout.strip()); out[name]["judge"] = r.stdout.strip()
json.dump(out, open(os.path.join(here, "seed_structure_results.json"), "w"), indent=1)
