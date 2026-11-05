"""Speed benchmark + sampler check for the PostScript doodle engine.
Generates N doodles per class (seeds 1001...) at DTEMP 0.9 inside ONE Ghostscript job (-dNODISPLAY,
nothing drawn), each inside save/restore as the pages do; prints every doodle's 64 patch values; then
(1) reports doodle tokens/s and seconds per doodle from Ghostscript's own usertime, and VM use, and
(2) scores the doodles with the judge CNN (src/classifier.py's net and weights), to compare against the
Python int8 sampler at T = 0.9 (out/samples_d48L4_16k_int8_n64_t09_judge.json).
usage: python -I pixelart/check/bench_judge.py <pixelart_root> [n_per_class] [story_toner1.ps]"""
import json, os, subprocess, sys
import numpy as np

root = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
story_job = sys.argv[3] if len(sys.argv) > 3 else None
GS = os.environ.get("GS", "gs")
here = os.path.join(root, "check")
w = open(os.path.join(root, "out", "doodle_weights.ps")).read()
e = open(os.path.join(root, "..", "build", "doodle_engine.ps")).read()
pre = ""
if story_job:   # run the story model first (12-token story, no page), as in the combined print job
    s = open(story_job).read().replace("%%EOF\n", "")
    lines = s.splitlines(True)
    cut = next(i for i, x in enumerate(lines) if x.startswith("/D 64 def"))
    pre = "".join(lines[:cut]) + "/STEPS 12 def /PAGE false def\n" + "".join(lines[cut:])
body = f"""
/NPER {n} def /totms 0 def
vmstatus (VM0 ) print exch =only ( ) print =only ( ) print =only (\\n) print
usertime /wall0 exch def
0 1 17 {{ /cls exch def
  0 1 NPER 1 sub {{ /i exch def
    /sd 1001 cls NPER mul add i add def
    save
      cls sd TVgen
      (D ) print cls =only ( ) print sd =only ( ) print TVtokprint (\\n) print
      TVms
    exch restore
    /totms exch totms add def
  }} for
}} for
usertime wall0 sub /wall exch def
(TOTAL ) print totms =only ( ) print wall =only (\\n) print
vmstatus (VM1 ) print exch =only ( ) print =only ( ) print =only (\\n) print
"""
path = os.path.join(here, "bench_judge%s.ps" % ("_story" if story_job else ""))
open(path, "w").write(pre + w + e + body)
r = subprocess.run([GS, "-q", "-dNODISPLAY", "-dBATCH", "-dNOPAUSE", path], capture_output=True, text=True)
if r.returncode != 0 or "Error" in r.stdout:
    raise SystemExit(r.stdout[-3000:] + r.stderr[-3000:])
docs = [ln.split() for ln in r.stdout.splitlines() if ln.startswith("D ")]
tot = [ln.split() for ln in r.stdout.splitlines() if ln.startswith("TOTAL ")][0]
vm = [ln for ln in r.stdout.splitlines() if ln.startswith("VM")]
N = len(docs)
assert N == 18 * n
gen_ms = float(tot[1]); all_ms = float(tot[2])
res = dict(doodles=N, generation_seconds=gen_ms / 1000, seconds_incl_save_restore=all_ms / 1000,
           seconds_per_doodle=gen_ms / 1000 / N, patch_tokens_per_second=64 * N / (gen_ms / 1000),
           forward_passes_per_second=65 * N / (gen_ms / 1000), ms_per_forward=gen_ms / (65 * N),
           vmstatus_lines=vm, with_story_model=bool(story_job))
print(json.dumps(res, indent=1))

# ---- judge
sys.path.insert(0, os.path.join(root, "src"))
from pix import unpatchify
pv = np.array([[int(t) for t in d[3:]] for d in docs])
assert pv.shape == (N, 64) and pv.min() >= 0 and pv.max() <= 15
cls = np.array([int(d[1]) for d in docs])
imgs = unpatchify(pv).astype(bool).reshape(18, n, 16, 16)
assert (cls.reshape(18, n) == np.arange(18)[:, None]).all()
tag = "_story" if story_job else ""
npy = os.path.join(here, f"ps_samples_t09_n{n}{tag}.npy")
np.save(npy, imgs)
res["samples_npy"] = npy
json.dump(res, open(os.path.join(here, f"bench_judge{tag}.json"), "w"), indent=1)
