"""Teacher-force the 3 sequences of out/reference_logits.json through the PostScript doodle engine
(build/doodle_engine.ps, from tools/build.sh = engine2.ps's forward pass inside DENG) in Ghostscript, print every logit EXACTLY
(as integer mantissa and power-of-two exponent, so Ghostscript's 6-digit real printing does not add
error), and compare against the Python float64 reference.

Two jobs:
  A  doodle_weights.ps + doodle_engine.ps only
  B  the shipped story job first (story weights + engine2.ps generating a short story, page off), then the
     doodle weights + engine, then the same teacher forcing; afterwards it checks that the story model's
     names and its forward pass are untouched (same story logits before and after the doodle runs).
For scale, a float32 torch forward of the same dequantized weights is compared against the same
float64 reference: that is what "float32 rounding" alone does.
Local Ghostscript only (-dNODISPLAY); nothing is sent to any printer.
usage: python -I pixelart/check/teacher_force.py <pixelart_root> <story_toner1.ps>"""
import json, os, subprocess, sys
import numpy as np

root, story_job = sys.argv[1], sys.argv[2]
GS = os.environ.get("GS", "gs")
here = os.path.join(root, "check")
ref = json.load(open(os.path.join(root, "out", "reference_logits.json")))
weights = open(os.path.join(root, "out", "doodle_weights.ps")).read()
engine = open(os.path.join(root, "..", "build", "doodle_engine.ps")).read()

PEXACT = r"""
% real -> prints  [-]m p k  meaning (+/-) m * 2^-k, exactly (m < 2^24 is an integer-valued float)
/pexact { dup 0 eq { pop (0) print } {
    /xv exch def xv 0 lt { (-) print } if /xa xv abs def /xk 0 def
    { xa 8388608 ge { exit } if /xa xa 2 mul def /xk xk 1 add def } loop
    { xa 16777216 lt { exit } if /xa xa 2 div def /xk xk 1 sub def } loop
    xa xa cvi sub 0 ne { (NONINT) print } if
    xa cvi =only (p) print xk =only } ifelse } bind def
(F32CHECK ) print 16777217.0 16777216.0 sub =only (\n) print
"""


def tf_block(seqs):
    s = []
    for k, sq in enumerate(seqs):
        s.append(f"(SEQ {k}\\n) print\n/REFTOKS [{' '.join(map(str, sq['tokens']))}] def\n"
                 "DOODLE begin DENG begin\n"
                 "usertime /tft0 exch def\n"
                 "0 1 REFTOKS length 1 sub { /tfp exch def REFTOKS tfp get tfp forward\n"
                 "  (L ) print logits { pexact ( ) print } forall (\\n) print } for\n"
                 "(T ) print usertime tft0 sub =only (\\n) print\n"
                 "end end\n")
    return "".join(s)


def run(job, path):
    open(path, "w").write(job)
    r = subprocess.run([GS, "-q", "-dNODISPLAY", "-dBATCH", "-dNOPAUSE", path], capture_output=True, text=True)
    if r.returncode != 0 or "Error" in r.stdout:
        raise SystemExit(r.stdout[-3000:] + r.stderr[-3000:])
    return r.stdout


def parse_exact(tok):
    neg = tok.startswith("-")
    tok = tok.lstrip("-")
    if tok == "0":
        return 0.0
    assert "NONINT" not in tok, tok
    m, k = tok.split("p")
    v = int(m) * 2.0 ** (-int(k))
    return -v if neg else v


def parse(out):
    assert "F32CHECK 0.0" in out, "Ghostscript reals are not 32-bit floats?"
    seqs, cur, times = [], None, []
    for ln in out.splitlines():
        if ln.startswith("SEQ "):
            cur = []; seqs.append(cur)
        elif ln.startswith("L "):
            cur.append([parse_exact(t) for t in ln[2:].split()])
        elif ln.startswith("T "):
            times.append(float(ln[2:]))
    return [np.array(s) for s in seqs], times


def compare(name, got, refs):
    res = []
    for g, sq in zip(got, refs):
        r = np.array(sq["logits"])
        assert g.shape == r.shape, (g.shape, r.shape)
        d = np.abs(g - r)
        big = np.abs(r) >= 0.1
        pa = (g[:, 2:18].argmax(1) == r[:, 2:18].argmax(1)).sum()
        res.append(dict(sequence=sq["name"], max_abs_diff=float(d.max()),
                        max_rel_diff_all=float((d / np.maximum(np.abs(r), 1e-300)).max()),
                        max_rel_diff_where_abs_ref_ge_0p1=float((d[big] / np.abs(r[big])).max()),
                        max_abs_diff_over_max_abs_logit=float(d.max() / np.abs(r).max()),
                        max_abs_logit=float(np.abs(r).max()),
                        patch_argmax_agree=f"{int(pa)}/{len(r)}",
                        patch_argmax_agree_after_class_token=f"{int((g[1:, 2:18].argmax(1) == r[1:, 2:18].argmax(1)).sum())}/{len(r) - 1}"))
    print(name)
    for x in res:
        print("  ", x)
    return res


out = {}
# ---- A: doodle model alone
jobA = weights + engine + PEXACT + tf_block(ref["sequences"])
gotA, tA = parse(run(jobA, os.path.join(here, "tf_A.ps")))
out["A_doodle_only"] = compare("A: doodle weights + doodle engine", gotA, ref["sequences"])
out["A_ms_per_sequence_66_forwards"] = tA

# ---- B: inside a job that first runs the story model (engine2.ps, 12-token story, no page)
story = open(story_job).read().replace("%%EOF\n", "")
pre = ("/STEPS 12 def /PAGE false def\n")
lines = story.splitlines(True)
cut = next(i for i, s in enumerate(lines) if s.startswith("/D 64 def"))   # after the DSC header
story_body = "".join(lines[:cut]) + pre + "".join(lines[cut:])
probe = ("(STORYPROBE ) print D =only ( ) print V =only ( ) print x length =only ( ) print logits length =only ( ) print "
         "KC length =only ( ) print /forward load length =only ( ) print\n"
         "  1 0 forward logits 0 get pexact ( ) print logits 511 get pexact (\\n) print\n")
jobB = story_body + PEXACT + probe + weights + engine + tf_block(ref["sequences"]) + probe
outB = run(jobB, os.path.join(here, "tf_B.ps"))
gotB, tB = parse(outB)
out["B_after_story_model"] = compare("B: story job (engine2) + doodle weights + doodle engine", gotB, ref["sequences"])
probes = [ln for ln in outB.splitlines() if ln.startswith("STORYPROBE")]
print("story model probe before/after:", probes)
assert len(probes) == 2 and probes[0] == probes[1], probes
out["B_story_probe_before_after"] = probes
out["A_equals_B_bitwise"] = all(np.array_equal(a, b) for a, b in zip(gotA, gotB))
print("A == B bit for bit:", out["A_equals_B_bitwise"])

# ---- float32 baseline: what pure float32 rounding does to the same dequantized weights
sys.path.insert(0, os.path.join(root, "src"))
import torch
from quant import load_model, extract, quantize_all, load_into_model
m, st = load_model(root, "d48L4_16k")
_, dq = quantize_all(extract(m))
load_into_model(m, dq)
m = m.float().eval()
base = []
with torch.no_grad():
    for sq in ref["sequences"]:
        t = torch.tensor([sq["tokens"]])
        lg = m(t, torch.zeros_like(t))[0].double().numpy()
        r = np.array(sq["logits"])
        base.append(dict(sequence=sq["name"], max_abs_diff=float(np.abs(lg - r).max())))
print("float32 torch vs float64 reference:", base)
out["float32_torch_baseline"] = base
# per-layer trace, positions 0 and 1 of sequence 0, is in the reference; the logits already cover it.
json.dump(out, open(os.path.join(here, "teacher_force_results.json"), "w"), indent=1)
