"""Run engine2.ps's forward pass (unchanged procedures) on out/doodle_weights.ps in Ghostscript and
compare against out/reference_logits.json. Local Ghostscript only (nodisplay, stdout); nothing is printed.
usage: python -I src/check_ps.py <pixelart_root> <engine2.ps> <work_dir> [gs] [story_weights.ps]
Mode A (default): engine2's forward section is defined inside DOODLE begin ... end.
Mode B (story_weights.ps given): the story weights and engine2's procedures live in userdict, exactly as
in the shipped job; only the activation buffers are re-created under DOODLE begin, then the story's own
/forward runs on the doodle model. This is the coexistence recipe documented in doodle_weights.ps."""
import json, os, subprocess, sys

root, engine, work = sys.argv[1], sys.argv[2], sys.argv[3]
gs = sys.argv[4] if len(sys.argv) > 4 else "gs"
story = sys.argv[5] if len(sys.argv) > 5 else None
os.makedirs(work, exist_ok=True)
src = open(engine).read().splitlines()
a = next(i for i, s in enumerate(src) if s.startswith("/E 2.718281828 def"))
b = next(i for i, s in enumerate(src) if s.strip() == "logits x EMB mv")
assert src[b + 1].strip() == "} bind def"
fwd = "\n".join(src[a:b + 2])
c = next(i for i, s in enumerate(src) if s.startswith("/x D zeros def"))
d = next(i for i, s in enumerate(src) if s.startswith("/isq 1 HS sqrt div def"))
bufs = "\n".join(src[c:d + 1])          # x xb xb2 q hb hb2 logits att KC VC isq, sized from D HID V SEQ
ref = json.load(open(os.path.join(root, "out", "reference_logits.json")))
w = open(os.path.join(root, "out", "doodle_weights.ps")).read()
results = []
for k, seq in enumerate(ref["sequences"]):
    guard = "\n/=only where { pop } { /=only { dup type /stringtype ne { 64 string cvs } if print } bind def } ifelse\n"
    if story:
        sw = "".join(open(story).read().splitlines(True)[2:])      # body, as build.sh does (tail -n +3)
        job = (sw + w + guard + fwd + "\n(story D ) print D =only (\n) print\n"
               "DOODLE begin\n" + bufs + "\n")
    else:
        job = w + guard + "DOODLE begin\n" + fwd + "\n"
    job = (job +
           "/REFTOKS [" + " ".join(map(str, seq["tokens"])) + "] def\n"
           "usertime /t0 exch def\n"
           "0 1 REFTOKS length 1 sub { /rp exch def REFTOKS rp get rp forward\n"
           "  (L ) print logits { =only ( ) print } forall (\\n) print } for\n"
           "(T ) print usertime t0 sub =only (\\n) print\nend\n"
           + ("(after end: D ) print D =only ( V ) print V =only (\\n) print\n" if story else ""))
    p = os.path.join(work, f"check_{k}.ps")
    open(p, "w").write(job)
    r = subprocess.run([gs, "-q", "-dNODISPLAY", "-dBATCH", "-dNOPAUSE", p], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    rows = [list(map(float, ln[2:].split())) for ln in r.stdout.splitlines() if ln.startswith("L ")]
    ms = float([ln for ln in r.stdout.splitlines() if ln.startswith("T ")][0][2:])
    assert len(rows) == len(seq["tokens"]), len(rows)
    assert all(len(ro) == len(seq["logits"][0]) for ro in rows), "logits buffer is not the doodle's V"
    md = max(abs(x - y) for ro, rr in zip(rows, seq["logits"]) for x, y in zip(ro, rr))
    am = sum(ro.index(max(ro[2:18])) == rr.index(max(rr[2:18])) for ro, rr in zip(rows, seq["logits"]))
    if story:
        print([ln for ln in r.stdout.splitlines() if not ln.startswith("L ")])
    results.append(dict(name=seq["name"], mode="B story+doodle" if story else "A", max_abs_diff=md, patch_argmax_agree=f"{am}/{len(rows)}",
                        ms_total=ms, ms_per_forward=ms / len(rows)))
    print(results[-1], flush=True)
json.dump(results, open(os.path.join(work, "check_ps_results%s.json" % ("_B" if story else "")), "w"), indent=1)
