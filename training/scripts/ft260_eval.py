#!/usr/bin/env python3
"""ft260_eval.py: evaluate stories260K checkpoints (original or fine-tuned) the way the print job uses them.

For each checkpoint: loss on ts_val.bin (held-out TinyStoriesV2 valid, 2,500 stories, 969,825 tokens) and on
printer_val.bin (461 held-out printer stories); then N stories sampled from BOS at temperature T for up to 200
tokens, stopping at BOS (exactly the print job's loop, but torch's RNG instead of PostScript rand), and the story
metrics of ft260_common.story_metrics. --int8 also evaluates the int8 per-row dequantized weights (what
export.py puts into weights.ps).
Writes eval/ft260/<label>.json (summary + every sample) and prints one TSV row per model.
Usage: $PY -I scripts/ft260_eval.py --label X ckpt.pt [--int8] [--n 100] [--seed 1234]
"""
import argparse
import json
import os
import sys

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft260_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--label", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--int8", action="store_true")
ap.add_argument("--no_loss", action="store_true")
a = ap.parse_args()

sp = spm.SentencePieceProcessor(model_file=C.TOKM)
dev = C.device()
m, ck = C.load_pt(a.ckpt)
variants = [("fp32", m)]
if a.int8:
    variants.append(("int8", C.dequantized_copy(m)))
res = dict(ckpt=a.ckpt, iter=ck.get("iter_num"), temp=a.temp, steps=a.steps, n=a.n, seed=a.seed, variants={})
for vn, mm in variants:
    mm.to(dev)
    r = {}
    if not a.no_loss:
        r["ts_val_loss"] = round(C.eval_loss(mm, C.stream("ts_val.bin"), dev), 4)
        r["printer_val_loss"] = round(C.eval_loss(mm, C.stream("printer_val.bin"), dev), 4)
    gens = C.generate(mm, dev, n=a.n, steps=a.steps, temp=a.temp, seed=a.seed)
    samples, ms = [], []
    for toks, ended in gens:
        text = sp.decode(toks)
        sm = C.story_metrics(text, ended)
        ms.append(sm)
        samples.append(dict(text=text, tokens=len(toks), **{k: v for k, v in sm.items()}))
    r["summary"] = C.summarize(ms)
    r["samples"] = samples
    res["variants"][vn] = r
    s = r["summary"]
    print("\t".join(str(x) for x in [a.label, vn, ck.get("iter_num"), r.get("ts_val_loss"), r.get("printer_val_loss"),
                                     s["printer_any"], s["printer3"], s["says_printer"], s["drawable_routed"],
                                     s["calculator"], s["ended_by_bos"], s["loop"], s["gibberish"], s["short"],
                                     s["nonword_rate"], s["mean_words"]]), flush=True)
os.makedirs(os.path.join(C.TT, "eval", "ft260"), exist_ok=True)
json.dump(res, open(os.path.join(C.TT, "eval", "ft260", f"{a.label}.json"), "w"), indent=1)
