"""ft260_common.py: shared helpers for the TONER-1 (stories260K) printer fine-tune.

Model code is llama2.c's model.py (MIT, commit 350e04f, in training/llama2c/). Nothing here talks to a network or a printer.
"""
import collections
import json
import math
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, L2C, KARPATHY, PDATA  # noqa: E402
sys.path.insert(0, L2C)
from model import ModelArgs, Transformer  # noqa: E402

D = os.path.join(TT, "data", "ft260")
BASE_PT = os.path.join(KARPATHY, "stories260K.pt")
TOKM = os.path.join(KARPATHY, "tok512.model")
SEQ = 512


def device():
    d = os.environ.get("FT260_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    if d == "cpu":
        torch.set_num_threads(int(os.environ.get("FT260_THREADS", "4")))
    return d


def load_pt(path, dropout=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = dict(ck["model_args"])
    if dropout is not None:
        a["dropout"] = dropout
    m = Transformer(ModelArgs(**a))
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    m.load_state_dict(sd, strict=True)
    return m, ck


def stream(name):
    return np.fromfile(os.path.join(D, name), dtype=np.uint16)


@torch.no_grad()
def eval_loss(model, arr, dev, bs=16, max_windows=None):
    """Mean next-token cross-entropy over consecutive non-overlapping 512-token windows (deterministic)."""
    model.eval()
    n = (len(arr) - 1) // SEQ
    if max_windows:
        n = min(n, max_windows)
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        j = min(n, i + bs)
        idx = np.stack([arr[k * SEQ:k * SEQ + SEQ + 1].astype(np.int64) for k in range(i, j)])
        t = torch.from_numpy(idx).to(dev)
        model(t[:, :-1].contiguous(), t[:, 1:].contiguous())
        tot += model.last_loss.item() * (j - i) * SEQ
        cnt += (j - i) * SEQ
    return tot / cnt


def quantize_rows(w):
    """export.py's int8 per-row scheme: s = max|row|/127, b = clamp(round(v/s)+128, 1, 255), w' = (b-128)*s."""
    w = w.detach().double()
    m = w.abs().max(dim=1).values
    m = torch.where(m == 0, torch.full_like(m, 1e-8), m)
    s = (m / 127.0).float().double()            # export.py writes the scale with %.6g; see dequant_like_ps
    b = torch.clamp(torch.round(w / s[:, None]) + 128, 1, 255)
    return (b - 128) * s[:, None]


def dequantized_copy(model):
    """A copy of model whose 2-D weight matrices are replaced by their int8 per-row dequantization."""
    import copy
    q = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in q.named_parameters():
            if p.dim() == 2:
                p.copy_(quantize_rows(p).to(p.dtype))
    return q


@torch.no_grad()
def generate(model, dev, n=100, steps=200, temp=0.8, seed=1234):
    """Sample like the print job: start at BOS (pos 0), softmax(logits/T), no top-p, stop when BOS (1) is sampled
    or after `steps` tokens. Returns list of (token_ids, ended_by_bos)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.ones((n, 1), dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool)
    ended = torch.zeros(n, dtype=torch.bool)
    out = [[] for _ in range(n)]
    for _ in range(steps):
        logits = model(x)[:, -1, :].float()
        if temp <= 0:
            nxt = logits.argmax(-1).cpu()
        else:
            p = torch.softmax(logits / temp, dim=-1).cpu()
            nxt = torch.multinomial(p, 1, generator=g)[:, 0]
        for i in range(n):
            if done[i]:
                continue
            t = int(nxt[i])
            if t == 1:
                done[i] = True; ended[i] = True
            else:
                out[i].append(t)
        if bool(done.all()):
            break
        x = torch.cat([x, nxt[:, None].to(dev)], dim=1)
    return [(o, bool(e)) for o, e in zip(out, ended)]


# ---------------- story metrics ----------------
PRINTER_WORDS = re.compile(r"\b(print\w*|paper\w*|ink|inky|toner|copier\w*|scanner\w*|tray|trays|jam|jammed|"
                           r"cartridge\w*)\b", re.I)   # = gen_stories.py PRINTER_WORDS (dataset filter)
# TONER-1-V router keywords, copied from pixelart/ps/src/story_page.ps (RT dict); class names from classes.py
RT = {"cat": 0, "kitty": 0, "kitties": 0, "dog": 1, "puppy": 1, "puppies": 1, "bird": 2, "fish": 3, "duck": 4,
      "bunny": 5, "bunnies": 5, "rabbit": 5, "frog": 6, "bear": 7, "teddy": 7, "teddies": 7, "sun": 8, "tree": 9,
      "flower": 10, "house": 11, "home": 11, "car": 12, "apple": 13, "cake": 14, "ball": 15, "lamp": 16, "light": 16}
CLASSES = ["cat", "dog", "bird", "fish", "duck", "rabbit", "frog", "teddy-bear", "sun", "tree", "flower", "house",
           "car", "apple", "cake", "soccer ball", "light bulb", "calculator"]
LETTERS = re.compile(r"[A-Za-z]+")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def route(text):
    """Exact emulation of story_page.ps /route: first maximal ASCII-letter run (<= 40 letters) that is a key,
    or a key + trailing 's'. Returns class name or None."""
    for m in LETTERS.finditer(text):
        w = m.group(0).lower()
        if len(w) > 40:
            continue
        if w in RT:
            return CLASSES[RT[w]]
        if len(w) >= 2 and w.endswith("s") and w[:-1] in RT:
            return CLASSES[RT[w[:-1]]]
    return None


_KNOWN = None


def known_words():
    global _KNOWN
    if _KNOWN is None:
        k = set()
        for ln in open(os.path.join(TT, "data/vocab/tinystories_wordfreq.tsv")):
            if ln.startswith("#"):
                continue
            k.add(ln.split("\t")[0])
        c = collections.Counter(WORD.findall(open(os.path.join(PDATA, "printer_train.txt")).read().lower()))
        k |= {w for w, n in c.items() if n >= 2}
        _KNOWN = k
    return _KNOWN


def story_metrics(text, ended):
    words = WORD.findall(text.lower())
    nw = len(words)
    kn = known_words()
    nonwords = [w for w in words if w not in kn and w.split("'")[0] not in kn]
    grams = collections.Counter(tuple(words[i:i + 4]) for i in range(max(0, nw - 3)))
    maxrep = max(grams.values()) if grams else 0
    pw = PRINTER_WORDS.findall(text)
    r = route(text)
    return dict(words=nw, ended=ended, printer_any=len(pw) >= 1, printer3=len(pw) >= 3,
                says_printer=bool(re.search(r"\bprinters?\b", text, re.I)), routed=r is not None, route=r,
                calculator=bool(re.search(r"\bcalculators?\b", text, re.I)),
                nonwords=len(nonwords), nonword_list=nonwords, loop=maxrep >= 3,
                gibberish=(nw > 0 and len(nonwords) / nw > 0.05) or nw < 5,
                short=nw < 30)


def summarize(ms):
    n = len(ms)
    f = lambda k: round(100.0 * sum(1 for m in ms if m[k]) / n, 1)
    tw = sum(m["words"] for m in ms)
    return dict(n=n, printer_any=f("printer_any"), printer3=f("printer3"), says_printer=f("says_printer"),
                drawable_routed=f("routed"), calculator=f("calculator"), ended_by_bos=f("ended"),
                loop=f("loop"), gibberish=f("gibberish"), short=f("short"),
                nonword_rate=round(100.0 * sum(m["nonwords"] for m in ms) / max(1, tw), 2),
                mean_words=round(tw / n, 1),
                route_counts=dict(collections.Counter(m["route"] or "None" for m in ms).most_common()))
