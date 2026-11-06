#!/usr/bin/env python3
"""Evaluate TONER-1 Pro before/after the printer fine-tune, with the TONER-1 (ft260) metric definitions.

Losses (deterministic, all non-overlapping 256-token windows, as llama2.c-style stream windows):
  ts_val      : data/tok4096/data00.bin (TinyStoriesV2-GPT4 valid; the pretraining val shard), all 22,196 windows
  printer_val : data/tok4096_printer/val.bin, all 268 windows
  per-story   : printer_val stories scored alone from BOS (<= 256 tokens), nats/token and nats/char
Models: pretrain (ckpt/pro/pretrain.pt), finetuned (fp32), finetuned_int8 (export.py's per-row int8, dequantized).
Samples (CPU, 4 threads): 50 per model, start at BOS, T=0.8, plain softmax sampling, stop at BOS or 200 tokens (= the print job's
STEPS 200), scored with ft260_common.story_metrics / summarize (printer words, TONER-1-V router, loops,
gibberish, non-words).
Usage: python -I pro_ft_eval.py FT_CKPT OUT_PREFIX [--n=50] [--seed=1234] [--skip_full_ts] [--skip_pretrain --base_ts_val=X] [--tag=2] [--stop2] [--skip_losses]
"""
import hashlib, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, L2C, TOK4096  # noqa: E402
sys.path.insert(0, L2C)
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import Transformer, ModelArgs
from textnorm import iter_tinystories, normalize
import ft260_common as M  # TONER-1 fine-tune metric definitions (shared on purpose)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[1:] if a.startswith("--"))
FT, OUT = args[0], args[1]
N = int(opts.get("n", 50)); SEED = int(opts.get("seed", 1234))
T = 256
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)
sp = spm.SentencePieceProcessor(model_file=f"{TOK4096}.model")
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Transformer(ModelArgs(**ck["model_args"]))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return m.eval(), ck


def stream_loss(m, arr, bs=64):
    n = (len(arr) - 1) // T
    tot = 0.0
    for i in range(0, n, bs):
        j = min(n, i + bs)
        t = torch.from_numpy(np.stack([np.asarray(arr[k * T:k * T + T + 1]) for k in range(i, j)]).astype(np.int64)).to(dev)
        m(t[:, :-1].contiguous(), t[:, 1:].contiguous())
        tot += float(m.last_loss) * (j - i) * T
    return tot / (n * T), n


def per_story(m, texts):
    nll = ntok = nch = 0
    for txt in texts:
        ids = [1] + sp.encode(txt)
        x = torch.tensor([ids[:-1]], device=dev); y = torch.tensor([ids[1:]], device=dev)
        logits = m(x, y)
        nll += float(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"))
        ntok += len(ids) - 1; nch += len(txt)
    return {"stories": len(texts), "nats_per_token": nll / ntok, "nats_per_char": nll / nch,
            "bits_per_char": nll / nch / math.log(2)}


ts_val = np.memmap(f"{TT}/data/tok4096/data00.bin", dtype=np.uint16, mode="r")
p_val = np.fromfile(f"{TT}/data/tok4096_printer/val.bin", dtype=np.uint16)
pv_texts = [t for t in (normalize(s) for s in iter_tinystories(f"{TT}/data/printer_val.txt")) if len(sp.encode(t)) + 1 <= T]

pre, _ = load(f"{TT}/ckpt/pro/pretrain.pt")
ft, ftck = load(FT)
ftq = M.dequantized_copy(ft)
models = {"pretrain": pre, "finetuned": ft, "finetuned_int8": ftq}
if "skip_pretrain" in opts:  # pretrain numbers do not depend on the fine-tune; reuse an earlier *_eval.json
    del models["pretrain"]
STOP2 = "stop2" in opts  # end a sample at token 2 (</s>) as well as at BOS, like engine2.ps after the Pro fix
TAG = int(opts.get("tag", 0))  # replay separator of a tagged fine-tune (2): also score TinyStories val in that format
if TAG:
    ts_val_tag = np.array(ts_val); ts_val_tag[ts_val_tag == 1] = TAG
res = {"ft_ckpt": FT, "ft_ckpt_md5": md5(FT), "ft_iter": ftck.get("iter_num"), "ft_cfg": ftck.get("cfg"),
       "ft260_common_md5": md5(f"{TT}/scripts/ft260_common.py"), "device": dev, "samples": {"n": N, "seed": SEED,
       "temperature": 0.8, "steps": 200, "stop_at_token2": STOP2}, "loss": {}, "metrics": {}}
for name, m in models.items():
    m.to(dev)
    t0 = time.time()
    r = {}
    if "skip_losses" in opts:
        pass
    elif "skip_full_ts" not in opts:
        r["ts_val"], r["ts_val_windows"] = stream_loss(m, ts_val)
        if TAG:
            r["ts_val_tag"], _ = stream_loss(m, ts_val_tag)
    if "skip_losses" not in opts:
        r["printer_val"], r["printer_val_windows"] = stream_loss(m, p_val)
        r["printer_val_per_story"] = per_story(m, pv_texts)
    res["loss"][name] = r
    m.to("cpu")
    if dev == "mps":
        torch.mps.empty_cache()
    print(name, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
    # sampling on CPU: no KV cache, so on MPS every step has a new shape and the allocator cache grew to 8 GB
    torch.set_num_threads(4)
    samples = M.generate(m, "cpu", n=N, steps=200, temp=0.8, seed=SEED)
    if STOP2:  # the print job also stops at token 2; rows are independent, so truncating = stopping there
        samples = [(t[:t.index(2)], True) if 2 in t else (t, e) for t, e in samples]
    rows, ms = [], []
    for i, (toks, ended) in enumerate(samples):
        text = sp.decode(toks)
        mt = M.story_metrics(text, ended)
        ms.append(mt)
        rows.append({"model": name, "i": i, "tokens": len(toks), "ended_with_bos": ended, "text": text, "ids": toks,
                     **{k: v for k, v in mt.items() if k not in ("ended",)},
                     "byte_tokens": sum(1 for t in toks if 5 <= t <= 260 and t != 15), "tok2": toks.count(2)})
    res["metrics"][name] = M.summarize(ms) | {
        "mean_tokens": round(sum(r["tokens"] for r in rows) / N, 1),
        "printer_words_per_100_words": round(100 * sum(len(M.PRINTER_WORDS.findall(r["text"])) for r in rows)
                                             / max(1, sum(r["words"] for r in rows)), 2),
        "any_nonword": round(100 * sum(1 for r in rows if r["nonwords"]) / N, 1),
        "samples_with_token2": sum(1 for r in rows if r["tok2"]),
        "broken_any(loop|gibberish|short|byte)": round(100 * sum(1 for r in rows if r["loop"] or r["gibberish"] or r["short"]
                                                                 or r["byte_tokens"]) / N, 1)}
    with open(f"{OUT}_samples_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, json.dumps(res["metrics"][name]), f"{time.time() - t0:.0f}s, maxrss "
          f"{__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB", flush=True)
if ("ts_val" in res["loss"].get("pretrain", {}) or "base_ts_val" in opts) and "skip_losses" not in opts:
    b = float(opts["base_ts_val"]) if "base_ts_val" in opts else res["loss"]["pretrain"]["ts_val"]
    res["ts_val_change_pct"] = {k: round(100 * (v["ts_val"] / b - 1), 2) for k, v in res["loss"].items()}
    if TAG:
        res["ts_val_tag_change_pct"] = {k: round(100 * (v["ts_val_tag"] / b - 1), 2) for k, v in res["loss"].items()}
json.dump(res, open(f"{OUT}_eval.json", "w"), indent=1)
print("wrote", f"{OUT}_eval.json")
