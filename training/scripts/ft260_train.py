#!/usr/bin/env python3
"""ft260_train.py: fine-tune Karpathy's stories260K on printer stories + TinyStories replay.

A small loop around llama2.c's model.py (MIT) that follows llama2.c train.py: AdamW from
model.configure_optimizers (weight decay on 2-D tensors only), betas (0.9, 0.99), grad clip 1.0, linear warmup then
cosine decay (here to lr/10), dropout 0 by default (the original run used 0.05; 0 is faster and lighter on MPS), max_seq_len 512, random 513-token windows of the
pretokenized uint16 streams (BOS-separated stories, as tinystories.py pretokenizes them).
Each batch has round(B * printer_frac) rows from printer_train.bin and the rest from ts_replay.bin, so
printer_frac 0.5 is a 1:1 mix by tokens.
Every --eval_every iterations: loss on ts_val.bin (held-out TinyStoriesV2 valid, 2,500 stories) and printer_val.bin,
then a checkpoint iter_XXXXX.pt in llama2.c's format (model, model_args, iter_num, best_val_loss, config).
MPS note: micro-batches of 8 rows (1.2-1.3 GB of GPU memory); larger single batches got the process killed here.
Runs in resumable chunks of --max_minutes (8) wall-clock minutes: each chunk ends by writing last.pt (model, AdamW
state, iteration, RNG state); --resume continues from it. (Long single runs were killed by the environment here.)
Usage: $PY -I scripts/ft260_train.py --name A --lr 3e-4 --iters 3000 --printer_frac 0.5 [--resume]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft260_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--init", default=C.BASE_PT)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--min_lr_frac", type=float, default=0.1)
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--printer_frac", type=float, default=0.5)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--max_minutes", type=float, default=8.0)   # one chunk; resume with --resume
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

out = os.path.join(C.TT, "ckpt", "ft260", a.name)
os.makedirs(out, exist_ok=True)
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = C.device()
model, ck = C.load_pt(a.init, dropout=a.dropout)
model.to(dev)
opt = model.configure_optimizers(a.weight_decay, a.lr, (0.9, 0.99), "cpu")

P = C.stream("printer_train.bin"); R = C.stream("ts_replay.bin")
TSV = C.stream("ts_val.bin"); PV = C.stream("printer_val.bin")
nP = int(round(a.batch * a.printer_frac)); nR = a.batch - nP


def lr_at(it):
    if it < a.warmup:
        return a.lr * (it + 1) / a.warmup
    r = (it - a.warmup) / max(1, a.iters - a.warmup)
    mn = a.lr * a.min_lr_frac
    return mn + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (a.lr - mn)


def batch():
    """B rows (nP printer + nR replay, shuffled), split into micro-batches of --micro rows."""
    rows = []
    for arr, k in ((P, nP), (R, nR)):
        if k:
            ix = rng.integers(0, len(arr) - C.SEQ - 1, size=k)
            rows += [arr[i:i + C.SEQ + 1].astype(np.int64) for i in ix]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    for j in range(0, len(rows), a.micro):
        t = torch.from_numpy(np.stack(rows[j:j + a.micro])).to(dev)
        yield t[:, :-1].contiguous(), t[:, 1:].contiguous(), t.shape[0] / len(rows)


start_it, ema = 0, float("nan")
LAST = os.path.join(out, "last.pt")
if a.resume:
    st = torch.load(LAST, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["optim"])
    rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
    start_it, ema = st["iter_num"], st["ema"]
    print("resumed at iter", start_it, flush=True)

cfg = dict(vars(a), device=dev, printer_rows=nP, replay_rows=nR, tokens_per_iter=a.batch * C.SEQ,
           init_iter=ck.get("iter_num"), model_args=ck["model_args"])
if not a.resume:
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
log = open(os.path.join(out, "evals.tsv"), "a")
if os.path.getsize(os.path.join(out, "evals.tsv")) == 0:
    log.write("iter\tlr\ttrain_loss_ema\tts_val\tprinter_val\tprinter_train_sub\tseconds\n")


def evaluate(it, ema, t0):
    tv = C.eval_loss(model, TSV, dev); pv = C.eval_loss(model, PV, dev)
    ptr = C.eval_loss(model, P, dev, max_windows=296)   # first 296 windows of printer_train (~ printer_val's size)
    model.train()
    sec = time.time() - t0
    log.write(f"{it}\t{lr_at(it):.3e}\t{ema:.4f}\t{tv:.4f}\t{pv:.4f}\t{ptr:.4f}\t{sec:.0f}\n"); log.flush()
    print(f"eval {it}: ts_val {tv:.4f} printer_val {pv:.4f} printer_train_sub {ptr:.4f} ({sec:.0f}s)", flush=True)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ma = dict(ck["model_args"]); ma["dropout"] = a.dropout
    torch.save(dict(model=sd, model_args=ma, iter_num=it, best_val_loss=tv, ts_val=tv, printer_val=pv,
                    config=cfg), os.path.join(out, f"iter_{it:05d}.pt"))
    return tv, pv


def save_last(it):
    torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    optim=opt.state_dict(), iter_num=it, ema=ema, rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state()), LAST + ".tmp")
    os.replace(LAST + ".tmp", LAST)


t0 = time.time()
if start_it == 0:
    evaluate(0, ema, t0)
model.train()
it = start_it
for it in range(start_it + 1, a.iters + 1):
    for gr in opt.param_groups:
        gr["lr"] = lr_at(it - 1)
    opt.zero_grad(set_to_none=True)
    lv = 0.0
    for x, y, w in batch():           # gradient accumulation over micro-batches (MPS memory stays ~1.3 GB)
        model(x, y)
        (model.last_loss * w).backward()
        lv += model.last_loss.item() * w
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    ema = lv if math.isnan(ema) else 0.98 * ema + 0.02 * lv
    if it % 50 == 0:
        print(f"{it} | loss {lv:.4f} ema {ema:.4f} | lr {lr_at(it - 1):.2e} | {(time.time() - t0) / it:.3f}s/it",
              flush=True)
    if it % a.eval_every == 0 or it == a.iters:
        evaluate(it, ema, t0)
    if (time.time() - t0) / 60 > a.max_minutes and it < a.iters:
        save_last(it)
        print("CHUNK_END at iter", it, "; continue with --resume", flush=True)
        break
else:
    save_last(it)
    print("RUN_DONE", it, round(time.time() - t0), "s", flush=True)
