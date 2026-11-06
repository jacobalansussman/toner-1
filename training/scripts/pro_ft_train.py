#!/usr/bin/env python3
"""TONER-1 Pro printer fine-tune: ckpt/pro/pretrain.pt -> printer stories + TinyStories replay (1:1 tokens).

Every micro-batch is BP printer rows + BT TinyStories rows of 256 tokens (llama2.c-style stream windows,
BOS-separated stories, windows may cross story boundaries, as in pretraining).
  printer rows : data/tok4096_printer/train.bin. Per epoch the stories are shuffled (seeded), concatenated
                 and cut into 256-token windows, and the windows are visited in a seeded random order.
  replay rows  : uniformly random 257-token windows of the pretraining train shards data/tok4096/data01-28.bin
                 (seeded by iteration and micro-step).
The batch of every iteration is a pure function of (seed, iter), so a resumed chunk continues the exact stream.
Evals (every eval_interval iters and at the end):
  printer_val  : all 268 non-overlapping 256-token windows of data/tok4096_printer/val.bin
  ts_val       : 640 fixed random windows of data/tok4096/data00.bin (TinyStoriesV2 valid, = pretraining val shard)
  printer_trn  : 160 fixed windows of the epoch-0 printer train stream (overfitting gap)
Checkpoints in out_dir: ckpt.pt (model + optimizer, for resume), best.pt (lowest printer_val),
snap_{iter}.pt (model only, every eval), evals.tsv.
Usage: python -I -u pro_ft_train.py [--key=value ...]   (resumes automatically when out_dir/ckpt.pt exists)
"""
import json, math, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, L2C  # noqa: E402
sys.path.insert(0, L2C)
import numpy as np
import torch
from model import Transformer, ModelArgs

cfg = dict(
    out_dir=f"{TT}/ckpt/pro_ft/run_ft1",
    init=f"{TT}/ckpt/pro/pretrain.pt",
    opt_init="",  # optional checkpoint whose AdamW moments seed the optimizer (e.g. run_B/ckpt.pt, pretraining end)
    seed=20261105,
    bp=8, bt=8, seq=256, accum=4,
    lr=2e-4, min_lr=2e-5, warmup=20, max_iters=1000,
    weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    eval_interval=50, ts_eval_windows=640, ptrain_eval_windows=160, eval_bs=32,
    max_wall_seconds=470, device="mps", log_interval=10,
    replay_bos=1,  # 1 = plain replay; 2 = replay stories delimited by token 2 (</s>) so BOS (1) means "printer story"
)
for a in sys.argv[1:]:
    k, v = a.lstrip("-").split("=", 1)
    assert k in cfg, f"unknown key {k}"
    cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
C = type("C", (), cfg)
torch.manual_seed(C.seed)
dev = torch.device(C.device)
os.makedirs(C.out_dir, exist_ok=True)
T = C.seq

# ------------------------------------------------------------------ data
PD = f"{TT}/data/tok4096_printer"
p_train = np.fromfile(f"{PD}/train.bin", dtype=np.uint16)
p_starts = np.load(f"{PD}/train_starts.npy")
p_lens = np.diff(np.append(p_starts, len(p_train)))
p_val = np.fromfile(f"{PD}/val.bin", dtype=np.uint16)
N_PWIN = (len(p_train) - 1) // T
ts_shards = [np.memmap(f"{TT}/data/tok4096/data{i:02d}.bin", dtype=np.uint16, mode="r") for i in range(1, 29)]
ts_w = np.array([len(m) - T - 1 for m in ts_shards], dtype=np.float64)
ts_w /= ts_w.sum()
ts_val = np.memmap(f"{TT}/data/tok4096/data00.bin", dtype=np.uint16, mode="r")

_epoch_cache = {}
def printer_epoch(e):
    if e not in _epoch_cache:
        rng = np.random.default_rng([C.seed, 1, e])
        order = rng.permutation(len(p_starts))
        stream = np.concatenate([p_train[p_starts[i]:p_starts[i] + p_lens[i]] for i in order])
        assert len(stream) == len(p_train)
        _epoch_cache.clear()
        _epoch_cache[e] = (stream, rng.permutation(N_PWIN))
    return _epoch_cache[e]

def printer_rows(r0, n):
    rows = []
    for r in range(r0, r0 + n):
        stream, perm = printer_epoch(r // N_PWIN)
        w = int(perm[r % N_PWIN])
        rows.append(stream[w * T: w * T + T + 1])
    return np.stack(rows)

def ts_rows(it, micro, n):
    rng = np.random.default_rng([C.seed, 2, it, micro])
    sh = rng.choice(len(ts_shards), size=n, p=ts_w)
    rows = []
    for s in sh:
        m = ts_shards[s]
        o = int(rng.integers(0, len(m) - T - 1))
        rows.append(np.asarray(m[o:o + T + 1]))
    rows = np.stack(rows)
    if C.replay_bos != 1:
        rows[rows == 1] = C.replay_bos
    return rows

def batch(it, micro):
    r0 = (it * C.accum + micro) * C.bp
    a = np.concatenate([printer_rows(r0, C.bp), ts_rows(it, micro, C.bt)]).astype(np.int64)
    t = torch.from_numpy(a).to(dev)  # blocking copy (non_blocking host->MPS copies corrupt batches in torch 2.8)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()

def fixed_windows():
    pv = np.stack([p_val[i * T:i * T + T + 1] for i in range((len(p_val) - 1) // T)])
    rng = np.random.default_rng([C.seed, 3])
    tv = np.stack([np.asarray(ts_val[o:o + T + 1]) for o in rng.integers(0, len(ts_val) - T - 1, C.ts_eval_windows)])
    s0, _ = printer_epoch(0)
    rng = np.random.default_rng([C.seed, 4])
    pt = np.stack([s0[w * T:w * T + T + 1] for w in rng.choice(N_PWIN, C.ptrain_eval_windows, replace=False)])
    out = {"printer_val": pv, "ts_val": tv, "printer_trn": pt}
    if C.replay_bos != 1:
        tvt = tv.copy(); tvt[tvt == 1] = C.replay_bos
        out["ts_val_tag"] = tvt  # TinyStories val in the replay format (story skill, separator = replay_bos)
    return out
EVAL = fixed_windows()

@torch.no_grad()
def evaluate(model):
    model.eval()
    out = {}
    for k, arr in EVAL.items():
        tot = n = 0
        for i in range(0, len(arr), C.eval_bs):
            t = torch.from_numpy(arr[i:i + C.eval_bs].astype(np.int64)).to(dev)
            model(t[:, :-1].contiguous(), t[:, 1:].contiguous())
            b = t.shape[0] * T
            tot += float(model.last_loss) * b; n += b
        out[k] = tot / n
    model.train()
    return out

# ------------------------------------------------------------------ model / optimizer / resume
ck_path = os.path.join(C.out_dir, "ckpt.pt")
resume = os.path.exists(ck_path)
src = torch.load(ck_path if resume else C.init, map_location="cpu", weights_only=False)
margs = src["model_args"]
model = Transformer(ModelArgs(**margs))
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in src["model"].items()})
model.to(dev).train()
optimizer = model.configure_optimizers(C.weight_decay, C.lr, (C.beta1, C.beta2), "cuda")  # fused AdamW, also on MPS
iter_num, train_seconds, best = 0, 0.0, float("inf")
if not resume and C.opt_init:
    o = torch.load(C.opt_init, map_location="cpu", weights_only=False)
    assert all(torch.equal(o["model"][k].cpu(), v.cpu()) for k, v in model.state_dict().items()), "opt_init weights differ from init"
    optimizer.load_state_dict(o["optimizer"])
    print(f"optimizer moments from {C.opt_init} (iter {o['iter_num']})", flush=True)
    del o
if resume:
    optimizer.load_state_dict(src["optimizer"])
    iter_num, train_seconds, best = src["iter_num"], src["train_seconds"], src["best_printer_val"]
    for k in ("bp", "bt", "seq", "accum", "seed", "lr", "min_lr", "warmup", "max_iters"):
        if src["cfg"][k] != cfg[k]:
            print(f"NOTE: {k} changed on resume: {src['cfg'][k]} -> {cfg[k]}", flush=True)
del src
tokens_per_iter = C.accum * (C.bp + C.bt) * T
print(f"{'resume' if resume else 'init'} iter {iter_num} | params {sum(p.numel() for p in model.parameters()):,} | "
      f"{tokens_per_iter} tok/iter ({C.accum * C.bp * T} printer) | printer windows/epoch {N_PWIN} | cfg {json.dumps(cfg)}",
      flush=True)

def get_lr(it):
    if it < C.warmup:
        return C.lr * (it + 1) / (C.warmup + 1)
    if it >= C.max_iters:
        return C.min_lr
    r = (it - C.warmup) / max(1, C.max_iters - C.warmup)
    return C.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (C.lr - C.min_lr)

def save(name, extra=None, opt=True):
    ck = {"model": model.state_dict(), "model_args": margs, "iter_num": iter_num, "cfg": cfg,
          "train_seconds": train_seconds + time.time() - t_chunk, "best_printer_val": best,
          "printer_tokens_seen": iter_num * C.accum * C.bp * T, "tokens_seen": iter_num * tokens_per_iter,
          "init": C.init, "saved_at": datetime.now().isoformat(timespec="seconds")}
    if opt:
        ck["optimizer"] = optimizer.state_dict()
    ck.update(extra or {})
    tmp = os.path.join(C.out_dir, name + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, os.path.join(C.out_dir, name))

def do_eval():
    global best
    l = evaluate(model)
    ep = iter_num * C.accum * C.bp / N_PWIN
    line = (f"{iter_num}\t{ep:.3f}\t{l['printer_val']:.4f}\t{l['ts_val']:.4f}\t{l['printer_trn']:.4f}\t{get_lr(iter_num):.3e}"
            f"\t{train_seconds + time.time() - t_chunk:.0f}\t{datetime.now().isoformat(timespec='seconds')}"
            + (f"\t{l['ts_val_tag']:.4f}" if "ts_val_tag" in l else ""))
    with open(os.path.join(C.out_dir, "evals.tsv"), "a") as f:
        f.write(line + "\n")
    print(f"eval {iter_num} (epoch {ep:.2f}): printer_val {l['printer_val']:.4f} ts_val {l['ts_val']:.4f} "
          f"printer_trn {l['printer_trn']:.4f}" + (f" ts_val_tag {l['ts_val_tag']:.4f}" if "ts_val_tag" in l else ""), flush=True)
    extra = {f"eval_{k}": v for k, v in l.items()}
    save(f"snap_{iter_num:05d}.pt", extra, opt=False)
    if l["printer_val"] < best:
        best = l["printer_val"]
        save("best.pt", extra, opt=False)
    save("ckpt.pt", extra)

t_chunk = time.time()
proc_t0 = time.time()
if iter_num == 0 and not resume:
    do_eval()
t_log = time.time()
while iter_num < C.max_iters:
    lr = get_lr(iter_num)
    for g in optimizer.param_groups:
        g["lr"] = lr
    lsum = 0.0
    for micro in range(C.accum):
        x, y = batch(iter_num, micro)
        model(x, y)
        loss = model.last_loss / C.accum
        loss.backward()
        lsum += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1
    if iter_num % C.log_interval == 0:
        dt = time.time() - t_log; t_log = time.time()
        print(f"{iter_num} | loss {lsum:.4f} | lr {lr:.3e} | {C.log_interval * tokens_per_iter / dt:.0f} tok/s", flush=True)
    if iter_num % C.eval_interval == 0 or iter_num == C.max_iters:
        do_eval()
        t_log = time.time()
    elif C.max_wall_seconds > 0 and time.time() - proc_t0 > C.max_wall_seconds:
        save("ckpt.pt")
        print(f"max_wall_seconds reached at iter {iter_num}, ckpt saved", flush=True)
        break
    if C.max_wall_seconds > 0 and time.time() - proc_t0 > C.max_wall_seconds and iter_num % C.eval_interval == 0:
        print(f"max_wall_seconds reached at iter {iter_num} (after eval), ckpt saved", flush=True)
        break
print(f"chunk done: iter {iter_num}, train_seconds {train_seconds + time.time() - t_chunk:.0f}", flush=True)
