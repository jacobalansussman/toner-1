"""Train a TONER-1-V doodle model (llama2.c Transformer) on 16x16 Quick, Draw! patch tokens.
usage: python -I src/train.py <pixelart_root> <run_name> [--dim 64 --layers 5 --heads 8 --kv 4
       --hidden 172 --steps 8000 --bs 256 --lr 2e-3]
Loss is computed only on the 64 patch targets (the CLASS token after BOS is given, not predicted)."""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "vendor"))
import numpy as np
import torch
import torch.nn.functional as F
from model import ModelArgs, Transformer          # vendored llama2.c model.py (MIT)
from classes import CLASSES, VOCAB, SEQLEN, CLASS0, NPATCH

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("name")
ap.add_argument("--dim", type=int, default=64); ap.add_argument("--layers", type=int, default=5)
ap.add_argument("--heads", type=int, default=8); ap.add_argument("--kv", type=int, default=4)
ap.add_argument("--hidden", type=int, default=172)
ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--min_lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_compile", action="store_true"); ap.add_argument("--device", default=None)
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
proc = os.path.join(a.root, "data", "proc")
tr = np.load(os.path.join(proc, "train.npy"), allow_pickle=False)
va = np.load(os.path.join(proc, "val.npy"), allow_pickle=False)
assert tr.shape[1] == SEQLEN and va.shape[1] == SEQLEN
tr = torch.from_numpy(tr.astype(np.int64)).to(dev)
va = torch.from_numpy(va.astype(np.int64)).to(dev)


def xy(s):
    x = s[:, :-1]
    y = s[:, 1:].clone()
    y[:, 0] = -1                      # do not train/score predicting the class from BOS
    return x, y


margs = ModelArgs(dim=a.dim, n_layers=a.layers, n_heads=a.heads, n_kv_heads=a.kv, vocab_size=VOCAB,
                  hidden_dim=a.hidden, multiple_of=1, norm_eps=1e-5, max_seq_len=SEQLEN, dropout=a.dropout)
model = Transformer(margs).to(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"{a.name}: params {nparams:,} on {dev}", flush=True)
opt = model.configure_optimizers(a.wd, a.lr, (0.9, 0.95), "mps")
# torch.compile fuses the many tiny elementwise kernels (RMSNorm, RoPE, SwiGLU): ~1.8x on MPS.
# It wraps the same parameters; evaluation uses the eager module.
step_model = model if a.no_compile else torch.compile(model)


def lr_at(it):
    if it < a.warmup:
        return a.lr * (it + 1) / a.warmup
    r = (it - a.warmup) / max(1, a.steps - a.warmup)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (a.lr - a.min_lr)


@torch.no_grad()
def evaluate():
    model.eval()
    tot = torch.zeros(len(CLASSES), device=dev); cnt = torch.zeros(len(CLASSES), device=dev)
    for i in range(0, len(va), 1000):
        s = va[i:i + 1000]
        x, y = xy(s)
        lg = model(x, y)
        nll = F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-1, reduction="none")
        nll = nll.view(len(s), -1)[:, 1:].sum(1)                 # 64 patch tokens per image
        c = s[:, 1] - CLASS0
        tot.index_add_(0, c, nll); cnt.index_add_(0, c, torch.ones_like(nll))
    model.train()
    per_img = (tot / cnt).cpu().numpy()
    nats_tok = float(tot.sum() / cnt.sum() / NPATCH)
    return nats_tok, {c: float(v / NPATCH) for c, v in zip(CLASSES, per_img)}


hist = []
t0 = time.time(); model.train()
for it in range(a.steps + 1):
    if it % a.eval_every == 0 or it == a.steps:
        vl, per = evaluate()
        bpp = vl * NPATCH / 256 / math.log(2)
        hist.append(dict(step=it, val_nats_per_token=vl, val_bits_per_pixel=bpp, t=time.time() - t0))
        print(f"step {it:6d}  val {vl:.4f} nats/token  {bpp:.4f} bits/pixel  {time.time() - t0:6.1f}s", flush=True)
    if it == a.steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    idx = torch.randint(0, len(tr), (a.bs,), device=dev)
    x, y = xy(tr[idx])
    step_model(x, y)
    loss = step_model.last_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if it % 250 == 0:
        print(f"  it {it:6d} train {loss.item():.4f} lr {lr_at(it):.2e}", flush=True)

train_s = time.time() - t0
os.makedirs(os.path.join(a.root, "ckpt"), exist_ok=True)
stats = dict(name=a.name, params=nparams, args=vars(a), model_args=margs.__dict__, device=dev,
             train_seconds=train_s, history=hist, final_val_nats_per_token=hist[-1]["val_nats_per_token"],
             final_val_bits_per_pixel=hist[-1]["val_bits_per_pixel"], per_class_val_nats_per_token=per,
             train_sequences=int(len(tr)), val_sequences=int(len(va)),
             epochs=a.steps * a.bs / len(tr))
torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, stats=stats),
           os.path.join(a.root, "ckpt", a.name + ".pt"))
json.dump(stats, open(os.path.join(a.root, "ckpt", a.name + ".json"), "w"), indent=1)
print(f"done {a.name}: {train_s:.0f}s, val {hist[-1]['val_nats_per_token']:.4f} nats/token", flush=True)
