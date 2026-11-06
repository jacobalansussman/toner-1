#!/usr/bin/env python3
"""Package the best TONER-1 Pro pretraining checkpoint.
  ckpt/pro/pretrain.pt      model weights + model_args + config + losses + provenance (no optimizer state)
  ckpt/pro/pretrain_v0.bin  llama2.c legacy (v0) float32 export, the format toner1/export/export.py reads
Usage: python -I pro_finalize.py [run_dir]
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, L2C, TOK4096  # noqa: E402
sys.path.insert(0, L2C)
import torch
from model import Transformer, ModelArgs
from export import legacy_export

run = sys.argv[1] if len(sys.argv) > 1 else f"{TT}/ckpt/pro/run_B"
best = torch.load(f"{run}/best.pt", map_location="cpu", weights_only=False)
last = torch.load(f"{run}/ckpt.pt", map_location="cpu", weights_only=False)
m = Transformer(ModelArgs(**best["model_args"]))
m.load_state_dict(best["model"])
nparams = sum(p.numel() for p in m.parameters())
evals = [l.rstrip("\n").split("\t") for l in open(f"{run}/evals.tsv")]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
out = {
    "model": best["model"],
    "model_args": best["model_args"],
    "config": best["config"],
    "params": nparams,
    "iter_num": best["iter_num"],
    "tokens_seen": best["tokens_seen"],
    "val_loss": best["val_loss"],
    "train_loss": best["train_loss"],
    "best_val_loss": float(best["best_val_loss"]),
    "train_seconds": best["train_seconds"],
    "train_seconds_total_run": last["train_seconds"],
    "evals_tsv_columns": ["iter", "tokens", "train_loss", "val_loss", "lr", "train_seconds", "time"],
    "evals": evals,
    "tokenizer": "tok4096.model",
    "tokenizer_md5": md5(f"{TOK4096}.model"),
    "data": json.load(open(f"{TT}/data/tok4096/meta.json")) | {"shards": "data/tok4096/data00.bin (val) + data01-28.bin (train)"},
    "code": "llama2.c 350e04f train.py/model.py/tinystories.py (MIT) with TONER-1 Pro patches, training/llama2c/PATCH_*.diff",
    "license": "MIT (our weights, our tokenizer); trained on TinyStoriesV2-GPT4 (CDLA-Sharing-1.0)",
    "source_checkpoint": f"{run}/best.pt",
}
torch.save(out, f"{TT}/ckpt/pro/pretrain.pt")
legacy_export(m, f"{TT}/ckpt/pro/pretrain_v0.bin")
print(json.dumps({k: out[k] for k in ["params", "iter_num", "tokens_seen", "val_loss", "train_loss", "train_seconds", "train_seconds_total_run", "model_args"]}))
print("md5 pretrain.pt", md5(f"{TT}/ckpt/pro/pretrain.pt"), "pretrain_v0.bin", md5(f"{TT}/ckpt/pro/pretrain_v0.bin"))
