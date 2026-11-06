#!/usr/bin/env python3
"""Package the TONER-1 Pro printer fine-tune.
  ckpt/pro/finetune.pt      model weights + model_args + fine-tune cfg + eval curve + provenance (no optimizer)
  ckpt/pro/finetune_v0.bin  llama2.c legacy (v0) float32 export, the format toner1/export/export.py reads
Usage: python -I pro_ft_finalize.py [run_dir]   (final TONER-1 Pro: ckpt/pro_ft/run_ft2_tag)
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, L2C, TOK4096  # noqa: E402
sys.path.insert(0, L2C)
import torch
from model import Transformer, ModelArgs
from export import legacy_export

run = sys.argv[1] if len(sys.argv) > 1 else f"{TT}/ckpt/pro_ft/run_ft2_tag"
best = torch.load(f"{run}/best.pt", map_location="cpu", weights_only=False)
last = torch.load(f"{run}/ckpt.pt", map_location="cpu", weights_only=False)
m = Transformer(ModelArgs(**best["model_args"]))
m.load_state_dict(best["model"])
assert m.tok_embeddings.weight is m.output.weight
nparams = sum(p.numel() for p in m.parameters())
evals = [l.rstrip("\n").split("\t") for l in open(f"{run}/evals.tsv")]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
pre = f"{TT}/ckpt/pro/pretrain.pt"
out = {
    "model": best["model"],
    "model_args": best["model_args"],
    "cfg": best["cfg"],
    "params": nparams,
    "iter_num": best["iter_num"],
    "tokens_seen": best["tokens_seen"],
    "printer_tokens_seen": best["printer_tokens_seen"],
    "eval_printer_val": best["eval_printer_val"],
    "eval_ts_val": best["eval_ts_val"],
    "eval_printer_trn": best["eval_printer_trn"],
    "eval_ts_val_tag": best.get("eval_ts_val_tag"),
    "train_seconds": best["train_seconds"],
    "train_seconds_total_run": last["train_seconds"],
    "evals_tsv_columns": ["iter", "printer_epoch", "printer_val", "ts_val(640 windows)", "printer_trn(160 windows)",
                          "lr", "train_seconds", "time", "ts_val_tag(640 windows, replay format; tagged runs only)"],
    "evals": evals,
    "init_checkpoint": pre,
    "init_checkpoint_md5": md5(pre),
    "tokenizer": "tok4096.model",
    "tokenizer_md5": md5(f"{TOK4096}.model"),
    "data": json.load(open(f"{TT}/data/tok4096_printer/meta.json")) | {
        "replay": "TinyStoriesV2-GPT4 train shards data/tok4096/data01-28.bin, 1:1 tokens with printer rows",
        "replay_separator": best["cfg"].get("replay_bos", 1),
        "note": "replay_separator 2: replay stories are delimited by token 2 (</s>) instead of BOS, so BOS (1), "
                "where the print job starts, means 'printer story'"},
    "code": "scripts/pro_ft_train.py + llama2.c 350e04f model.py (MIT)",
    "license": "MIT (our weights, our tokenizer); trained on TinyStoriesV2-GPT4 (CDLA-Sharing-1.0) and printer stories "
               "written by Xing4.0-29B-A4B (Apache-2.0) via OpenRouter free tier",
    "source_checkpoint": f"{run}/best.pt",
}
torch.save(out, f"{TT}/ckpt/pro/finetune.pt")
legacy_export(m, f"{TT}/ckpt/pro/finetune_v0.bin")
print(json.dumps({k: out[k] for k in ["params", "iter_num", "tokens_seen", "printer_tokens_seen", "eval_printer_val",
                                      "eval_ts_val", "eval_ts_val_tag", "train_seconds", "train_seconds_total_run", "model_args"]}))
print("md5 finetune.pt", md5(f"{TT}/ckpt/pro/finetune.pt"), "finetune_v0.bin", md5(f"{TT}/ckpt/pro/finetune_v0.bin"))
