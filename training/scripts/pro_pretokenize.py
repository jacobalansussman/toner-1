#!/usr/bin/env python3
"""Pretokenize TinyStoriesV2-GPT4 with the TONER-1 Pro tokenizer into llama2.c's custom-vocab layout.

Output: data/tok4096/data00.bin (TinyStoriesV2-GPT4-valid: llama2.c's PretokDataset uses shard 0 as val)
        data/tok4096/data01.bin ... (TinyStoriesV2-GPT4-train, 100k stories per shard)
Each story is normalized (textnorm.py), stripped, encoded with BOS=1 prepended and no EOS, stored as uint16,
exactly like llama2.c tinystories.py process_shard().
Usage: python -I pro_pretokenize.py [--vocab_size=4096]
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, TOK4096  # noqa: E402
import numpy as np
import sentencepiece as spm
from textnorm import normalize, iter_tinystories

SRC = {"valid": f"{TT}/data/tinystories/TinyStoriesV2-GPT4-valid.txt",
       "train": f"{TT}/data/tinystories/TinyStoriesV2-GPT4-train.txt"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--per_shard", type=int, default=100000)
    a = ap.parse_args()
    model = f"{TOK4096}.model"
    sp = spm.SentencePieceProcessor(model_file=model)
    assert sp.vocab_size() == a.vocab_size and sp.bos_id() == 1
    out_dir = f"{TT}/data/tok{a.vocab_size}"
    os.makedirs(out_dir, exist_ok=True)
    assert not [f for f in os.listdir(out_dir) if f.endswith(".bin")], f"{out_dir} already has .bin shards"
    t0 = time.time()
    meta = {"tokenizer": model, "tokenizer_md5": hashlib.md5(open(model, "rb").read()).hexdigest(), "shards": []}
    shard_id = 0
    def flush(texts, split):
        nonlocal shard_id
        toks = sp.encode(texts, num_threads=os.cpu_count())
        arr = np.fromiter((t for ids in toks for t in ([1] + ids)), dtype=np.uint16)
        fn = os.path.join(out_dir, f"data{shard_id:02d}.bin")
        arr.tofile(fn)
        meta["shards"].append({"file": os.path.basename(fn), "split": split, "stories": len(texts), "tokens": int(arr.size)})
        print(f"{fn}: {split} {len(texts)} stories, {arr.size:,} tokens, avg {arr.size/len(texts):.1f}, {time.time()-t0:.0f}s", flush=True)
        shard_id += 1
    for split in ["valid", "train"]:
        kept = dropped = 0
        buf = []
        for s in iter_tinystories(SRC[split]):
            t = normalize(s)
            if t is None:
                dropped += 1
                continue
            buf.append(t); kept += 1
            if split == "train" and len(buf) >= a.per_shard:
                flush(buf, split); buf = []
        if buf:
            flush(buf, split)
        meta[f"{split}_stories"] = kept
        meta[f"{split}_dropped_nonascii"] = dropped
    for split in ["valid", "train"]:
        meta[f"{split}_tokens"] = sum(s["tokens"] for s in meta["shards"] if s["split"] == split)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "shards"}), f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
