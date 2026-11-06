#!/usr/bin/env python3
"""Tokenize the printer-story datasets with the TONER-1 Pro tokenizer (tok4096) for the Pro printer fine-tune.

Input : printer_train.txt, printer_val.txt in paths.PDATA (TinyStoriesV2 layout, from build_datasets.py)
Output: data/tok4096_printer/{train,val}.bin   uint16 token stream, each story = BOS(1) + ids, no EOS
                                               (exactly like pro_pretokenize.py / llama2.c process_shard)
        data/tok4096_printer/{train,val}_starts.npy  int64 offset of each story's BOS in the stream
        data/tok4096_printer/meta.json
Usage: python -I pro_ft_pretokenize.py
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, PDATA, TOK4096  # noqa: E402
import numpy as np
import sentencepiece as spm
from textnorm import normalize, iter_tinystories

MODEL = f"{TOK4096}.model"
OUT = f"{TT}/data/tok4096_printer"


def main():
    sp = spm.SentencePieceProcessor(model_file=MODEL)
    assert sp.vocab_size() == 4096 and sp.bos_id() == 1
    os.makedirs(OUT, exist_ok=True)
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    meta = {"tokenizer": MODEL, "tokenizer_md5": md5(MODEL)}
    for split in ["train", "val"]:
        src = f"{PDATA}/printer_{split}.txt"
        texts = []
        changed = 0
        for s in iter_tinystories(src):
            t = normalize(s)
            assert t is not None, s[:80]
            changed += t != s
            texts.append(t)
        ids = sp.encode(texts)
        lens = np.array([len(x) + 1 for x in ids], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        arr = np.fromiter((t for x in ids for t in ([1] + x)), dtype=np.uint16)
        assert arr.size == lens.sum() and (arr[starts] == 1).all()
        unk = int((arr == 0).sum())
        byte_tok = int(((arr >= 5) & (arr <= 260)).sum())
        arr.tofile(f"{OUT}/{split}.bin")
        np.save(f"{OUT}/{split}_starts.npy", starts)
        meta[split] = {"source": src, "source_md5": md5(src), "stories": len(texts), "tokens": int(arr.size),
                       "tokens_per_story_mean": round(float(lens.mean()), 1), "tokens_per_story_max": int(lens.max()),
                       "stories_over_256_tokens": int((lens > 256).sum()), "unk_tokens": unk,
                       "byte_fallback_tokens": byte_tok, "stories_changed_by_normalize": changed,
                       "chars": sum(len(t) for t in texts)}
        print(split, json.dumps(meta[split]), flush=True)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
