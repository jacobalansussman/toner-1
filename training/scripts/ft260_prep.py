#!/usr/bin/env python3
"""ft260_prep.py: tokenize the fine-tune data for TONER-1 (stories260K) with Karpathy's tok512, unchanged.

Format = llama2.c tinystories.py pretokenize: each story is text.strip(), encoded with BOS (id 1) in front,
no EOS; stories concatenated into one uint16 stream.
Outputs (data/ft260/):
  printer_train.bin / printer_val.bin   all stories of printer_train.txt / printer_val.txt (paths.PDATA)
  ts_replay.bin                         TinyStoriesV2-GPT4-train.txt, every 18th story (ASCII-normalized
                                        with textnorm.normalize; stories that stay non-ASCII are skipped)
  ts_val.bin                            TinyStoriesV2-GPT4-valid.txt, first 2,500 normalized stories (held out)
  prep_meta.json                        counts and md5s
Run: $PY -I scripts/ft260_prep.py
"""
import hashlib, json, os, sys, time
import numpy as np
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, KARPATHY, PDATA  # noqa: E402
import textnorm  # noqa: E402

OUT = os.path.join(TT, "data", "ft260")
os.makedirs(OUT, exist_ok=True)
sp = spm.SentencePieceProcessor(model_file=os.path.join(KARPATHY, "tok512.model"))
assert sp.vocab_size() == 512 and sp.bos_id() == 1


def encode_all(stories):
    ids = sp.encode([s.strip() for s in stories], num_threads=4)
    toks = []
    for t in ids:
        toks.append(1); toks.extend(t)
    a = np.array(toks, dtype=np.uint16)
    return a


def save(name, stories, meta):
    a = encode_all(stories)
    p = os.path.join(OUT, name)
    a.tofile(p)
    meta[name] = dict(stories=len(stories), tokens=int(a.size), tokens_per_story=round(a.size / len(stories), 1),
                      md5=hashlib.md5(open(p, "rb").read()).hexdigest())
    print(name, meta[name], flush=True)


meta = {"tokenizer": "tok512.model (Karpathy stories260K, md5 of tok512.bin 2b4a5ca2ec4199f7a12e1b9a2d6f2d82)",
        "format": "BOS + sp.encode(text.strip()) per story, concatenated, uint16"}
t0 = time.time()
for split in ("train", "val"):
    st = list(textnorm.iter_tinystories(os.path.join(PDATA, f"printer_{split}.txt")))
    assert all(textnorm.normalize(s) == s.strip() for s in st)
    save(f"printer_{split}.bin", st, meta)

val = []
for s in textnorm.iter_tinystories(os.path.join(TT, "data/tinystories/TinyStoriesV2-GPT4-valid.txt")):
    n = textnorm.normalize(s)
    if n is not None:
        val.append(n)
    if len(val) == 2500:
        break
save("ts_val.bin", val, meta)

rep, seen, skipped = [], 0, 0
for i, s in enumerate(textnorm.iter_tinystories(os.path.join(TT, "data/tinystories/TinyStoriesV2-GPT4-train.txt"))):
    seen += 1
    if i % 18:
        continue
    n = textnorm.normalize(s)
    if n is None:
        skipped += 1
        continue
    rep.append(n)
meta["replay_source"] = dict(train_stories_seen=seen, every_nth=18, skipped_non_ascii=skipped)
save("ts_replay.bin", rep, meta)
meta["seconds"] = round(time.time() - t0, 1)
json.dump(meta, open(os.path.join(OUT, "prep_meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
