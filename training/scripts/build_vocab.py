#!/usr/bin/env python3
"""Word frequencies of TinyStoriesV2-GPT4-train.txt -> data/vocab/tinystories_wordfreq.tsv (count >= 5).

The "simple vocabulary" check in gen_stories.py allows a word if it occurs often enough in TinyStories
(the threshold is chosen there), plus a small printer-domain whitelist.
Usage: python -I build_vocab.py [path-to-train.txt]
"""
import collections, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else f"{TT}/data/tinystories/TinyStoriesV2-GPT4-train.txt"
OUT = f"{TT}/data/vocab/tinystories_wordfreq.tsv"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
cnt = collections.Counter()
stories = 0
with open(SRC, encoding="utf-8", errors="replace") as fh:
    buf = []
    for line in fh:
        if line.startswith("<|endoftext|>"):
            stories += 1
            continue
        buf.append(line.lower())
        if len(buf) >= 20000:
            cnt.update(WORD.findall("".join(buf))); buf = []
    cnt.update(WORD.findall("".join(buf)))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fo:
    fo.write(f"# source {os.path.basename(SRC)}; stories {stories}; total words {sum(cnt.values())}\n")
    for w, c in cnt.most_common():
        if c < 5:
            break
        fo.write(f"{w}\t{c}\n")
print(f"stories {stories}, distinct {len(cnt)}, written {OUT}")
