#!/usr/bin/env python3
"""Train the TONER-1 Pro SentencePiece BPE tokenizer (our own; no Meta/Llama 2 material involved).

Same SentencePieceTrainer settings as llama2.c's tinystories.py train_vocab (MIT, commit 350e04f),
but the corpus is a seeded random sample of TinyStoriesV2-GPT4 train stories plus the pilot printer
stories repeated --printer_repeat times so printer-domain words become single tokens.
After training, scrub_paths() replaces the local corpus and output paths that SentencePiece records in
the .model's TrainerSpec by their base names. The .bin export writes llama2.c's tokenizer.bin format
(max_token_length, then score/len/bytes per token; BOS -> "\\n<s>\\n", EOS -> "\\n</s>\\n", U+2581 -> space).
It is re-implemented here instead of importing llama2.c's tokenizer.py, which carries a Meta / Llama 2
Community License header.

Usage (the shipped tokenizer): python -I pro_tokenizer.py --vocab_size=4096 --sample=300000 --printer_repeat=60
                                   --user_symbols=\u2581toner,\u2581Toner [--prefix=PATH]   (default prefix TT/tokenizer/tok4096)
"""
import argparse, json, os, random, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT, PDATA  # noqa: E402
import sentencepiece as spm
from textnorm import normalize, iter_tinystories

TRAIN_TXT = f"{TT}/data/tinystories/TinyStoriesV2-GPT4-train.txt"
PRINTER = f"{PDATA}/pilot_printer_stories_601.jsonl"  # the first 601 generated stories (the pilot), id and text only
N_STORIES = 2717699  # stories in TinyStoriesV2-GPT4-train.txt (counted by build_vocab.py)

def export_bin(model_path, bin_path):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    tokens, scores = [], []
    for i in range(sp.vocab_size()):
        t = sp.id_to_piece(i)
        if i == sp.bos_id():
            t = "\n<s>\n"
        elif i == sp.eos_id():
            t = "\n</s>\n"
        tokens.append(t.replace("\u2581", " ").encode("utf-8"))
        scores.append(sp.get_score(i))
    with open(bin_path, "wb") as f:
        f.write(struct.pack("I", max(len(b) for b in tokens)))
        for b, s in zip(tokens, scores):
            f.write(struct.pack("fI", s, len(b)))
            f.write(b)

def _varint(b, i):
    r = sh = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << sh; sh += 7
        if c < 0x80:
            return r, i

def _enc_varint(v):
    out = bytearray()
    while True:
        c = v & 0x7f; v >>= 7
        out.append(c | (0x80 if v else 0))
        if not v:
            return bytes(out)

def _fields(b):
    i, out = 0, []
    while i < len(b):
        k, i = _varint(b, i); f, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i); v = b[i:i + n]; i += n
        elif wt in (1, 5):
            n = 8 if wt == 1 else 4; v = b[i:i + n]; i += n
        else:
            raise ValueError(f"unexpected protobuf wire type {wt}")
        out.append((f, wt, v))
    return out

def _encode(fields):
    out = bytearray()
    for f, wt, v in fields:
        out += _enc_varint(f << 3 | wt)
        out += _enc_varint(v) if wt == 0 else (_enc_varint(len(v)) + v if wt == 2 else v)
    return bytes(out)

def scrub_paths(model_path):
    """Replace the two local paths SentencePiece stores in the model's TrainerSpec (input, field 1, and
    model_prefix, field 2) by their base names. They are training metadata: encoding is unchanged."""
    raw = open(model_path, "rb").read()
    top = _fields(raw)
    assert _encode(top) == raw, "protobuf round trip is not exact"
    new = []
    for f, wt, v in top:
        if f == 2 and wt == 2:
            v = _encode([(g, w, os.path.basename(x.decode()).encode()) if g in (1, 2) and w == 2 else (g, w, x)
                         for g, w, x in _fields(v)])
        new.append((f, wt, v))
    with open(model_path, "wb") as fh:
        fh.write(_encode(new))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=300000)
    ap.add_argument("--printer_repeat", type=int, default=60)
    ap.add_argument("--prefix", default=f"{TT}/tokenizer/tok4096")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--user_symbols", default="", help="comma-separated pieces forced into the vocab, e.g. \u2581toner")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.prefix), exist_ok=True)
    t0 = time.time()
    rng = random.Random(a.seed)
    pick = set(rng.sample(range(N_STORIES), a.sample))
    corpus = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"tok_train_{a.vocab_size}_{a.sample}_{a.printer_repeat}.txt")
    n_ts = n_drop = words_ts = 0
    with open(corpus, "w", encoding="utf-8") as of:
        for i, s in enumerate(iter_tinystories(TRAIN_TXT)):
            if i not in pick:
                continue
            t = normalize(s)
            if t is None:
                n_drop += 1
                continue
            of.write(t + "\n"); n_ts += 1; words_ts += len(t.split())
        printer = [normalize(json.loads(l)["text"]) for l in open(PRINTER)]
        printer = [p for p in printer if p]
        words_pr = sum(len(p.split()) for p in printer)
        for _ in range(a.printer_repeat):
            for p in printer:
                of.write(p + "\n")
    share = words_pr * a.printer_repeat / (words_ts + words_pr * a.printer_repeat)
    print(f"corpus {corpus}: {n_ts} TinyStories stories ({words_ts} words, {n_drop} non-ASCII dropped), "
          f"{len(printer)} printer stories x{a.printer_repeat} ({words_pr * a.printer_repeat} words, {share:.1%} of words), "
          f"{os.path.getsize(corpus)/1e6:.1f} MB, {time.time()-t0:.0f}s", flush=True)
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=a.prefix, model_type="bpe",
        vocab_size=a.vocab_size, self_test_sample_size=0, input_format="text",
        character_coverage=1.0, num_threads=os.cpu_count(), split_digits=True,
        allow_whitespace_only_pieces=True, byte_fallback=True,
        unk_surface=r" \342\201\207 ", normalization_rule_name="identity",
        minloglevel=1, user_defined_symbols=[u for u in a.user_symbols.split(",") if u])
    scrub_paths(a.prefix + ".model")   # no local paths in the shipped .model
    export_bin(a.prefix + ".model", a.prefix + ".bin")
    meta = dict(vocab_size=a.vocab_size, sample=a.sample, seed=a.seed, printer_repeat=a.printer_repeat,
                tinystories_stories=n_ts, tinystories_words=words_ts, nonascii_dropped=n_drop,
                printer_stories=len(printer), printer_word_share=round(share, 4),
                user_defined_symbols=[u for u in a.user_symbols.split(",") if u], spm_settings="llama2.c tinystories.py train_vocab")
    json.dump(meta, open(a.prefix + ".json", "w"), indent=1)
    os.remove(corpus)
    print(f"wrote {a.prefix}.model/.bin/.json in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
