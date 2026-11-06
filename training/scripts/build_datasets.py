#!/usr/bin/env python3
"""build_datasets.py: final printer-story datasets from gen_stories.py output (toner1-train).

Inputs : data/printer_stories.jsonl (kept at generation time), data/printer_stories_rejected.jsonl
         (stories dropped only for "feature_leak" or "format:markup" are re-tried, because the later
         generator versions trim short leak sentences and bold title lines instead of dropping the story),
         logs/gen_requests.jsonl (request counts).
Every candidate goes through the FINAL filter set of gen_stories.py again, in generation order:
  clean_story (ASCII, markup, digits/symbols, titles, lists) -> trim_leaks -> 80..220 words ->
  >= 3 printer/paper/ink/toner words -> no feature-name leak -> <= 2 hard words -> <= 3 unknown names ->
  near-duplicate (6-word shingles) and exact duplicate.
Split  : 95/5 train/val, seeded shuffle (seed "printer-split-v1").
Outputs (data/):
  printer_train.txt, printer_val.txt   TinyStoriesV2-GPT4 .txt layout: each story, then a line "<|endoftext|>";
                                       paragraph breaks as a single "\\n" (as in TinyStoriesV2). Readable with
                                       scripts/textnorm.py iter_tinystories().
  printer_train.json, printer_val.json llama2.c tinystories.py shard layout: a JSON array of {"story": text, ...}
                                       (pretokenize reads only "story"; the other keys are metadata).
  printer_stats.txt, printer_stats.json
Usage: python -I build_datasets.py [--out DIR]   (default DIR: data/)
"""
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT as TRAIN  # noqa: E402
import gen_stories as g  # noqa: E402

D = os.path.join(TRAIN, "data")
SPLIT_SEED = "printer-split-v1"
VAL_SHARE = 0.05
SALVAGE = {"feature_leak", "format:markup"}
META_KEYS = ("theme_id", "theme", "noun", "verb", "adjective", "features", "drawable", "name")
STOP = set("""a an the and or but so to of in on at for with from by as is are was were be been it its it's he she
they them his her their him we you i me my your our this that there then than when what who how not no yes do did
does done can could would will just very too all one up down out into over again said says say had has have
like if because while about off some more many much any each other only also back away now day time once upon
went go going get got make made see saw look looked came come let us am oh""".split())


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def refilter(text, vocab):
    """Final filter set; returns (text, reason or None, extra)."""
    try:
        text = g.clean_story(text)
    except ValueError as e:
        return text, str(e), {}
    text, n_trim = g.trim_leaks(text)
    nw = len(g.WORD_RE.findall(text))
    hard, names = g.hard_words(text, vocab)
    extra = {"words": nw, "hard_words": sorted(hard), "names": sorted(names), "leak_trimmed": n_trim}
    if nw < g.MIN_WORDS:
        return text, "too_short", extra
    if nw > g.MAX_WORDS:
        return text, "too_long", extra
    if len(g.PRINTER_WORDS.findall(text)) < g.MIN_PRINTER_MENTIONS:
        return text, "off_theme", extra
    if g.FEATURE_LEAK.search(text):
        return text, "feature_leak", extra
    if len(hard) > g.MAX_HARD_WORDS:
        return text, "hard_words", extra
    if len(names) > 3:
        return text, "too_many_unknown_names", extra
    return text, None, extra


def main():
    global D
    if "--out" in sys.argv:
        D = sys.argv[sys.argv.index("--out") + 1]
        os.makedirs(D, exist_ok=True)
    vocab = g.load_vocab()
    kept0 = load(g.OUT)
    rej0 = load(g.REJ)
    log = load(g.LOG)
    cands = [("kept", r) for r in kept0]
    for r in rej0:
        if r["reason"] in SALVAGE and r.get("text"):
            s = r["spec"]
            cands.append(("salvage", {"text": r["text"], **{k: s.get(k) for k in META_KEYS},
                                      "prompt_version": r.get("prompt_version"), "req": r["req"], "slot": r["slot"],
                                      "per_request": None, "model": g.MODEL, "orig_reason": r["reason"]}))
    cands.sort(key=lambda c: (c[1]["req"], c[1]["slot"]))
    dedup, seen = g.Dedup(), set()
    final, dropped = [], collections.Counter()
    salvaged = collections.Counter()
    reqs = {r["req"]: r for r in log}
    for src, r in cands:
        text, reason, extra = refilter(r["text"], vocab)
        if reason is None:
            sid = hashlib.sha1(text.encode()).hexdigest()[:16]
            if sid in seen:
                reason = "exact_duplicate"
            elif dedup.is_dup(text):
                reason = "near_duplicate"
        if reason is not None:
            dropped[(src, reason)] += 1
            continue
        seen.add(sid)
        dedup.add(text)
        if src == "salvage":
            salvaged[r["orig_reason"]] += 1
        rc, rw = g.router(text)
        lg = reqs.get(r["req"], {})
        row = {"id": sid, "story": text, "words": extra["words"],
               **{k: r.get(k) for k in META_KEYS},
               "has_noun": g.has_word(text, r["noun"]), "has_verb": g.has_word(text, r["verb"]),
               "has_adjective": g.has_word(text, r["adjective"]),
               "drawable_found": (any(g.has_word(text, w) for w in g.DRAWABLE_WORDS[r["drawable"]])
                                  if r.get("drawable") else None),
               "router_class": rc, "router_word": rw, "hard_words": extra["hard_words"],
               "leak_trimmed": extra["leak_trimmed"], "prompt_version": r.get("prompt_version"),
               "per_request": r.get("per_request") or len(lg.get("specs") or []) or None,
               "req": r["req"], "slot": r["slot"], "source": src, "model": g.MODEL}
        final.append(row)

    order = list(range(len(final)))
    random.Random(SPLIT_SEED).shuffle(order)
    n_val = round(len(final) * VAL_SHARE)
    val_idx = set(order[:n_val])
    split = {"train": [final[i] for i in range(len(final)) if i not in val_idx],
             "val": [final[i] for i in range(len(final)) if i in val_idx]}

    def v2(text):  # TinyStoriesV2 layout: single newline between paragraphs
        return re.sub(r"\n\s*\n", "\n", text.strip())

    for name, rows in split.items():
        with open(os.path.join(D, f"printer_{name}.txt"), "w") as fh:
            for r in rows:
                fh.write(v2(r["story"]) + "\n<|endoftext|>\n")
        with open(os.path.join(D, f"printer_{name}.json"), "w") as fh:
            json.dump([{**r, "story": v2(r["story"])} for r in rows], fh)

    # ------------------------------------------------------------------ stats
    n = len(final)
    w = [r["words"] for r in final]
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=os.path.join(TRAIN, "ckpt/karpathy_stories260K/tok512.model"))
        toks = [len(t) + 1 for t in sp.encode([v2(r["story"]) for r in final])]  # +1 BOS, as llama2.c
    except Exception as e:  # noqa: BLE001
        toks, sp = [], None
        print("tok512 count skipped:", e)
    th = collections.Counter(r["theme_id"] for r in final)
    req_d = [r for r in final if r["drawable"]]
    rc = collections.Counter(r["router_class"] for r in final)
    wc = collections.Counter(m.group(0).lower() for r in final for m in g.WORD_RE.finditer(r["story"]))
    content = collections.Counter({k: v for k, v in wc.items() if k not in STOP})
    printer_hits = collections.Counter(m.group(0).lower() for r in final for m in g.PRINTER_WORDS.finditer(r["story"]))
    ok = [r for r in log if r.get("status") == "ok"]
    att = collections.Counter(a["status"] for r in log for a in r.get("attempts", []))
    slots_returned = len(kept0) + sum(1 for r in rej0 if r["reason"] != "missing")
    stats = {
        "model": g.MODEL, "model_license": g.MODEL_LICENSE,
        "requests_logged": len(log), "requests_ok": len(ok), "http_attempts": dict(att),
        "usage_cost_sum_ok_requests": sum(r.get("cost") or 0 for r in ok),
        "usage_cost_nonzero_requests": sum(1 for r in ok if r.get("cost")),
        "stories_requested_in_ok_requests": sum(len(r["specs"]) for r in ok),
        "stories_returned": slots_returned,
        "kept_at_generation": len(kept0), "rejected_at_generation": len(rej0),
        "rejected_at_generation_by_reason": dict(collections.Counter(r["reason"] for r in rej0).most_common()),
        "salvaged_at_build": dict(salvaged),
        "dropped_at_build": {f"{s}:{k}": v for (s, k), v in dropped.most_common()},
        "final_stories": n, "train": len(split["train"]), "val": len(split["val"]),
        "by_prompt_version": dict(sorted(collections.Counter(r["prompt_version"] for r in final).items())),
        "by_per_request": dict(sorted(collections.Counter(str(r["per_request"]) for r in final).items())),
        "words": {"mean": round(statistics.mean(w), 1), "median": statistics.median(w), "stdev": round(statistics.stdev(w), 1),
                  "min": min(w), "max": max(w), "total": sum(w)},
        "tok512_tokens_per_story": ({"mean": round(statistics.mean(toks), 1), "median": statistics.median(toks),
                                     "total": sum(toks)} if toks else None),
        "themes_covered": f"{len(th)}/{len(g.THEMES)}",
        "theme_histogram": {g.THEMES[i]: th.get(i, 0) for i in range(len(g.THEMES))},
        "drawable_required_share": round(len(req_d) / n, 4),
        "drawable_required_and_present": round(sum(bool(r["drawable_found"]) for r in req_d) / max(1, len(req_d)), 4),
        "drawable_required_histogram": dict(collections.Counter(r["drawable"] for r in req_d).most_common()),
        "router_finds_drawable_share": round(1 - rc.get(None, 0) / n, 4),
        "router_class_histogram": {str(k): v for k, v in rc.most_common()},
        "router_picks_required_class": round(sum(r["router_class"] == r["drawable"].replace("teddy bear", "teddy-bear")
                                                 for r in req_d) / max(1, len(req_d)), 4),
        "required_word_used": {k: round(sum(r[k] for r in final) / n, 4) for k in ("has_noun", "has_verb", "has_adjective")},
        "once_upon_a_time_share": round(sum(r["story"].startswith("Once upon a time") for r in final) / n, 4),
        "stories_with_paragraph_breaks": sum("\n\n" in r["story"] for r in final),
        "stories_with_leak_sentence_trimmed": sum(1 for r in final if r["leak_trimmed"]),
        "distinct_words": len(wc),
        "top_words": wc.most_common(40),
        "top_content_words": content.most_common(60),
        "printer_word_counts": printer_hits.most_common(25),
        "split_seed": SPLIT_SEED,
        "files": {k: os.path.join(D, k) for k in ("printer_train.txt", "printer_val.txt", "printer_train.json",
                                                   "printer_val.json")},
        "format": "printer_*.txt: TinyStoriesV2-GPT4 layout (story, then a line '<|endoftext|>'); "
                  "printer_*.json: llama2.c tinystories.py shard layout (JSON array of {'story': ...} plus metadata)",
    }
    json.dump(stats, open(os.path.join(D, "printer_stats.json"), "w"), indent=1)
    with open(os.path.join(D, "printer_stats.txt"), "w") as fh:
        p = lambda *a: print(*a, file=fh)  # noqa: E731
        p(f"TONER-1 printer stories, built {__import__('time').strftime('%Y-%m-%d %H:%M %Z')}")
        p(f"writer model: {g.MODEL} -- {g.MODEL_LICENSE}")
        p(f"requests: {len(log)} logged, {len(ok)} ok; HTTP attempts {dict(att)}; usage.cost sum over ok requests "
          f"{stats['usage_cost_sum_ok_requests']} (non-zero: {stats['usage_cost_nonzero_requests']})")
        p(f"stories asked for in ok requests: {stats['stories_requested_in_ok_requests']}; returned: {slots_returned}")
        p(f"kept at generation: {len(kept0)}; rejected at generation: {len(rej0)} {stats['rejected_at_generation_by_reason']}")
        p(f"salvaged at build (leak sentence / title trimmed): {dict(salvaged)}; dropped at build: {stats['dropped_at_build']}")
        p(f"FINAL: {n} stories -> train {len(split['train'])}, val {len(split['val'])} (seeded 95/5 split)")
        p(f"by prompt version: {stats['by_prompt_version']}; by stories per request: {stats['by_per_request']}")
        p(f"words per story: {stats['words']}")
        if toks:
            p(f"tok512 tokens per story (with BOS): {stats['tok512_tokens_per_story']}")
        p(f"themes covered: {stats['themes_covered']}; per theme min {min(th.values())} max {max(th.values())}")
        p(f"drawable required: {stats['drawable_required_share']:.1%}; required one present: "
          f"{stats['drawable_required_and_present']:.1%}")
        p(f"TONER-1-V router finds a drawable word: {stats['router_finds_drawable_share']:.1%}; picks the required "
          f"class: {stats['router_picks_required_class']:.1%} (no router keyword exists for 'calculator')")
        p(f"router classes: {stats['router_class_histogram']}")
        p(f"required words used: {stats['required_word_used']}; 'Once upon a time' openings: "
          f"{stats['once_upon_a_time_share']:.1%}; distinct words: {len(wc)}")
        p(f"stories with paragraph breaks: {stats['stories_with_paragraph_breaks']}; with a leak sentence trimmed: "
          f"{stats['stories_with_leak_sentence_trimmed']}")
        p("theme histogram:")
        for i in sorted(range(len(g.THEMES)), key=lambda i: -th.get(i, 0)):
            p(f"  {th.get(i, 0):4d}  {g.THEMES[i]}")
        p("drawable required histogram: " + ", ".join(f"{k} {v}" for k, v in stats["drawable_required_histogram"].items()))
        p("top words: " + ", ".join(f"{k} {v}" for k, v in stats["top_words"]))
        p("top content words: " + ", ".join(f"{k} {v}" for k, v in stats["top_content_words"]))
        p("printer words: " + ", ".join(f"{k} {v}" for k, v in stats["printer_word_counts"]))
        p(f"format: {stats['format']}")
    print(open(os.path.join(D, "printer_stats.txt")).read())


if __name__ == "__main__":
    main()
