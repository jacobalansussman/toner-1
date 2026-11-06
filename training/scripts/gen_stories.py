#!/usr/bin/env python3
"""gen_stories.py: printer-themed TinyStories-style stories from a FREE OpenRouter model.

Recipe after TinyStories (Eldan and Li 2023): each story is built around random constraints, here a
printer theme (>= 40), one required noun, verb and adjective from simple-word lists, one or two story
features (dialogue, happy ending, moral, twist, bad ending, conflict, foreshadowing), an optional
character name, and in ~70% of stories one of TONER-1-V's 18 drawable things.
Several stories per request, in a strict delimiter format, parsed robustly.

Filters (dropped stories go to data/printer_stories_rejected.jsonl with the reason):
  ascii (after mapping curly quotes, dashes and ellipses to ASCII), format (no markdown, titles, lists,
  digits or symbols such as $ % @ = + & /),
  80..220 words, at least 3 printer/paper/ink/toner words, simple vocabulary (words outside TinyStories' own
  frequent words plus a printer whitelist), near-duplicates (sampled 6-word shingles).

Resumable: good stories are appended to data/printer_stories.jsonl; on start the file is re-read to
rebuild the duplicate index and the counts. Every request is logged to logs/gen_requests.jsonl
(prompt, latency, tokens, usage.cost, generation id, kept/dropped counts).

$0 guards: the model id must end in ':free' and /models must list it with zero pricing; any response
with usage.cost > 0 stops the run. Free-tier limits (OpenRouter docs, 2026-11-05): 20 requests/minute;
50/day below 10 credits purchased, 1000/day at or above. The run also reads free_model_daily_requests
from GET /key and stops before the daily ceiling.

Usage:
  python -I gen_stories.py --target 600 [--per-request 5] [--workers 3] [--dry-run]
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT as TRAIN  # noqa: E402
import or_free  # noqa: E402

MODEL = "xingchen-agi/xing4.0-29b-a4b:free"
MODEL_LICENSE = "Apache-2.0 (https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/blob/main/LICENSE)"
PROMPT_VERSION = "v8"   # v4 (bulk): drawable in the first three sentences + "no feature names" rule;
                        # v5: that rule dropped again (it primed the leaks); leaks are filtered instead
                        # v6: length rule says "12 to 18 short sentences"; short leak sentences and a bold
                        #     title line are trimmed instead of dropping the story (v6 made stories shorter)
                        # v7: length rule "120 to 180 words each; a story under 120 words is too short"
                        # v8: ask for 150 to 200 words (replies had drifted to ~100 words; filter stays 80..220)
OUT = os.path.join(TRAIN, "data/printer_stories.jsonl")
REJ = os.path.join(TRAIN, "data/printer_stories_rejected.jsonl")
LOG = os.path.join(TRAIN, "logs/gen_requests.jsonl")
VOCAB = os.path.join(TRAIN, "data/vocab/tinystories_wordfreq.tsv")
VOCAB_MIN_COUNT = 300        # a word is "simple" if it occurs >= this often in TinyStoriesV2-GPT4-train
MAX_HARD_WORDS = 2           # stories with more distinct non-simple words are dropped
MIN_WORDS, MAX_WORDS = 80, 220
RPM = 20                     # OpenRouter free-model limit
MIN_INTERVAL = 60.0 / RPM + 0.5
DAILY_RESERVE = 25           # stop when this few free requests remain today

# ---------------------------------------------------------------- prompt ingredients
THEMES = [
    "a printer that is shy",
    "a paper jam",
    "a printer that runs out of ink in the middle of a picture",
    "a printer that runs out of toner (say the word \"toner\")",
    "printing a picture for grandma",
    "an office at night, after everyone has gone home",
    "a printer that prints a bedtime story",
    "a tiny robot that fixes printers",
    "recycling old paper",
    "an empty paper tray",
    "a blank sheet of paper that wants to become a picture",
    "a new printer coming to a family's home",
    "an old printer that is very slow",
    "a printer that beeps too loudly",
    "a child printing a birthday card",
    "the printer at school printing pictures for the class",
    "a cat that likes to sleep on the warm printer",
    "a puppy that runs off with a printed page",
    "a paper airplane made from a printed page",
    "printing a treasure map",
    "a printer that can only print in one color",
    "a little drop of ink that wants to see the world",
    "a printer that prints the wrong picture by mistake",
    "a printer that learns to say sorry",
    "a tired printer that needs a rest",
    "two printers who become friends",
    "a printer in a library",
    "printing a poster to find a lost pet",
    "printing a thank-you note",
    "a printer that is afraid of the dark office",
    "a tall stack of paper that falls over",
    "a printer that hums a song when it prints",
    "a mouse who lives behind the office printer",
    "a broken printer and a kind helper",
    "printing and cutting paper snowflakes",
    "a printer that wants to see outside the window",
    "printing a picture of the whole family",
    "a big printer and a small printer",
    "a printer that prints a drawing a child made",
    "a printer that gets a name",
    "the very first page a new printer prints",
    "a printer and a scanner who are best friends",
    "a printer that will not stop printing pages",
    "a printer that wakes up at night",
    "printing signs for a lemonade stand",
    "a paper boat made from a printed page",
    "a lonely printer on the weekend",
    "printing a sunny picture on a rainy day",
    "a crumpled page that becomes flat again",
    "a printer, a rainy day and wet paper",
    "a printer that prints the same word again and again",
    "a printer that wants to draw like a crayon",
    "a busy office printer on a busy day",
    "a child who is allowed to press the print button",
    "printing a picture of a pet",
    "a printer that moves to a new room",
    "a printer that prints silly jokes",
    "a printer that is happy to be fixed",
    "the last sheet of paper in the tray",
    "printing a menu for a pretend restaurant",
    "a robot that brings paper to the printer",
    "printing a picture and coloring it in with crayons",
    "warm paper coming out of the printer on a cold morning",
    "a printer that prints invitations for a party",
    "a printer that helps a teacher",
    "a page that flies away out of an open window",
    "a printer's blinking light",
    "a printer that is too small for a big picture",
]

NOUNS = """ball box hat cup bed door window tree flower garden park house room toy book bag shoe
sock coat car bus boat train bike kite drum bell star moon sun cloud rain snow wind sky hill rock
stick leaf seed nut apple banana cookie cake bread milk juice soup egg pie candy jar bowl spoon
plate table chair desk lamp clock rug bath soap brush comb mirror key lock gate fence road path
river pond lake sea fish frog duck bird bug bee ant cat dog puppy kitten bunny horse cow pig
sheep hen fox bear lion mouse owl monkey robot doll teddy block crayon pencil sticker card gift
party song game friend family mom dad baby sister brother grandma grandpa teacher farmer cook
store shop school bench tower bridge cave nest shell sand wave ice button light picture letter
note map plant bucket basket blanket pillow wheel balloon bubble cloud""".split()
VERBS = """run jump walk hop skip swim fly climb sit stand sleep wake eat drink cook bake help
fix find look see watch hear listen sing dance play draw paint color write read count share give
take bring carry push pull open close shut catch throw kick roll spin shake wave smile laugh cry
hug kiss clap wish hope try learn build make cut fold glue stack pick plant water clean wash dry
hide seek call shout whisper ask answer thank wait follow chase race visit print beep blink
hum fit tear tape send grow""".split()
ADJS = """big small little tiny huge tall short long round flat soft hard smooth bumpy warm cold
hot cool wet dry clean dirty new old happy sad glad mad angry scared brave kind nice mean shy
proud silly funny quiet loud fast slow busy lazy sleepy tired hungry full empty heavy light bright
dark shiny dull pretty ugly red blue green yellow pink purple orange black white brown gray fluffy
sticky sweet sour fresh curious careful gentle strong weak lucky lonely friendly helpful special
messy neat sparkly colorful wild calm cozy""".split()
NAMES = """Lily Tom Ben Mia Sam Max Anna Tim Sue Jack Lucy Zoe Leo Ella Sara Mike Amy Joe Kim Bob
Ann Dan Rosa Jen Nick Pam Will Tess Ray Jill Ted Molly Kate Ollie Ruby Finn Ivy Noah Emma Lena
Pip Dot Beep Inky Sunny Bo Milo Nina Jade Rex""".split()
FEATURES = {
    "dialogue": "the story should contain at least one dialogue",
    "happy_ending": "the story has a happy ending",
    "moral": "the story teaches a simple lesson",
    "twist": "something unexpected happens",
    "bad_ending": "the story has a sad ending (gentle, nothing scary)",
    "conflict": "the story has some sort of conflict",
    "foreshadowing": "the story has a small hint early on of what will happen later",
}
DRAWABLES = ["cat", "dog", "bird", "fish", "duck", "rabbit", "frog", "teddy bear", "sun", "tree",
             "flower", "house", "car", "apple", "cake", "soccer ball", "light bulb", "calculator"]
DRAWABLE_SHARE = 0.70
NAME_SHARE = 0.35

# TONER-1-V's keyword router (pixelart/ps/src/story_page.ps): first word naming a class wins;
# whole words, case-insensitive, trailing "s" allowed. "calculator" has no router keyword there.
ROUTER = {"cat": "cat", "kitty": "cat", "kitties": "cat", "dog": "dog", "puppy": "dog", "puppies": "dog",
          "bird": "bird", "fish": "fish", "duck": "duck", "bunny": "rabbit", "bunnies": "rabbit",
          "rabbit": "rabbit", "frog": "frog", "bear": "teddy-bear", "teddy": "teddy-bear",
          "teddies": "teddy-bear", "sun": "sun", "tree": "tree", "flower": "flower", "house": "house",
          "home": "house", "car": "car", "apple": "apple", "cake": "cake", "ball": "soccer ball",
          "lamp": "light bulb", "light": "light bulb"}
DRAWABLE_WORDS = {"cat": ["cat", "kitty", "kitten"], "dog": ["dog", "puppy"], "bird": ["bird"],
                  "fish": ["fish"], "duck": ["duck"], "rabbit": ["rabbit", "bunny"], "frog": ["frog"],
                  "teddy bear": ["teddy", "bear"], "sun": ["sun"], "tree": ["tree"], "flower": ["flower"],
                  "house": ["house"], "car": ["car"], "apple": ["apple"], "cake": ["cake"],
                  "soccer ball": ["ball"], "light bulb": ["bulb", "light", "lamp"], "calculator": ["calculator"]}

PRINTER_WORDS = re.compile(r"\b(print\w*|paper\w*|ink|inky|toner|copier\w*|scanner\w*|tray|trays|jam|jammed|"
                           r"cartridge\w*)\b", re.I)
MIN_PRINTER_MENTIONS = 3     # the printer or the paper must matter, not just appear once
# the prompt's feature names leaking into the story text ("Something unexpected happened.", "a hint")
FEATURE_LEAK = re.compile(r"\b(something unexpected|unexpected(ly)? happened|unexpected thing|a hint|hint of|"
                          r"foreshadow\w*|happy ending|sad ending|plot twist|the twist|a twist)\b", re.I)
WHITELIST = set("""printer printers printer's print prints printed printing printout printouts reprint
reprinted toner toner's ink inks inky cartridge cartridges tray trays jam jams jammed jamming unjam
scanner scanners copier copiers photocopy copy copies copied stapler staplers staple staples stapled
crumpled crumple crumples uncrumple flatten flattened recycle recycles recycling recycled calculator
calculators beeped beeping beeps whir whirred whirring whirr whirrs hummed humming clunk clunked
clack clacked paperclip paperclips sheet sheets snowflake snowflakes lemonade okay ok""".split())

SYSTEM = ("You write very short, simple stories for small children, in the style of the TinyStories "
          "dataset. You always follow the requested output format exactly.")


def build_prompt(specs):
    n = len(specs)
    lines = [
        f"Write {n} short stories that a 3-year-old can understand, using only very simple words.",
        "",
        "Rules for every story:",
        "- 150 to 200 words each. Simple sentences. Only words a 3-year-old knows.",
        "- Every story is about printers, printing or paper (see its topic below). The printer or the "
        "paper matters in the story from the beginning to the end; it is not just mentioned once.",
        "- In every story someone prints something, or printing goes wrong, and that is what the story "
        "is about.",
        "- Each story must make sense: a clear beginning, middle and end, and things happen for a "
        "reason, the way they could in a child's day. Use the required words in a natural way; if a "
        "word does not fit well, use it in a small side moment.",
        "- A printer can be a normal machine or, sometimes, a friendly machine that can talk or feel, "
        "but it stays on its table and does printer things: it hums, beeps, takes paper and prints. "
        "It does not walk, eat or drive.",
        "- Write like a children's picture book: plain, warm and concrete. No big words, no grown-up "
        "things (no money, jobs, bosses or meetings), no technical words.",
        "- Do not start the stories the same way. Use \"Once upon a time\" in at most two of them. "
        "Each story has its own characters.",
        "- Plain text only: no title, no headings, no bold, no lists, no emoji. Straight quotes only.",
        "",
    ]
    for i, s in enumerate(specs, 1):
        parts = [f"Story {i}: about {s['theme']}.",
                 f"Use the noun \"{s['noun']}\", the verb \"{s['verb']}\" and the adjective \"{s['adjective']}\"."]
        if s.get("drawable"):
            art = "an" if s["drawable"][0] in "aeiou" else "a"
            parts.append(f"{art.capitalize()} {s['drawable']} is in the story (it can be printed in a picture); "
                         f"the word \"{s['drawable']}\" must appear in the first three sentences.")
        if s.get("name"):
            parts.append(f"One character is named {s['name']}.")
        parts.append("The story should have the following features: " +
                     ", ".join(FEATURES[f] for f in s["features"]) + ".")
        lines.append(" ".join(parts))
    lines += ["", "Output format, exactly:",
              "### STORY 1", "<story 1 text>", "### STORY 2", "<story 2 text>", "...",
              f"### STORY {n}", f"<story {n} text>", "### END",
              "Write nothing before the first ### line and nothing after ### END. "
              "Remember to only use simple words!"]
    return "\n".join(lines)


def make_specs(rng, n, theme_counts):
    specs = []
    # favour the least-used themes so coverage stays even
    pool = sorted(range(len(THEMES)), key=lambda t: (theme_counts.get(t, 0), rng.random()))
    chosen = pool[:n]
    rng.shuffle(chosen)
    for t in chosen:
        feats = rng.sample(sorted(FEATURES), rng.choice([1, 1, 2, 2, 2]))
        if "happy_ending" in feats and "bad_ending" in feats:
            feats.remove("bad_ending")
        s = {"theme_id": t, "theme": THEMES[t], "noun": rng.choice(NOUNS), "verb": rng.choice(VERBS),
             "adjective": rng.choice(ADJS), "features": feats,
             "drawable": rng.choice(DRAWABLES) if rng.random() < DRAWABLE_SHARE else None,
             "name": rng.choice(NAMES) if rng.random() < NAME_SHARE else None}
        specs.append(s)
    return specs


# ---------------------------------------------------------------- parsing and filters
ASCII_MAP = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
             "“": '"', "”": '"', "„": '"', "″": '"', "–": "-", "—": " - ",
             "―": " - ", "−": "-", "…": "...", " ": " ", " ": " ", "​": ""}


def to_ascii(t):
    for k, v in ASCII_MAP.items():
        t = t.replace(k, v)
    return t


def parse_reply(text, n):
    text = to_ascii(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.split("### END")[0]
    parts = re.split(r"^\s*#{2,4}\s*STORY\s*(\d*)\s*:?\s*$", text, flags=re.M | re.I)
    out = {}
    numbered = all(parts[k] for k in range(1, len(parts) - 1, 2))
    for j, k in enumerate(range(1, len(parts) - 1, 2), 1):
        idx = int(parts[k]) if numbered else j   # headers without numbers: take them in order
        body = parts[k + 1].strip()
        if 1 <= idx <= n and body and idx not in out:
            out[idx] = body
    return out


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def load_vocab():
    v = set()
    with open(VOCAB) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            w, c = line.rstrip("\n").split("\t")
            if int(c) >= VOCAB_MIN_COUNT:
                v.add(w)
    return v


def trim_leaks(text):
    """Remove short sentences that only echo a prompt feature ("A happy ending for the duck!",
    "Then something unexpected happened."). Returns (text, number of sentences removed)."""
    if not FEATURE_LEAK.search(text):
        return text, 0
    out, n = [], 0
    for para in text.split("\n\n"):
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', para)
        keep = []
        for snt in sents:
            if FEATURE_LEAK.search(snt) and len(WORD_RE.findall(snt)) <= 10 and '"' not in snt:
                n += 1
                continue
            keep.append(snt)
        if keep:
            out.append(" ".join(keep))
    return "\n\n".join(out), n


def clean_story(body):
    """Normalize whitespace; returns text or raises ValueError(reason)."""
    body = body.strip()
    # a bold or plain title line such as "**Story 1**" or "**The Shy Printer**" on its own first line
    body = re.sub(r"^\*\*[^*\n]{1,60}\*\*[ \t]*\n", "", body).strip()
    body = re.sub(r"\*\*([^*\n]{1,80})\*\*", r"\1", body)   # leftover bold emphasis
    if re.search(r"[*#_`~|<>\[\]{}]", body):
        raise ValueError("format:markup")
    if re.search(r"[0-9$%@=+&/\\]", body):
        raise ValueError("format:digits_or_symbols")
    if re.match(r"(?i)^(title|story)\b", body):
        raise ValueError("format:title")
    if re.search(r"^\s*([-+]|\d+[.)])\s", body, flags=re.M):
        raise ValueError("format:list")
    paras = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
    paras = [re.sub(r"\s*\n\s*", " ", p) for p in paras if p]
    text = "\n\n".join(paras)
    if any(ord(c) > 126 or (ord(c) < 32 and c != "\n") for c in text):
        raise ValueError("non_ascii")
    return text


def hard_words(text, vocab):
    hard, names = set(), set()
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        lw = w.lower()
        base = lw[:-2] if lw.endswith("'s") else lw
        if lw in vocab or base in vocab or lw in WHITELIST or base in WHITELIST:
            continue
        if w[0].isupper():
            names.add(w)        # a capitalized unknown word is taken as a name
            continue
        hard.add(lw)
    return hard, names


def router(text):
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group(0).lower()
        if w in ROUTER:
            return ROUTER[w], m.group(0)
        if len(w) >= 2 and w.endswith("s") and w[:-1] in ROUTER:
            return ROUTER[w[:-1]], m.group(0)
    return None, None


def has_word(text, word):
    lw = text.lower()
    return re.search(r"\b" + re.escape(word.lower()) + r"(s|es|ed|d|ing|ies)?\b", lw) is not None


class Dedup:
    """Near-duplicate index: 6-word shingles, keeping the ~1/4 whose hash is 0 mod 4."""
    K, MOD = 6, 4

    def __init__(self):
        self.idx = {}
        self.n = 0

    def shingles(self, text):
        ws = [w.lower() for w in WORD_RE.findall(text)]
        out = set()
        for i in range(len(ws) - self.K + 1):
            h = int(hashlib.md5(" ".join(ws[i:i + self.K]).encode()).hexdigest()[:12], 16)
            if h % self.MOD == 0:
                out.add(h)
        return out

    def is_dup(self, text, thresh=0.5):
        sh = self.shingles(text)
        if not sh:
            return False
        hits = {}
        for h in sh:
            for sid in self.idx.get(h, ()):
                hits[sid] = hits.get(sid, 0) + 1
        return bool(hits) and max(hits.values()) / len(sh) >= thresh

    def add(self, text):
        self.n += 1
        for h in self.shingles(text):
            self.idx.setdefault(h, []).append(self.n)


# ---------------------------------------------------------------- run
class Runner:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.vocab = load_vocab()
        self.dedup = Dedup()
        self.good = 0
        self.theme_counts = {}
        self.req_index = 0
        self.stop = None
        self.next_slot = 0.0
        self.free_remaining = None
        self.requests_this_run = 0
        self.zero_cost_requests = 0
        self.t_start = time.time()
        if os.path.exists(OUT):
            with open(OUT) as fh:
                for line in fh:
                    r = json.loads(line)
                    self.dedup.add(r["text"])
                    self.good += 1
                    self.theme_counts[r["theme_id"]] = self.theme_counts.get(r["theme_id"], 0) + 1
        if os.path.exists(LOG):
            with open(LOG) as fh:
                self.req_index = sum(1 for _ in fh)

    def wait_slot(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_slot)
            self.next_slot = t + MIN_INTERVAL
        if t > now:
            time.sleep(t - now)

    def check_daily(self):
        info = or_free.key_info()
        f = info.get("free_model_daily_requests") or {}
        self.free_remaining = f.get("remaining")
        if self.free_remaining is not None and self.free_remaining <= DAILY_RESERVE:
            self.stop = f"free-model daily requests nearly used up ({f})"
        return info

    def one_request(self):
        with self.lock:
            if self.stop or self.good >= self.a.target:
                return False
            ri = self.req_index
            self.req_index += 1
            rng = random.Random(f"{self.a.seed}-{ri}")
            specs = make_specs(rng, self.a.per_request, self.theme_counts)
            for s in specs:  # reserve themes so parallel workers spread out
                self.theme_counts[s["theme_id"]] = self.theme_counts.get(s["theme_id"], 0) + 1
        prompt = build_prompt(specs)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        rec = {"req": ri, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL,
               "prompt_version": PROMPT_VERSION, "prompt": prompt, "specs": specs}
        d = None
        for attempt in range(self.a.max_attempts):
            if self.stop:
                break
            self.wait_slot()
            if self.stop:
                break
            t0 = time.time()
            try:
                st, d, h = or_free.chat(MODEL, msgs, max_tokens=max(3000, 330 * len(specs)), temperature=0.8,
                                        extra={"top_p": 0.95, "reasoning": {"enabled": False}}, timeout=100)
            except or_free.CostError as e:
                self.stop = f"COST GUARD: {e}"
                rec.update(status="cost_error", error=str(e))
                break
            with self.lock:
                self.requests_this_run += 1
            rec.setdefault("attempts", []).append({"t": time.strftime("%H:%M:%S", time.localtime(t0)), "status": st,
                                                   "latency_s": round(time.time() - t0, 1),
                                                   "error": (d.get("error") or "")[:300] if st != 200 else ""})
            if st == 200 and d.get("choices"):
                break
            err = (d.get("error") or "") if isinstance(d, dict) else ""
            if st == 429 and ("free-models-per-day" in err or "per-day" in err or "daily" in err.lower()):
                self.stop = "daily free-model limit reached (429)"
                break
            if st in (401, 402, 403):
                self.stop = f"HTTP {st}: {err[:200]}"
                break
            if st == 429 and self.a.retry_sleep:   # upstream pool 429s: retry soon; wait_slot() keeps <= RPM
                back = self.a.retry_sleep * (0.7 + 0.6 * random.random())
            else:
                back = min(120, 6 * 2 ** attempt) * (0.7 + 0.6 * random.random())
            end = time.time() + back
            while time.time() < end and not self.stop:   # interruptible, so a time budget ends cleanly
                time.sleep(min(1.0, end - time.time()))
        kept, dropped = 0, {}
        if d and d.get("choices") and not rec.get("status"):
            u = d.get("usage") or {}
            rec.update(status="ok", provider=d.get("provider"), generation_id=d.get("id"),
                       prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       cost=u.get("cost"), finish_reason=d["choices"][0].get("finish_reason"))
            if u.get("cost") == 0:
                with self.lock:
                    self.zero_cost_requests += 1
            content = (d["choices"][0].get("message") or {}).get("content") or ""
            stories = parse_reply(content, len(specs))
            if stories and d["choices"][0].get("finish_reason") != "stop":
                stories.pop(max(stories))   # reply cut off: the last story may be unfinished
                rec["truncated"] = True
            if not stories:
                rec["raw_head"] = content[:400]
            for i, s in enumerate(specs, 1):
                reason, text, extra = None, None, {}
                body = stories.get(i)
                if body is None:
                    reason = "missing"
                else:
                    try:
                        text = clean_story(body)
                    except ValueError as e:
                        reason = str(e)
                if text is not None and reason is None:
                    text, n_trim = trim_leaks(text)
                    nw = len(WORD_RE.findall(text))
                    hard, names = hard_words(text, self.vocab)
                    extra = {"words": nw, "hard_words": sorted(hard), "names": sorted(names), "leak_trimmed": n_trim}
                    if nw < MIN_WORDS:
                        reason = "too_short"
                    elif nw > MAX_WORDS:
                        reason = "too_long"
                    elif len(PRINTER_WORDS.findall(text)) < MIN_PRINTER_MENTIONS:
                        reason = "off_theme"
                    elif FEATURE_LEAK.search(text):
                        reason = "feature_leak"
                    elif len(hard) > MAX_HARD_WORDS:
                        reason = "hard_words"
                    elif len(names) > 3:
                        reason = "too_many_unknown_names"
                with self.lock:
                    if reason is None and self.dedup.is_dup(text):
                        reason = "near_duplicate"
                    if reason is None:
                        rc, rw = router(text)
                        row = {"id": hashlib.sha1(text.encode()).hexdigest()[:16], "text": text,
                               "words": extra["words"], **{k: s[k] for k in ("theme_id", "theme", "noun", "verb",
                                                                              "adjective", "features", "drawable", "name")},
                               "has_noun": has_word(text, s["noun"]), "has_verb": has_word(text, s["verb"]),
                               "has_adjective": has_word(text, s["adjective"]),
                               "drawable_found": (any(has_word(text, w) for w in DRAWABLE_WORDS[s["drawable"]])
                                                  if s["drawable"] else None),
                               "router_class": rc, "router_word": rw, "hard_words": extra["hard_words"],
                               "names": extra["names"], "leak_trimmed": extra["leak_trimmed"],
                               "model": MODEL, "provider": d.get("provider"),
                               "prompt_version": PROMPT_VERSION, "per_request": len(specs), "req": ri, "slot": i,
                               "generation_id": d.get("id"), "created": rec["time"]}
                        with open(OUT, "a") as fo:
                            fo.write(json.dumps(row) + "\n")
                        self.dedup.add(text)
                        self.good += 1
                        kept += 1
                    else:
                        dropped[reason] = dropped.get(reason, 0) + 1
                        with open(REJ, "a") as fo:
                            fo.write(json.dumps({"req": ri, "slot": i, "reason": reason, "spec": s,
                                                 "text": text if text is not None else (body or ""),
                                                 **extra, "prompt_version": PROMPT_VERSION}) + "\n")
                    if reason is not None:  # give back the reserved theme slot
                        self.theme_counts[s["theme_id"]] -= 1
        elif not rec.get("status"):
            rec["status"] = "failed"
            with self.lock:
                for s in specs:
                    self.theme_counts[s["theme_id"]] -= 1
        rec.update(kept=kept, dropped=dropped)
        with self.lock:
            with open(LOG, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
            print(f"req {ri}: {rec.get('status')} kept {kept} dropped {dropped} "
                  f"tok {rec.get('completion_tokens')} cost {rec.get('cost')} | good {self.good}/{self.a.target}",
                  flush=True)
            every = 40 if (self.free_remaining is None or self.free_remaining > 100) else 5
            if self.requests_this_run and self.requests_this_run % every == 0:
                self.check_daily()
        return True

    def worker(self):
        while not self.stop and self.good < self.a.target:
            if self.a.max_seconds and time.time() - self.t_start > self.a.max_seconds:
                self.stop = self.stop or "time budget reached (resumable)"
                break
            if not self.one_request():
                break
            if self.a.max_requests and self.requests_this_run >= self.a.max_requests:
                self.stop = "max-requests reached"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600, help="total good stories wanted in the jsonl")
    ap.add_argument("--per-request", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", default=f"printer-{PROMPT_VERSION}")
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=7)
    ap.add_argument("--retry-sleep", type=float, default=0,
                    help="fixed (jittered) sleep after an upstream 429 instead of exponential backoff")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop starting requests after this many seconds; in-flight ones finish (resumable)")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit (no network)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if a.dry_run:
        print(build_prompt(make_specs(random.Random(a.seed), a.per_request, {})))
        bad = [w for w in NOUNS + VERBS + ADJS if w not in load_vocab() and w not in WHITELIST]
        print("\nword-list entries outside the simple vocabulary:", bad)
        print("themes:", len(THEMES), "nouns:", len(NOUNS), "verbs:", len(VERBS), "adjectives:", len(ADJS))
        return
    or_free.assert_free(MODEL)
    r = Runner(a)
    info = r.check_daily()
    print(f"model {MODEL} is free; resume with {r.good} good stories, next request {r.req_index}; "
          f"free requests remaining today: {r.free_remaining}; key usage ${info.get('usage')}", flush=True)
    if r.stop:
        print("STOP:", r.stop)
        return
    ts = [threading.Thread(target=r.worker, daemon=True) for _ in range(max(1, a.workers))]
    for t in ts:
        t.start()
        time.sleep(1.0)
    for t in ts:
        t.join()
    info = or_free.key_info()
    print(f"done: good {r.good}; requests this run {r.requests_this_run} (zero-cost 200s {r.zero_cost_requests}); "
          f"stop reason: {r.stop or 'target reached'}; key usage ${info.get('usage')}; "
          f"free requests: {info.get('free_model_daily_requests')}", flush=True)


if __name__ == "__main__":
    main()
