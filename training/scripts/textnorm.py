"""Shared text normalization for TONER-1 Pro tokenizer training and pretokenization.

TinyStoriesV2-GPT4 has curly quotes and dashes in about 2.6% of stories. The PostScript engine shows
raw bytes in Helvetica, so UTF-8 punctuation would print as junk. We map the common punctuation to
ASCII and drop the few stories that still contain non-ASCII characters afterwards.
"""
_MAP = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2018": "'", "\u2019": "'", "\u201a": "'",
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
    "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "-", "\x97": "-",
}
_TABLE = str.maketrans(_MAP)

def normalize(text):
    """Return the ASCII-normalized story, or None if it still has non-ASCII or control characters."""
    t = text.strip().translate(_TABLE).replace("\r", "")
    if not t or not t.isascii():
        return None
    if any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in t):
        return None
    return t

def iter_tinystories(path):
    """Yield raw story strings from a TinyStoriesV2 .txt file (stories separated by <|endoftext|> lines)."""
    buf = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("<|endoftext|>"):
                s = "".join(buf).strip()
                if s:
                    yield s
                buf = []
            else:
                buf.append(line)
    s = "".join(buf).strip()
    if s:
        yield s
