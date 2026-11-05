"""names_used.py: list the executable names a PostScript job uses that it never defines itself (so they
must come from the interpreter), skipping comments, (strings) and <hex> strings. Prints them sorted, so
they can be checked against the PLRM Level 1/2 operator lists.
usage: python -I names_used.py file.ps [file.ps ...]"""
import re, sys
used, defined = set(), set()
for path in sys.argv[1:]:
    src = open(path, encoding="latin-1").read()
    out, i, n = [], 0, len(src)
    while i < n:                      # strip comments, literal strings (nested, with escapes) and hex strings
        c = src[i]
        if c == "%":
            j = src.find("\n", i); i = n if j < 0 else j
        elif c == "(":
            depth = 1; i += 1
            while i < n and depth:
                if src[i] == "\\": i += 2; continue
                depth += {"(": 1, ")": -1}.get(src[i], 0); i += 1
            out.append(" ")
        elif c == "<" and src[i:i + 2] != "<<":
            j = src.find(">", i); i = j + 1; out.append(" ")
        else:
            out.append(c); i += 1
    toks = re.findall(r"(?<![0-9A-Za-z_.])/?(?:[A-Za-z=]|\.[A-Za-z])[A-Za-z0-9_.=\-]*", "".join(out))
    for t in toks:
        if t.startswith("/"): defined.add(t[1:])
        else: used.add(t)
print(" ".join(sorted(used - defined)))
