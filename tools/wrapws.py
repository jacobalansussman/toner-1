"""wrapws.py: re-wrap a PostScript weights body (numbers, names, brackets and <hex> strings only; no
literal (...) strings, no comments) so no line exceeds 200 characters. Breaks go at an existing space
(turned into a newline) or, inside a long <hex> string, between two hex digits (white space inside
<...> is ignored, PLRM 3.2.2). PostScript reads the result as exactly the same program.
usage: python -I wrapws.py < in > out"""
import sys
src = sys.stdin.read()
assert "(" not in src and ")" not in src and "%" not in src, "literal strings or comments: not safe to wrap"
W = 200
out = []
for line in src.split("\n"):
    while len(line) > W:
        cut = line.rfind(" ", 0, W)
        if cut > 0:
            out.append(line[:cut]); line = line[cut + 1:]
            continue
        lt, gt = line.rfind("<", 0, W), line.rfind(">", 0, W)
        assert lt > gt, "no safe break point"
        out.append(line[:W]); line = line[W:]
    out.append(line)
sys.stdout.write("\n".join(out))
