"""f32_emulate.py: is the PostScript-vs-float64 gap just float32 rounding?
Re-implements engine2.ps's forward pass in numpy with Ghostscript's arithmetic: every real is a 32-bit
float, every add/mul/div is rounded to float32 in the same order as the PostScript loops (sums are
sequential, via cumsum), and exp/sqrt/sin/cos are computed in double and rounded to float32, as
Ghostscript's operators do. The weights are parsed from out/doodle_weights.ps itself.
Then it compares three things on the 3 teacher-forced sequences of out/reference_logits.json:
  PS (Ghostscript, exact values from check/tf_A.ps)  vs  this emulation      -> should be ~0 (same arithmetic)
  this emulation                                     vs  float64 reference   -> the float32 rounding error
  PS                                                 vs  float64 reference   -> what teacher_force.py reports
usage: python -I pixelart/check/f32_emulate.py <pixelart_root>"""
import json, math, os, re, subprocess, sys
import numpy as np

root = sys.argv[1]
GS = os.environ.get("GS", "gs")
f32 = np.float32


# ---------- parse doodle_weights.ps (numbers, names, [ ], <hex>; comments stripped) ----------
def parse_weights(path):
    src = "\n".join(l.split("%", 1)[0] for l in open(path).read().splitlines())
    toks = re.findall(r"<[0-9a-fA-F\s]*>|\[|\]|/?[A-Za-z_][A-Za-z0-9_\-]*|-?\d+\.?\d*(?:[eE]-?\d+)?|\([^)]*\)", src)
    out, stack, cur = {}, [], None
    pending = None
    for t in toks:
        if t == "[":
            stack.append([])
        elif t == "]":
            v = stack.pop()
            (stack[-1].append(v) if stack else None)
            if not stack: cur = v
        elif t.startswith("<"):
            v = bytes.fromhex(re.sub(r"\s", "", t[1:-1]))
            (stack[-1].append(v) if stack else None)
        elif t.startswith("/"):
            if not stack: pending = t[1:]
            else: stack[-1].append(t[1:])
        elif t == "def":
            if pending is not None and pending != "DOODLE":
                out[pending] = cur
            pending = None
        elif t.startswith("("):
            if stack: stack[-1].append(t[1:-1])
        elif re.match(r"-?\d", t):
            v = float(t) if any(c in t for c in ".eE") else int(t)
            if stack: stack[-1].append(v)
            else: cur = v
        # other names (begin, end, dict, DOODLE) are ignored
    return out


W = parse_weights(os.path.join(root, "out", "doodle_weights.ps"))
D, HID, NL, NH, NKV, V, HS, KVMUL = (W[k] for k in ("D", "HID", "NL", "NH", "NKV", "V", "HS", "KVMUL"))
KVD = W["KVD"]


def mat(m):   # [ [scales] [row strings] ] -> (float32 scales, float32 bytes 0..255)
    s = np.array(m[0], dtype=np.float64).astype(f32)
    b = np.array([list(r) for r in m[1]], dtype=f32)
    return s, b


EMB = mat(W["EMB"])
L = {k: [mat(m) for m in W[k]] for k in ("WQ", "WK", "WV", "WO", "W1", "W2", "W3")}
RA = [np.array(r, dtype=np.float64).astype(f32) for r in W["RMSATT"]]
RF = [np.array(r, dtype=np.float64).astype(f32) for r in W["RMSFFN"]]
RFin = np.array(W["RMSF"], dtype=np.float64).astype(f32)
E = f32(2.718281828)


def seqsum(v):            # PostScript  0 v { add } forall  : sequential float32 sum
    return np.cumsum(v.astype(f32), axis=-1, dtype=f32)[..., -1] if v.shape[-1] else f32(0)


def ps_exp(base, e):      # Ghostscript exp: pow in double, stored as float
    return np.power(np.float64(base), np.asarray(e, dtype=np.float64)).astype(f32)


def rmsnorm(x, w):
    ss = seqsum(x * x)
    rs = f32(1) / f32(math.sqrt(float(ss / f32(len(x)) + f32(1e-5))))
    return (x * rs) * w


def mv(x, m):
    s, b = m
    msx = seqsum(x) * f32(128)
    acc = seqsum(b * x[None, :])
    return (acc - msx) * s


def deg_trig(ang):        # Ghostscript sin/cos take degrees: exact at multiples of 90, else double sin/cos
    a = float(ang)
    q = a / 90.0
    if math.floor(q) == q:
        k = int(q) % 4
        return f32([1, 0, -1, 0][k]), f32([0, 1, 0, -1][k])
    r = a * (math.pi / 180.0)
    return f32(math.cos(r)), f32(math.sin(r))


def rope(v, pos):
    v = v.copy()
    for i in range(0, len(v), 2):
        hd = i % HS
        ang = (f32(pos) / ps_exp(10000, f32(hd) / f32(HS))) * f32(57.29577951)
        c, s = deg_trig(ang)
        v0, v1 = v[i], v[i + 1]
        v[i] = v0 * c - v1 * s
        v[i + 1] = v0 * s + v1 * c
    return v


isq = f32(1) / f32(math.sqrt(HS))


def forward_step(KC, VC, tok, pos):
    """engine2.ps /forward for one token: appends to the KV caches KC, VC (lists per layer), returns logits."""
    es, er = EMB[0][tok], EMB[1][tok]
    x = (er - f32(128)) * es
    for l in range(NL):
        xb = rmsnorm(x, RA[l])
        q = mv(xb, L["WQ"][l]); k = mv(xb, L["WK"][l]); v = mv(xb, L["WV"][l])
        q = rope(q, pos); k = rope(k, pos)
        KC[l].append(k); VC[l].append(v)
        out = np.zeros(D, dtype=f32)
        Kt = np.stack(KC[l][:pos + 1]); Vt = np.stack(VC[l][:pos + 1])
        for h in range(NH):
            qo, ko = h * HS, (h // KVMUL) * HS
            sc = seqsum(q[qo:qo + HS][None, :] * Kt[:, ko:ko + HS]) * isq
            amx = sc.max()
            ex = ps_exp(E, np.maximum(sc - amx, f32(-80)))
            ssum = seqsum(ex)
            for j in range(HS):
                out[qo + j] = seqsum(ex * Vt[:, ko + j]) / ssum
        x = x + mv(out, L["WO"][l])
        xb = rmsnorm(x, RF[l])
        hb = mv(xb, L["W1"][l]); hb2 = mv(xb, L["W3"][l])
        sig = hb / (f32(1) + ps_exp(E, np.maximum(-hb, f32(-80))))
        x = x + mv(sig * hb2, L["W2"][l])
    x = rmsnorm(x, RFin)
    return mv(x, EMB)


def run(tokens):
    KC = [[] for _ in range(NL)]; VC = [[] for _ in range(NL)]
    return np.array([forward_step(KC, VC, tok, pos).astype(np.float64) for pos, tok in enumerate(tokens)])


def main():
    # ---------- PS logits, printed exactly by check/tf_A.ps (written by teacher_force.py) ----------
    r = subprocess.run([GS, "-q", "-dNODISPLAY", "-dBATCH", "-dNOPAUSE", os.path.join(root, "check", "tf_A.ps")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "Error" not in r.stdout, r.stdout[-2000:]


    def pe(t):
        neg = t.startswith("-"); t = t.lstrip("-")
        if t == "0": return 0.0
        m, k = t.split("p"); v = int(m) * 2.0 ** -int(k)
        return -v if neg else v


    ps, cur = [], None
    for ln in r.stdout.splitlines():
        if ln.startswith("SEQ "): cur = []; ps.append(cur)
        elif ln.startswith("L "): cur.append([pe(t) for t in ln[2:].split()])
    ref = json.load(open(os.path.join(root, "out", "reference_logits.json")))["sequences"]
    res = []
    for sq, p in zip(ref, ps):
        p = np.array(p); r64 = np.array(sq["logits"]); emu = run(sq["tokens"])
        res.append(dict(sequence=sq["name"],
                        ps_vs_emulation_max_abs=float(np.abs(p - emu).max()),
                        ps_vs_emulation_bit_identical_fraction=float((p == emu).mean()),
                        emulation_vs_float64_max_abs=float(np.abs(emu - r64).max()),
                        ps_vs_float64_max_abs=float(np.abs(p - r64).max())))
        print(res[-1])
    json.dump(res, open(os.path.join(root, "check", "f32_emulate_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
