"""sampler_emulate.py: exact check of the PostScript doodle sampler.
Re-creates whole doodles in Python with (1) Ghostscript's rand (Park-Miller minimal standard,
x <- 16807 x mod 2^31-1, as srand/rand in PLRM implementations and Ghostscript's zmath.c),
(2) the float32 forward pass of f32_emulate.py, which is bit-identical to Ghostscript on the reference
sequences, and (3) dsample's float32 arithmetic (softmax over tokens 2..17 at DTEMP, one rand draw,
first cumulative sum above it). Then compares the 64 patch values with what the PostScript job printed
("D cls seed v1 .. v64" lines in a page log). Identical doodles mean the sampler, the masking and the
seeding (srand + 4 discarded draws) do exactly what they say.
usage: python -I pixelart/check/sampler_emulate.py <pixelart_root> <log with D lines> [max_doodles]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import f32_emulate as fe     # reads sys.argv[1] = pixelart root

f32 = np.float32
M = 2147483647
root = sys.argv[1]
log = sys.argv[2] if len(sys.argv) > 2 else None
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
W = fe.W
BOS, CLASS0, PATCH0, NPV = W["BOS"], W["CLASS0"], W["PATCH0"], W["NPATCHV"]


class PSRand:
    def __init__(self, seed):
        s = seed
        if s < 1: s = -(s % 0x7ffffffe) + 1
        elif s > 0x7ffffffe: s = 0x7ffffffe
        self.x = s
    def rand(self):
        self.x = (16807 * self.x) % M
        return self.x


def dsample(logits, rng, T):
    lg = logits[PATCH0:PATCH0 + NPV]
    dbv, dbest = f32(-1e30), 0
    for j, v in enumerate(lg):
        if v > dbv: dbv, dbest = v, j
    e = fe.ps_exp(fe.E, np.maximum((lg - dbv) / f32(T), f32(-80)))
    dsum = fe.seqsum(e)
    dr = (f32(rng.rand()) / f32(M)) * dsum
    cum, pick = f32(0), dbest
    for j in range(NPV):
        cum = cum + e[j]
        if cum > dr:
            pick = j; break
    return pick


def doodle(cls, seed, T=0.9):
    rng = PSRand(seed)
    for _ in range(4): rng.rand()
    KC = [[] for _ in range(fe.NL)]; VC = [[] for _ in range(fe.NL)]
    fe.forward_step(KC, VC, BOS, 0)
    tok, pv = CLASS0 + cls, []
    for pos in range(1, 65):
        v = dsample(fe.forward_step(KC, VC, tok, pos), rng, T)
        pv.append(v); tok = PATCH0 + v
    return pv


def main():
    rows = [ln.split() for ln in open(log) if ln.startswith("D ")][:limit]
    res = []
    for r in rows:
        cls, seed, ps = int(r[1]), int(r[2]), [int(t) for t in r[3:]]
        em = doodle(cls, seed)
        res.append(dict(cls=cls, seed=seed, identical=(em == ps), n_diff=sum(a != b for a, b in zip(em, ps))))
        print(res[-1], flush=True)
    print(f"{sum(x['identical'] for x in res)}/{len(res)} doodles identical")
    json.dump(res, open(os.path.join(root, "check", "sampler_emulate_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
