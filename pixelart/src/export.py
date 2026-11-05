"""Export a TONER-1-V checkpoint for the print job.
Writes out/doodle_weights.ps (int8, one scale per row, same packing as toner1/export.py, all inside
/DOODLE), out/doodle_meta.json and out/reference_logits.json (float64 logits from the DEQUANTIZED
weights for 3 fixed sequences).
usage: python -I src/export.py <pixelart_root> <run_name>"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from classes import (CLASSES, BOS, EOS, PATCH0, CLASS0, VOCAB, IMG, P, GRID, NPATCH, SEQLEN,
                     PATCH_BITS)
from quant import load_model, extract, quantize_all, load_into_model, forward_seq, MATS

root, name = sys.argv[1], sys.argv[2]
out = os.path.join(root, "out")
m, st = load_model(root, name)
a = m.params
D, HID, L, NH, NKV, V, SEQ = a.dim, a.hidden_dim, a.n_layers, a.n_heads, a.n_kv_heads, a.vocab_size, a.max_seq_len
HS, KVD, KVMUL = D // NH, D * NKV // NH, NH // NKV
assert V == VOCAB and SEQ == SEQLEN and HS % 2 == 0
w = extract(m)
qp, dq = quantize_all(w)
nparams = V * D + L * (2 * D * D + 2 * D * KVD + 3 * D * HID + 2 * D) + D
assert nparams == sum(p.numel() for p in m.parameters())

# ------------------------------------------------------------------ PostScript weights
def wrapjoin(items, width=200):
    # space-join, breaking lines so no line exceeds ~width chars (DSC asks for <= 255); PostScript
    # treats the newlines as ordinary whitespace, so the packing is unchanged
    out, cur = [], 0
    for it in items:
        first = it.split("\n", 1)[0]
        if cur and cur + 1 + len(first) > width:
            out.append("\n"); cur = 0
        elif cur:
            out.append(" "); cur += 1
        out.append(it)
        cur = len(it.rsplit("\n", 1)[-1]) if "\n" in it else cur + len(it)
    return "".join(out)


def psarr(items, chunk=200):
    # same as toner1/export.py: build long arrays in chunks so the operand stack stays small
    if len(items) <= chunk:
        return "[" + wrapjoin(items) + "]"
    s = "%d array" % len(items)
    for i in range(0, len(items), chunk):
        s += " dup %d [%s] putinterval\n" % (i, wrapjoin(items[i:i + chunk]))
    return s


def realarr(v):
    return psarr(["%.6g" % x for x in v])


def hexrow(r):
    h = r.tobytes().hex()               # white space inside <...> is ignored (PLRM 3.2.2)
    return "<" + "\n".join(h[i:i + 128] for i in range(0, len(h), 128)) + ">"


def qmat(q, s_txt):
    # [ [scales] [<row hex> ...] ], bytes = int8 + 128
    return "[" + realarr(s_txt) + "\n" + psarr([hexrow(r) for r in q]) + "]"


ps = []
ps.append("%!PS-Adobe-3.0\n%%Title: TONER-1-V doodle weights (int8, one scale per row)\n")
ps.append(f"""% TONER-1-V: 16x16 one-bit pixel-art generator, llama2.c architecture, {nparams} parameters.
% Trained on Google Quick, Draw! (CC BY 4.0); model code after karpathy/llama2.c model.py (MIT).
% Layout. Defines exactly one name in the current dictionary: /DOODLE, a dictionary. Its keys reuse
% the story model's names with identical meaning and packing (toner1/export.py):
%   D HID NL NH NKV V SEQ HS KVD KVMUL NPARAMS      integers
%   EMB                       [ [V scales] [V strings of D bytes] ]   tied input/output embedding
%   WQ WK WV WO W1 W2 W3      [ NL x [ [rows scales] [rows strings of cols bytes] ] ]
%                             WQ D x D, WK/WV KVD x D, WO D x D, W1/W3 HID x D, W2 D x HID
%   RMSATT RMSFFN             [ NL x [D reals] ]      RMSF  [D reals]
% A matrix row r, column c dequantizes to ((byte r c) - 128) * (scale r); rows are output units, so
% out[r] = scale[r] * (sum_c byte[r][c]*x[c] - 128*sum_c x[c])  (engine2.ps /mv).
% Image vocabulary: BOS EOS PATCH0 NPATCHV CLASS0 NCLASS IMG PATCH GRID NPATCH CLASSES (see below).
% Token sequence: BOS, CLASS0+k (class k of CLASSES), then 64 patch tokens PATCH0+v, raster order
% over the 8x8 grid of 2x2 patches; v = 8*TL + 4*TR + 2*BL + 1*BR, bit set = ink.
% Sample patches only from tokens PATCH0 .. PATCH0+15, at temperature about 0.9 (argmax gives a blank doodle).
% Running it with engine2.ps's procedures (verified in Ghostscript next to the story weights): keep the
% story's /forward etc. in userdict; under  DOODLE begin  re-create the activation buffers so they get the
% doodle's sizes and live in DOODLE, then call the same /forward (names resolve through DOODLE first):
%   DOODLE begin
%     /x D zeros def /xb D zeros def /xb2 D zeros def /q D zeros def /hb HID zeros def /hb2 HID zeros def
%     /logits V zeros def /att SEQ zeros def /KC [ NL {{ SEQ array }} repeat ] def
%     /VC [ NL {{ SEQ array }} repeat ] def /isq 1 HS sqrt div def
%     BOS 0 forward  CLASS0 k add 1 forward   ... sample, then  tok pos forward  for pos 2..64 ...
%   end
% forward's scratch names (pos tok l j k v ...) are then also defined in DOODLE, not in userdict.
/DOODLE 200 dict def
DOODLE begin
""")
ps.append(f"/D {D} def /HID {HID} def /NL {L} def /NH {NH} def /NKV {NKV} def /V {V} def /SEQ {SEQ} def "
          f"/HS {HS} def /KVD {KVD} def /KVMUL {KVMUL} def /NPARAMS {nparams} def\n")
ps.append("/EMB " + qmat(*qp["emb"]) + " def\n")
for k in MATS:
    ps.append("/%s [\n" % k.upper())
    for l in range(L):
        ps.append(qmat(*qp[k][l]) + "\n")
    ps.append("] def\n")
ps.append("/RMSATT [" + "\n".join(realarr(r) for r in dq["rms_att"]) + "] def\n")
ps.append("/RMSFFN [" + "\n".join(realarr(r) for r in dq["rms_ffn"]) + "] def\n")
ps.append("/RMSF " + realarr(dq["rms_f"]) + " def\n")
ps.append(f"/BOS {BOS} def /EOS {EOS} def /PATCH0 {PATCH0} def /NPATCHV 16 def /CLASS0 {CLASS0} def "
          f"/NCLASS {len(CLASSES)} def\n/IMG {IMG} def /PATCH {P} def /GRID {GRID} def /NPATCH {NPATCH} def\n")
ps.append("/CLASSES [" + wrapjoin(["(%s)" % c for c in CLASSES]) + "] def\n")
ps.append("end\n")
ps_path = os.path.join(out, "doodle_weights.ps")
open(ps_path, "w").write("".join(ps))

# ------------------------------------------------------------------ quality checks
va = torch.from_numpy(np.load(os.path.join(root, "data", "proc", "val.npy"), allow_pickle=False).astype(np.int64))


@torch.no_grad()
def val_loss(model):
    tot = 0.0; n = 0
    for i in range(0, len(va), 1000):
        s = va[i:i + 1000]; x = s[:, :-1]; y = s[:, 1:].clone(); y[:, 0] = -1
        lg = model(x, y)
        tot += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), ignore_index=-1, reduction="sum").item()
        n += (y != -1).sum().item()
    return tot / n


vl_float = val_loss(m)
mq, _ = load_model(root, name)
load_into_model(mq, dq)
vl_int8 = val_loss(mq)
rel_err = {k: float(np.mean([np.linalg.norm(dq[k][l] - w[k][l]) / np.linalg.norm(w[k][l]) for l in range(L)]))
           for k in MATS}
rel_err["emb"] = float(np.linalg.norm(dq["emb"] - w["emb"]) / np.linalg.norm(w["emb"]))

# ------------------------------------------------------------------ reference logits
vnp = va.numpy()
def first_val(c):
    return vnp[vnp[:, 1] == CLASS0 + CLASSES.index(c)][0].tolist()


seqs = [("val_cat_0", first_val("cat")), ("val_house_0", first_val("house"))]
# greedy decoding check (argmax over patch tokens): for a sparse one-bit image model the blank patch
# usually wins every position, so TEMP 0 gives an empty doodle; recorded in the meta file.
g = [BOS, CLASS0 + CLASSES.index("apple")]
for _ in range(NPATCH):
    g.append(PATCH0 + int(np.argmax(forward_seq(dq, a, g)[-1][PATCH0:PATCH0 + 16])))
greedy_ink_patches = int(sum(t != PATCH0 for t in g[2:]))
# third reference: the model's own sample for "sun" at temperature 1.0 (numpy seed 0), patch tokens only
rng = np.random.default_rng(0)
g = [BOS, CLASS0 + CLASSES.index("sun")]
for _ in range(NPATCH):
    pl = forward_seq(dq, a, g)[-1][PATCH0:PATCH0 + 16]
    pr = np.exp(pl - pl.max()); pr /= pr.sum()
    g.append(PATCH0 + int(rng.choice(16, p=pr)))
seqs.append(("sample_sun_T1_seed0", g))

trace = {}
def tr(pos, l, x):
    if pos < 2:
        trace.setdefault(pos, []).append(x.tolist())

ref = []
for i, (nm, toks) in enumerate(seqs):
    assert len(toks) == SEQLEN
    lg = forward_seq(dq, a, toks, trace=tr if i == 0 else None)
    # cross-check against the vendored torch model (float64 weights; its RoPE/RMSNorm run in float32)
    mq64, _ = load_model(root, name); load_into_model(mq64, dq); mq64 = mq64.double()
    with torch.no_grad():
        tl = mq64(torch.tensor([toks]), torch.zeros(1, SEQLEN, dtype=torch.long))[0].numpy()
    diff = float(np.abs(tl - lg).max())
    assert np.isfinite(lg).all() and diff < 1e-3, diff
    pl = lg[:, PATCH0:PATCH0 + 16]
    srt = np.sort(pl, axis=1)
    ref.append(dict(name=nm, tokens=toks, logits=lg.tolist(),
                    argmax=[int(v) for v in lg.argmax(1)],
                    patch_argmax=[PATCH0 + int(v) for v in pl.argmax(1)],
                    min_patch_top2_gap=float((srt[1:, -1] - srt[1:, -2]).min()),
                    nll_nats_per_patch_token=float(-np.mean([lg[t][toks[t + 1]] - lg[t].max()
                                                             - np.log(np.exp(lg[t] - lg[t].max()).sum())
                                                             for t in range(1, SEQLEN - 1)])),
                    max_abs_diff_vs_torch=diff))
json.dump(dict(
    description="Full logits (V=36) at every position, float64, from the DEQUANTIZED int8 weights in "
                "doodle_weights.ps (int8 * printed scale; RMSNorm weights as printed).",
    convention="logits[t] is the model output after feeding tokens[t] at position t, with tokens[0..t-1] "
               "already in the KV cache (llama2.c forward(token, pos)); it predicts tokens[t+1]. "
               "logits[65] (after the last patch) is included for completeness.",
    forward="numpy float64, one token at a time, same operation order as llama2.c run.c / engine2.ps",
    tolerance_hint="PostScript reals are 32-bit floats: expect |diff| ~1e-4..1e-3 on logits, identical argmax "
                   "except where min_patch_top2_gap is tiny.",
    sequences=ref,
    trace=dict(sequence=0, note="residual stream x after each layer, positions 0 and 1 of sequence 0",
               residual_after_layer={str(k): v for k, v in trace.items()}),
), open(os.path.join(out, "reference_logits.json"), "w"))

# ------------------------------------------------------------------ meta
prep = json.load(open(os.path.join(root, "data", "proc", "prep_stats.json")))
meta = dict(
    name="TONER-1-V", checkpoint=name,
    model=dict(arch="llama2.c Transformer (RMSNorm eps 1e-5, RoPE theta 10000 on pairs (2i,2i+1) within "
                    "each head, grouped-query attention, SwiGLU, tied embeddings, no biases)",
               dim=D, hidden_dim=HID, n_layers=L, n_heads=NH, n_kv_heads=NKV, head_size=HS, kv_dim=KVD,
               kv_mul=KVMUL, vocab_size=V, max_seq_len=SEQ, params=nparams,
               forward_passes_per_doodle=SEQLEN - 1),
    vocab=dict(BOS=BOS, EOS=EOS, EOS_note="unused (sequences are fixed length)", patch_tokens=[PATCH0, PATCH0 + 15],
               class_tokens={c: CLASS0 + i for i, c in enumerate(CLASSES)},
               map=["BOS", "EOS"] + [f"patch{v}" for v in range(16)] + [f"class:{c}" for c in CLASSES]),
    classes=CLASSES,
    sequence="BOS, CLASS, then 64 patch tokens (length 66)",
    sampling=dict(restrict_to="patch tokens 2..17 (other tokens get small but nonzero probability)",
                  recommended_temperature=0.9, tested_temperatures=[1.0, 0.9, 0.8],
                  greedy_warning=f"TEMP 0 (argmax) is degenerate: greedy 'apple' has {greedy_ink_patches} "
                                 "non-blank patches out of 64"),
    image=dict(size=IMG, patch=P, grid=GRID, patch_order="raster over the 8x8 patch grid: token index i "
               "(0..63) is patch row i//8 (top first), column i%8 (left first)",
               patch_value="v = 8*TL + 4*TR + 2*BL + 1*BR; token = 2 + v; bit 1 = ink (dark)",
               patch_bits_msb_first=PATCH_BITS),
    preprocessing=dict(source="Quick, Draw! numpy_bitmap 28x28 uint8, ink high on 0 background",
                       downsample="exact area-weighted box filter 28->16 (1.75 px per output pixel)",
                       threshold=prep["threshold"], threshold_on="box-mean intensity 0..255, ink if >= threshold",
                       dilation=prep["dilate"], drawings_per_class=10000, val_frac=prep["val_frac"],
                       train_sequences=prep["train"], val_sequences=prep["val"]),
    training=dict(device=st["device"], steps=st["args"]["steps"], batch_size=st["args"]["bs"],
                  lr=st["args"]["lr"], min_lr=st["args"]["min_lr"], warmup=st["args"]["warmup"],
                  weight_decay=st["args"]["wd"], optimizer="AdamW betas (0.9, 0.95), grad clip 1.0, cosine",
                  epochs=st["epochs"], train_seconds=st["train_seconds"],
                  loss_note="cross-entropy on the 64 patch targets only (class token is given)",
                  val_nats_per_token_float=vl_float, val_bits_per_pixel_float=vl_float * NPATCH / 256 / math.log(2),
                  val_nats_per_token_int8=vl_int8, val_bits_per_pixel_int8=vl_int8 * NPATCH / 256 / math.log(2),
                  per_class_val_nats_per_token=st["per_class_val_nats_per_token"], history=st["history"]),
    quantization=dict(scheme="int8, one scale per row; scale = max|row|/127, byte = round(w/scale)+128 in 1..255, "
                             "scales and RMSNorm weights printed with %.6g", relative_frobenius_error=rel_err),
    postscript=dict(file="doodle_weights.ps", bytes=os.path.getsize(ps_path), dictionary="DOODLE",
                    names_added_to_current_dict=["DOODLE"],
                    packing="same as toner1/export.py: matrices are [ [row scales] [row strings] ], bytes = int8+128",
                    usage="keep engine2.ps procedures in userdict; inside DOODLE begin ... end re-create the buffers "
                          "x xb xb2 q (D), hb hb2 (HID), logits (V), att (SEQ), KC VC (NL x SEQ), isq, then call "
                          "the same /forward; verified by src/check_ps.py mode B",
                    keys=["D", "HID", "NL", "NH", "NKV", "V", "SEQ", "HS", "KVD", "KVMUL", "NPARAMS", "EMB", "WQ",
                          "WK", "WV", "WO", "W1", "W2", "W3", "RMSATT", "RMSFFN", "RMSF", "BOS", "EOS", "PATCH0",
                          "NPATCHV", "CLASS0", "NCLASS", "IMG", "PATCH", "GRID", "NPATCH", "CLASSES"]),
    credits=["Quick, Draw! dataset by Google, CC BY 4.0", "model.py from karpathy/llama2.c, MIT"],
)
judge = {}
for t in ["10", "09", "08"]:
    for q in ["", "_int8"]:
        f = os.path.join(out, f"samples_{name}{q}_n64_t{t}_judge.json")
        if os.path.exists(f):
            j = json.load(open(f))
            judge[f"{'int8' if q else 'float'}_T{t[0]}.{t[1]}"] = dict(accuracy=j["accuracy"], per_class=j["per_class"],
                                                                     n_per_class=j["n_per_class"])
cj = os.path.join(root, "ckpt", "judge_cnn.json")
if os.path.exists(cj):
    meta["recognizability"] = dict(
        method="small CNN trained on the real 16x16 training images classifies generated samples; "
               "accuracy on real validation images is the ceiling", real_val_ceiling=json.load(open(cj)),
        samples=judge)
meta["contact_sheets"] = sorted(f for f in os.listdir(out) if f.startswith(f"samples_{name}") and f.endswith(".png"))
json.dump(meta, open(os.path.join(out, "doodle_meta.json"), "w"), indent=1)
print(f"wrote {ps_path} ({os.path.getsize(ps_path)} bytes), params {nparams}")
print(f"val nats/token float {vl_float:.4f}  int8 {vl_int8:.4f}  (bits/pixel {vl_int8 * NPATCH / 256 / math.log(2):.4f})")
print("relative quantization error", {k: round(v, 4) for k, v in rel_err.items()})
for r in ref:
    print(r["name"], "max|numpy-torch|", f"{r['max_abs_diff_vs_torch']:.2e}", "min top2 gap", f"{r['min_patch_top2_gap']:.3f}",
          "nll", f"{r['nll_nats_per_patch_token']:.3f}")
