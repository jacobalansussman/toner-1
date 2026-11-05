"""Shared: load a TONER-1-V checkpoint, int8-per-row quantization identical to toner1/export.py,
and a float64 numpy forward pass that mirrors llama2.c run.c / engine2.ps step by step."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "vendor"))
import numpy as np
import torch
from model import ModelArgs, Transformer


def load_model(root, name, device="cpu"):
    ck = torch.load(os.path.join(root, "ckpt", name + ".pt"), map_location="cpu", weights_only=True)
    st = ck["stats"]
    m = Transformer(ModelArgs(**st["model_args"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), st


def g6(v):
    """The value a reader of '%.6g' text gets back (what the PostScript interpreter parses)."""
    return float("%.6g" % float(v))


def quantize_rows(W):
    """W (rows, cols) float -> (bytes uint8 in 1..255 with 128 = zero, scales float64 as printed).
    Exactly toner1/export.py: s = max|row|/127 (1e-8 if the row is all zero),
    byte = clamp(round(v/s) + 128, 1, 255); the scale is written with '%.6g'."""
    W = np.asarray(W, dtype=np.float64)
    m = np.abs(W).max(axis=1)
    m = np.where(m == 0, 1e-8, m)
    s = m / 127.0
    q = np.clip(np.rint(W / s[:, None]) + 128, 1, 255).astype(np.uint8)
    s_txt = np.array([g6(v) for v in s])
    return q, s, s_txt


def dequant(q, s_txt):
    return (q.astype(np.float64) - 128.0) * s_txt[:, None]


def extract(m):
    """Float weights in llama2.c layout (each matrix is (out_rows, in_cols), as mv expects)."""
    sd = {k: v.detach().double().cpu().numpy() for k, v in m.state_dict().items()}
    a = m.params
    L = a.n_layers
    w = dict(emb=sd["tok_embeddings.weight"],
             rms_att=[sd[f"layers.{l}.attention_norm.weight"] for l in range(L)],
             rms_ffn=[sd[f"layers.{l}.ffn_norm.weight"] for l in range(L)],
             rms_f=sd["norm.weight"])
    for k, n in [("wq", "attention.wq"), ("wk", "attention.wk"), ("wv", "attention.wv"),
                 ("wo", "attention.wo"), ("w1", "feed_forward.w1"), ("w2", "feed_forward.w2"),
                 ("w3", "feed_forward.w3")]:
        w[k] = [sd[f"layers.{l}.{n}.weight"] for l in range(L)]
    return w


MATS = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]


def quantize_all(w):
    """-> (qpack, deq): qpack holds bytes+printed scales per matrix; deq the float64 weights the
    PostScript engine effectively uses (int8 * printed scale; RMSNorm weights as printed)."""
    qp, dq = {}, {}
    q, s, st = quantize_rows(w["emb"]); qp["emb"] = (q, st); dq["emb"] = dequant(q, st)
    for k in MATS:
        qp[k], dq[k] = [], []
        for W in w[k]:
            q, s, st = quantize_rows(W); qp[k].append((q, st)); dq[k].append(dequant(q, st))
    for k in ["rms_att", "rms_ffn"]:
        dq[k] = [np.array([g6(v) for v in r]) for r in w[k]]
    dq["rms_f"] = np.array([g6(v) for v in w["rms_f"]])
    return qp, dq


def load_into_model(m, dq):
    """Copy dequantized float64 weights back into a torch Transformer (for sampling / val loss)."""
    sd = m.state_dict()
    L = m.params.n_layers
    new = {"tok_embeddings.weight": dq["emb"], "output.weight": dq["emb"], "norm.weight": dq["rms_f"]}
    for l in range(L):
        new[f"layers.{l}.attention_norm.weight"] = dq["rms_att"][l]
        new[f"layers.{l}.ffn_norm.weight"] = dq["rms_ffn"][l]
        for k, n in [("wq", "attention.wq"), ("wk", "attention.wk"), ("wv", "attention.wv"),
                     ("wo", "attention.wo"), ("w1", "feed_forward.w1"), ("w2", "feed_forward.w2"),
                     ("w3", "feed_forward.w3")]:
            new[f"layers.{l}.{n}.weight"] = dq[k][l]
    for k, v in new.items():
        sd[k] = torch.tensor(v, dtype=sd[k].dtype)
    m.load_state_dict(sd)
    return m


def rmsnorm(x, w):
    return x * (1.0 / np.sqrt(np.mean(x * x) + 1e-5)) * w


def rope(v, pos, hs):
    v = v.copy()
    for i in range(0, len(v), 2):
        hd = i % hs
        ang = pos * (1.0 / (10000.0 ** (hd / hs)))
        c, s = np.cos(ang), np.sin(ang)
        v0, v1 = v[i], v[i + 1]
        v[i] = v0 * c - v1 * s
        v[i + 1] = v0 * s + v1 * c
    return v


def forward_seq(dq, a, toks, trace=None):
    # numpy 2.0 + Apple Accelerate raises spurious FP-exception warnings on ordinary matmuls
    # ("divide by zero encountered in matmul" on random data); callers assert finiteness instead.
    with np.errstate(all="ignore"):
        return _forward_seq(dq, a, toks, trace)


def _forward_seq(dq, a, toks, trace=None):
    """float64, one token at a time with a KV cache, exactly the run.c / engine2.ps order.
    a: ModelArgs. Returns logits (len(toks), V). trace(pos, layer, x) is called after each layer."""
    D, L, NH, NKV = a.dim, a.n_layers, a.n_heads, a.n_kv_heads
    HS = D // NH; KVMUL = NH // NKV
    kc = [[] for _ in range(L)]; vc = [[] for _ in range(L)]
    out = []
    for pos, tok in enumerate(toks):
        x = dq["emb"][tok].copy()
        for l in range(L):
            xb = rmsnorm(x, dq["rms_att"][l])
            q = dq["wq"][l] @ xb; k = dq["wk"][l] @ xb; v = dq["wv"][l] @ xb
            q = rope(q, pos, HS); k = rope(k, pos, HS)
            kc[l].append(k); vc[l].append(v)
            K = np.stack(kc[l]); Vv = np.stack(vc[l])
            att_out = np.zeros(D)
            for h in range(NH):
                qo = h * HS; ko = (h // KVMUL) * HS
                sc = K[:, ko:ko + HS] @ q[qo:qo + HS] / np.sqrt(HS)
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                att_out[qo:qo + HS] = sc @ Vv[:, ko:ko + HS]
            x = x + dq["wo"][l] @ att_out
            xb = rmsnorm(x, dq["rms_ffn"][l])
            h1 = dq["w1"][l] @ xb; h3 = dq["w3"][l] @ xb
            x = x + dq["w2"][l] @ (h1 / (1.0 + np.exp(-h1)) * h3)
            if trace:
                trace(pos, l, x)
        x = rmsnorm(x, dq["rms_f"])
        out.append(dq["emb"] @ x)
    return np.array(out)
