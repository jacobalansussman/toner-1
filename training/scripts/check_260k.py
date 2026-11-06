#!/usr/bin/env python3
"""Check that Karpathy's stories260K.pt reproduces the logits of the shipped
checkpoints/stories260K.bin (llama2.c legacy v0 format).

Also provides load_bin(path) -> llama2.c Transformer, a loader from the .bin
(usable as a fallback when only the .bin exists).

Run:  .venv/bin/python -I scripts/check_260k.py
"""
import os, struct, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import L2C, KARPATHY  # noqa: E402
sys.path.insert(0, L2C)
from model import ModelArgs, Transformer  # noqa: E402  (llama2.c, MIT)

PT = os.path.join(KARPATHY, "stories260K.pt")
TOKM = os.path.join(KARPATHY, "tok512.model")
BIN = os.path.join(KARPATHY, "stories260K.bin")


def load_pt(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    args = ModelArgs(**ck["model_args"])
    m = Transformer(args)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval()
    return m, ck, missing, unexpected


def load_bin(path):
    """Load a llama2.c legacy (v0) .bin into llama2.c's Transformer."""
    raw = open(path, "rb").read()
    dim, hidden, L, H, KV, V, T = struct.unpack("7i", raw[:28])
    shared = V > 0
    V = abs(V)
    a = np.frombuffer(raw, dtype=np.float32, offset=28)
    hs = dim // H
    args = ModelArgs(dim=dim, n_layers=L, n_heads=H, n_kv_heads=KV, vocab_size=V,
                     hidden_dim=hidden, max_seq_len=T, dropout=0.0)
    m = Transformer(args)
    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        t = torch.from_numpy(a[off:off + n].copy()).view(*shape)
        off += n
        return t

    with torch.no_grad():
        m.tok_embeddings.weight.copy_(take(V, dim))
        for l in m.layers: l.attention_norm.weight.copy_(take(dim))
        for l in m.layers: l.attention.wq.weight.copy_(take(H * hs, dim))
        for l in m.layers: l.attention.wk.weight.copy_(take(KV * hs, dim))
        for l in m.layers: l.attention.wv.weight.copy_(take(KV * hs, dim))
        for l in m.layers: l.attention.wo.weight.copy_(take(dim, H * hs))
        for l in m.layers: l.ffn_norm.weight.copy_(take(dim))
        for l in m.layers: l.feed_forward.w1.weight.copy_(take(hidden, dim))
        for l in m.layers: l.feed_forward.w2.weight.copy_(take(dim, hidden))
        for l in m.layers: l.feed_forward.w3.weight.copy_(take(hidden, dim))
        m.norm.weight.copy_(take(dim))
        fc = take(T, hs // 2); fs = take(T, hs // 2)
        if not shared:
            m.output.weight = torch.nn.Parameter(take(V, dim))
    assert off == a.size, (off, a.size)
    m.eval()
    return m, dict(dim=dim, hidden=hidden, n_layers=L, n_heads=H, n_kv_heads=KV,
                   vocab=V, seq=T, shared=shared), fc, fs


def full_logits(m, ids):
    x = torch.tensor([ids])
    with torch.no_grad():
        h = m.tok_embeddings(x)
        for l in m.layers:
            h = l(h, m.freqs_cos[:x.shape[1]], m.freqs_sin[:x.shape[1]])
        return m.output(m.norm(h))[0]


def main():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=TOKM)
    text = "Once upon a time, there was a little printer named Toner. It loved to print pictures of cats for grandma."
    ids = [1] + sp.encode(text)
    mp, ck, missing, unexpected = load_pt(PT)
    mb, hdr, fc, fs = load_bin(BIN)
    print("pt model_args:", ck["model_args"])
    print("bin header:", hdr)
    print("pt missing keys:", missing, "unexpected:", unexpected)
    print("pt iter_num:", ck.get("iter_num"), "best_val_loss:", float(ck.get("best_val_loss", float("nan"))))
    npar = sum(p.numel() for p in mp.parameters())
    print("params (pt model):", npar)
    # parameter-level comparison
    sdp, sdb = mp.state_dict(), mb.state_dict()
    maxw = max((sdp[k].float() - sdb[k].float()).abs().max().item() for k in sdp)
    print("max |weight diff| over all tensors:", maxw)
    print("freqs_cos max diff:", (mp.freqs_cos[:hdr['seq']] - fc).abs().max().item(),
          "freqs_sin:", (mp.freqs_sin[:hdr['seq']] - fs).abs().max().item())
    lp, lb = full_logits(mp, ids), full_logits(mb, ids)
    d = (lp - lb).abs().max().item()
    agree = (lp.argmax(-1) == lb.argmax(-1)).float().mean().item()
    print(f"test sequence: {len(ids)} tokens; max |logit diff| = {d:.3e}; argmax agreement = {agree*100:.1f}%")
    # greedy continuation from BOS with both
    def greedy(m, n=40):
        x = [1]
        for _ in range(n):
            x.append(int(full_logits(m, x)[-1].argmax()))
        return sp.decode(x[1:])
    gp, gb = greedy(mp), greedy(mb)
    print("greedy (pt): ", gp)
    print("greedy match:", gp == gb)
    ok = d < 1e-4 and agree == 1.0 and gp == gb
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
