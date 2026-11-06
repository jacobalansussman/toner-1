#!/usr/bin/env python3
"""l2c_safetensors.py: convert a llama2.c legacy (version 0) checkpoint .bin to safetensors and back.
Standard library only; nothing here uses the network.

  python3 l2c_safetensors.py to-safetensors MODEL.bin OUT_DIR NAME TOKENIZER
      writes OUT_DIR/model.safetensors (float32, llama2.c model.py state_dict names) and OUT_DIR/config.json
  python3 l2c_safetensors.py to-bin DIR OUT.bin
      reads DIR/model.safetensors + DIR/config.json and writes a llama2.c v0 .bin, the format export.py reads

A v0 .bin is 7 int32 (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len; vocab_size is
negative when the output head is not shared with the embedding), then float32 arrays in this order:
tok_embeddings, attention_norm x L, wq x L, wk x L, wv x L, wo x L, ffn_norm x L, w1 x L, w2 x L, w3 x L, norm,
freqs_cos, freqs_sin (the RoPE tables), and output only if it is not shared.

The embedding is shared with the output head, so model.safetensors stores it once, as tok_embeddings.weight; with
llama2.c's model.py, load_state_dict(..., strict=False) reports output.weight as missing and the tie fills it.
The RoPE tables are not weights and are not stored: to-bin recomputes them, emulating float32 step by step. They can
differ from PyTorch's in the last bit, so the .bin need not be byte-identical to the original; every weight byte is,
and export.py, which reads only the weights, writes the same weights.ps.
"""
import json
import math
import os
import struct
import sys
from array import array

assert sys.byteorder == "little", "this tool assumes a little-endian machine (as llama2.c's .bin does)"

HDR = ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "max_seq_len")


def layout(c):
    """(name, shape) of every stored tensor, in v0 .bin order."""
    d, h, L, nh, nkv, V = c["dim"], c["hidden_dim"], c["n_layers"], c["n_heads"], c["n_kv_heads"], c["vocab_size"]
    kvd = d * nkv // nh
    out = [("tok_embeddings.weight", [V, d])]
    per = [("attention_norm", "attention_norm.weight", [d]), ("wq", "attention.wq.weight", [d, d]),
           ("wk", "attention.wk.weight", [kvd, d]), ("wv", "attention.wv.weight", [kvd, d]),
           ("wo", "attention.wo.weight", [d, d]), ("ffn_norm", "ffn_norm.weight", [d]),
           ("w1", "feed_forward.w1.weight", [h, d]), ("w2", "feed_forward.w2.weight", [d, h]),
           ("w3", "feed_forward.w3.weight", [h, d])]
    for _, key, shape in per:
        out += [(f"layers.{i}.{key}", shape) for i in range(L)]
    out.append(("norm.weight", [d]))
    return out


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def rope_tables(c):
    """llama2.c precompute_freqs_cis(head_size, max_seq_len), each step rounded to float32."""
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    hs = c["dim"] // c["n_heads"]
    fr = [f32(1.0 / f32(10000.0 ** f32(f32(2 * i) / hs))) for i in range(hs // 2)]
    cos, sin = array("f"), array("f")
    for t in range(c["max_seq_len"]):
        for f in fr:
            a = f32(t * f)
            cos.append(math.cos(a)); sin.append(math.sin(a))
    return cos, sin


def to_safetensors(bin_path, out_dir, name, tokenizer):
    raw = open(bin_path, "rb").read()
    c = dict(zip(HDR, struct.unpack("7i", raw[:28])))
    shared = c["vocab_size"] > 0
    assert shared, "unshared output heads are not supported (TONER-1 models share them)"
    f = array("f"); f.frombytes(raw[28:])
    lay = layout(c)
    n_w = sum(numel(s) for _, s in lay)
    hs = c["dim"] // c["n_heads"]
    assert len(f) == n_w + c["max_seq_len"] * hs, (len(f), n_w)
    header, off = {}, 0
    for key, shape in lay:
        nb = 4 * numel(shape)
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    header["__metadata__"] = {"format": "pt", "source": "llama2.c v0 .bin, converted by l2c_safetensors.py"}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.safetensors"), "wb") as o:
        o.write(struct.pack("<Q", len(hj))); o.write(hj); o.write(raw[28:28 + 4 * n_w])
    cfg = {
        "model_name": name,
        "architecture": "llama2.c Transformer (Llama-2-style decoder: RMSNorm, RoPE, grouped-query attention, "
                        "SwiGLU feed-forward, tied input/output embedding)",
        "code": "https://github.com/karpathy/llama2.c/blob/master/model.py",
        "model_args": {"dim": c["dim"], "n_layers": c["n_layers"], "n_heads": c["n_heads"],
                       "n_kv_heads": c["n_kv_heads"], "vocab_size": c["vocab_size"], "hidden_dim": c["hidden_dim"],
                       "multiple_of": 1, "norm_eps": 1e-05, "max_seq_len": c["max_seq_len"], "dropout": 0.0},
        "head_dim": hs,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "n_params": n_w,
        "torch_dtype": "float32",
        "tokenizer": tokenizer,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "weights": "model.safetensors holds the float32 weights. output.weight is not stored: it is "
                   "tok_embeddings.weight (load with strict=False). The print job runs an int8 copy, one scale "
                   "per row, written by export.py.",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as o:
        json.dump(cfg, o, indent=2); o.write("\n")
    print(f"{out_dir}/model.safetensors: {len(lay)} tensors, {n_w:,} params, "
          f"{os.path.getsize(os.path.join(out_dir, 'model.safetensors')):,} bytes")


def to_bin(in_dir, out_path):
    cfg = json.load(open(os.path.join(in_dir, "config.json")))
    c = {k: cfg["model_args"][k] for k in HDR}
    raw = open(os.path.join(in_dir, "model.safetensors"), "rb").read()
    (hl,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hl])
    base = 8 + hl
    with open(out_path, "wb") as o:
        o.write(struct.pack("7i", *(c[k] for k in HDR)))
        for key, shape in layout(c):
            t = header[key]
            assert t["dtype"] == "F32" and t["shape"] == shape, (key, t)
            a, b = t["data_offsets"]
            assert b - a == 4 * numel(shape)
            o.write(raw[base + a:base + b])
        cos, sin = rope_tables(c)
        cos.tofile(o); sin.tofile(o)
    print(f"{out_path}: {os.path.getsize(out_path):,} bytes (llama2.c v0)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "to-safetensors":
        to_safetensors(*a[1:])
    elif len(a) == 3 and a[0] == "to-bin":
        to_bin(a[1], a[2])
    else:
        sys.exit(__doc__)
