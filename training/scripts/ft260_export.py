#!/usr/bin/env python3
"""ft260_export.py: export a fine-tuned stories260K checkpoint (.pt) to llama2.c's legacy v0 .bin with llama2.c's own
export.py (legacy_export, the format of Karpathy's stories260K.bin), then check the round trip: the .bin loaded
back (check_260k.load_bin) must equal the .pt weights exactly and give the same logits.
Usage: $PY -I ft260_export.py ckpt.pt out.bin
"""
import os, struct, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import L2C  # noqa: E402
sys.path.insert(0, L2C)
import export as l2c_export  # noqa: E402  llama2.c export.py (MIT)
from check_260k import load_bin, full_logits  # noqa: E402

pt, out = sys.argv[1], sys.argv[2]
model = l2c_export.load_checkpoint(pt)        # llama2.c's loader (strips _orig_mod., strict=False)
ck = torch.load(pt, map_location="cpu", weights_only=False)
sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
msd = model.state_dict()
assert set(sd) == set(msd), (set(sd) ^ set(msd))
assert all(torch.equal(sd[k], msd[k]) for k in sd), "load_checkpoint did not load every tensor"
l2c_export.legacy_export(model, out)
mb, hdr, fc, fs = load_bin(out)
print("bin header:", hdr)
assert hdr == dict(dim=64, hidden=172, n_layers=5, n_heads=8, n_kv_heads=4, vocab=512, seq=512, shared=True), hdr
bsd = mb.state_dict()
maxw = max((bsd[k].float() - msd[k].float()).abs().max().item() for k in msd)
ids = [1, 291, 282, 420, 271, 413, 285, 410, 449, 314, 423, 266, 426]
d = (full_logits(mb, ids) - full_logits(model, ids)).abs().max().item()
print(f"round trip: max |weight diff| {maxw}, max |logit diff| {d}, bytes {os.path.getsize(out)}")
assert maxw == 0.0 and d == 0.0
print("RESULT: PASS")
