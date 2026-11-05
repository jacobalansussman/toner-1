"""Judge for recognizability: a small CNN trained on the REAL 16x16 one-bit training images.
  train:  python -I src/classifier.py <pixelart_root> train
  score:  python -I src/classifier.py <pixelart_root> score out/samples_<...>.npy [...]
Accuracy on real validation images is the ceiling for what generated doodles can reach."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from classes import CLASSES, CLASS0, PATCH0
from pix import unpatchify

torch.set_num_threads(4)
root, mode = sys.argv[1], sys.argv[2]
path = os.path.join(root, "ckpt", "judge_cnn.pt")


def net():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, len(CLASSES)))


def load(split):
    s = np.load(os.path.join(root, "data", "proc", split + ".npy"), allow_pickle=False)
    x = torch.from_numpy(unpatchify(s[:, 2:].astype(np.int64) - PATCH0).astype(np.float32))[:, None]
    return x, torch.from_numpy(s[:, 1].astype(np.int64) - CLASS0)


def acc_per_class(m, x, y):
    with torch.no_grad():
        p = torch.cat([m(x[i:i + 2000]).argmax(1) for i in range(0, len(x), 2000)])
    return {c: float((p[y == i] == i).float().mean()) for i, c in enumerate(CLASSES)}, float((p == y).float().mean())


if mode == "train":
    torch.manual_seed(0)
    xt, yt = load("train"); xv, yv = load("val")
    m = net(); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.01)
    for ep in range(3):
        m.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(m(xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); pc, a = acc_per_class(m, xv, yv)
        print(f"epoch {ep}: real val accuracy {a:.3f}", flush=True)
    torch.save(m.state_dict(), path)
    json.dump(dict(real_val_accuracy=a, per_class=pc), open(path.replace(".pt", ".json"), "w"), indent=1)
    print({k: round(v, 2) for k, v in pc.items()})
else:
    m = net(); m.load_state_dict(torch.load(path, weights_only=True)); m.eval()
    ceil = json.load(open(path.replace(".pt", ".json")))
    for f in sys.argv[3:]:
        imgs = np.load(f, allow_pickle=False)                 # (classes, n, 16, 16) bool
        C, n = imgs.shape[:2]
        x = torch.from_numpy(imgs.reshape(-1, 1, 16, 16).astype(np.float32))
        y = torch.arange(C).repeat_interleave(n)
        pc, a = acc_per_class(m, x, y)
        print(f"{os.path.basename(f)}: judge accuracy {a:.3f} (real val ceiling {ceil['real_val_accuracy']:.3f}) "
              f"ink {imgs.mean():.3f}")
        print("  " + " ".join(f"{c}={pc[c]:.2f}" for c in CLASSES))
        json.dump(dict(accuracy=a, per_class=pc, ink=float(imgs.mean()), n_per_class=n),
                  open(f.replace(".npy", "_judge.json"), "w"), indent=1)
