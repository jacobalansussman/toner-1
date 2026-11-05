"""Image helpers for TONER-1-V: 28x28 -> 16x16 one-bit, 2x2 patch tokens, contact sheets."""
import os
import numpy as np

from classes import CLASSES, slug, IMG, P, GRID, NPATCH, PATCH0, CLASS0, BOS, SEQLEN

ROW = 784


def load_raw(root, c, n=None):
    """Read a raw slice written by fetch_data.py (untrusted bytes -> uint8 array only)."""
    path = os.path.join(root, "data", "raw", slug(c) + ".u8")
    buf = open(path, "rb").read()
    assert len(buf) % ROW == 0
    a = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)
    return a if n is None else a[:n]


def box_matrix(n_in=28, n_out=IMG):
    """Exact area-weighted (box) resampling matrix, rows sum to 1."""
    s = n_in / n_out
    A = np.zeros((n_out, n_in))
    for i in range(n_out):
        lo, hi = i * s, (i + 1) * s
        for j in range(n_in):
            A[i, j] = max(0.0, min(hi, j + 1) - max(lo, j))
    return A / s


_A = box_matrix()


def downsample(x):
    """x: (N,28,28) uint8 -> (N,16,16) float in 0..255 (mean ink per output pixel)."""
    return np.einsum("ij,njk,lk->nil", _A, x.astype(np.float64), _A)


def dilate(b):
    """4-neighbour (plus-shaped) 1-pixel dilation of a boolean (N,H,W) array."""
    o = b.copy()
    o[:, 1:, :] |= b[:, :-1, :]; o[:, :-1, :] |= b[:, 1:, :]
    o[:, :, 1:] |= b[:, :, :-1]; o[:, :, :-1] |= b[:, :, 1:]
    return o


def to_bits(x, thr, dil=False):
    b = downsample(x) >= thr
    return dilate(b) if dil else b


def patchify(b):
    """(N,16,16) bool -> (N,64) patch values 0..15, raster order over the 8x8 patch grid.
    value = 8*TL + 4*TR + 2*BL + 1*BR."""
    N = b.shape[0]
    t = b.reshape(N, GRID, P, GRID, P).astype(np.int64)   # n, pr, dy, pc, dx
    v = 8 * t[:, :, 0, :, 0] + 4 * t[:, :, 0, :, 1] + 2 * t[:, :, 1, :, 0] + t[:, :, 1, :, 1]
    return v.reshape(N, NPATCH)


def unpatchify(v):
    v = np.asarray(v).reshape(-1, GRID, GRID)
    N = v.shape[0]
    b = np.zeros((N, GRID, P, GRID, P), dtype=bool)
    b[:, :, 0, :, 0] = (v >> 3) & 1; b[:, :, 0, :, 1] = (v >> 2) & 1
    b[:, :, 1, :, 0] = (v >> 1) & 1; b[:, :, 1, :, 1] = v & 1
    return b.reshape(N, IMG, IMG)


def sequences(bits, cls_idx):
    """bits (N,16,16) bool, cls_idx (N,) -> (N,66) int tokens: BOS, CLASS, 64 patches."""
    N = bits.shape[0]
    s = np.empty((N, SEQLEN), dtype=np.int64)
    s[:, 0] = BOS
    s[:, 1] = CLASS0 + cls_idx
    s[:, 2:] = PATCH0 + patchify(bits)
    return s


def sheet(rows, labels, path, scale=4, gap=4, title=None, col_labels=None):
    """rows: list of lists of 2-D arrays (bool ink=True or uint8 ink-high). Saves PNG."""
    from PIL import Image, ImageDraw
    lw = 90
    cell = max(max(a.shape[0] for a in r) for r in rows) * scale
    ncol = max(len(r) for r in rows)
    top = 18 if title else 4
    top += 12 if col_labels else 0
    W = lw + ncol * (cell + gap) + gap
    H = top + len(rows) * (cell + gap) + gap
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    if title:
        d.text((4, 3), title, fill=(0, 0, 0))
    if col_labels:
        for j, cl in enumerate(col_labels):
            d.text((lw + gap + j * (cell + gap), top - 12), cl, fill=(90, 90, 90))
    for i, (r, lab) in enumerate(zip(rows, labels)):
        y = top + gap + i * (cell + gap)
        d.text((4, y + cell // 2 - 6), lab, fill=(0, 0, 0))
        for j, a in enumerate(r):
            x = lw + gap + j * (cell + gap)
            if a.dtype == bool:
                g = np.where(a, 0, 255).astype(np.uint8)
            else:
                g = (255 - a.astype(np.float64).clip(0, 255)).astype(np.uint8)
            s = cell // a.shape[0]
            g = np.kron(g, np.ones((s, s), dtype=np.uint8))
            im.paste(Image.fromarray(g, "L").convert("RGB"), (x, y))
            d.rectangle([x - 1, y - 1, x + g.shape[1], y + g.shape[0]], outline=(200, 200, 200))
    im.save(path)
    return path
