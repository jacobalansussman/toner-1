"""TONER-1-V class list and vocabulary (single source of truth)."""
# Order defines class token ids: CLASS_TOKEN[c] = 18 + index.
CLASSES = ["cat", "dog", "bird", "fish", "duck", "rabbit", "frog", "teddy-bear", "sun",
           "tree", "flower", "house", "car", "apple", "cake", "soccer ball", "light bulb",
           "calculator"]

BOS, EOS = 0, 1
PATCH0 = 2                     # tokens 2..17 = 2x2 patch values 0..15
CLASS0 = 18                    # tokens 18..35 = classes in CLASSES order
VOCAB = CLASS0 + len(CLASSES)  # 36
IMG = 16                       # 16x16 one-bit image
P = 2                          # 2x2 patches
GRID = IMG // P                # 8x8 patches
NPATCH = GRID * GRID           # 64 patch tokens
SEQLEN = 2 + NPATCH            # BOS, CLASS, 64 patches = 66
# Patch value bit order: value = 8*TL + 4*TR + 2*BL + 1*BR (1 = ink).
PATCH_BITS = ["TL", "TR", "BL", "BR"]


def slug(c):
    return c.replace(" ", "_")
