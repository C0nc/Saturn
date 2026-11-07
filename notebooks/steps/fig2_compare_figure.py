"""Pixel comparison of a page with the published PNG. Usage: fig2_compare_figure.py <name> ... (under published_figures/fig2)."""
import os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT, PUB = os.environ["SATURN_ROOT"], os.environ.get("SATURN_PUBLISHED")
if not PUB:
    print("SATURN_PUBLISHED not set: no comparison"); raise SystemExit
for n in sys.argv[1:]:
    a = np.asarray(Image.open(f"{ROOT}/Saturn/published_figures/fig2/{n}.png").convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(f"{PUB}/Saturn/published_figures/fig2/{n}.png").convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        print(f"{n}: size {a.shape} vs {b.shape}"); continue
    d = np.abs(a - b).max(2); print(f"{n}: size {a.shape}, pixels differing {(d > 0).sum()} of {d.size}, max channel diff {d.max()}")
