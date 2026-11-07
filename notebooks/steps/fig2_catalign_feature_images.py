"""Turn both inputs into 'tissue bright / empty dark' images so Catalign's fluorescence lumen rule
(intensity < 0.6 x Otsu) separates tissue from empty space instead of grey from white matter.

chip : pale brightfield, tissue ~230 vs empty ~251 -> f = clip((253 - blur) / 38) (inverted, texture kept)
codex: composite at ~4 um/px; ratio to local tissue mean inside the silhouette (ventricles dark), written back
       at level-1 size (1.013 um/px) so Catalign's default search downsampling applies
"""
import numpy as np, tifffile, cv2
from scipy import ndimage as ndi
from PIL import Image

import os; D = os.environ["SATURN_ROOT"] + "/catalign_lpc10/inputs"


def main():
    c = np.asarray(Image.open(f"{D}/dbit_chip_gray.png")).astype(np.float32)
    cb = cv2.GaussianBlur(c, (0, 0), 1.5)
    fc = np.clip((253 - cb) / 38, 0, 1)
    Image.fromarray((255 * fc).astype(np.uint8)).save(f"{D}/dbit_chip_feat.png")

    # CODEX at ~4 um/px (level 1 / 4): ventricles carry dim autofluorescence, so a global threshold misses
    # them. Divide by the local tissue mean (sigma ~240 um) inside the main tissue silhouette instead.
    h = tifffile.imread(f"{D}/codex_lvl1_tissue.tif").astype(np.float32)
    h = cv2.resize(h, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hb = cv2.GaussianBlur(h, (0, 0), 1.5)
    tis = ndi.binary_fill_holes(ndi.binary_closing(hb > 3, np.ones((15, 15))))
    lab, n = ndi.label(tis)
    tis = lab == (np.argmax(ndi.sum(tis, lab, range(1, n + 1))) + 1)
    sig = 60
    loc = cv2.GaussianBlur(hb * tis, (0, 0), sig) / np.maximum(cv2.GaussianBlur(tis.astype(np.float32), (0, 0), sig), 1e-3)
    r = cv2.GaussianBlur(np.where(tis, hb / np.maximum(loc, 1e-3), 0).astype(np.float32), (0, 0), 2)
    fh = np.where(tis, np.clip((r - 0.3) / 0.7, 0, 1), 0)
    full = tifffile.imread(f"{D}/codex_lvl1_tissue.tif", key=0).shape
    fh = cv2.resize(fh.astype(np.float32), (full[1], full[0]), interpolation=cv2.INTER_LINEAR)
    tifffile.imwrite(f"{D}/codex_lvl1_feat.tif", (255 * fh).astype(np.uint8), compression="zlib")
    print("chip feat mean", fc.mean(), "codex feat mean", fh.mean())


if __name__ == "__main__":
    main()
