"""Prepare Catalign inputs for LPC10 S1: DBiT chip image (moving) and CODEX tissue composite (fixed).

DBiT : GEO GSM9247593 tissue_hires_image (scale factor 1 -> same pixel frame as tissue_positions_list).
CODEX: Codex_LPC10_S1.qptiff pyramid level 1 (0.507 um/px at level 0 -> 1.013 um/px).
       Fixed image = mean of percentile-normalised DAPI/MBP/NeuN/GFAP/IBA1, so all tissue is bright
       and ventricles / background stay dark (Catalign 'fluorescence' lumen rule).
"""
import json
import numpy as np, tifffile
from PIL import Image

import os; H = os.environ["SATURN_ROOT"]
DB = f"{H}/external_data/dbit_lpc10_spatial/GSM9247593_07_LPC10S1_"
CX = f"{H}/external_data/codex_lpc"
OUT = f"{H}/catalign_lpc10/inputs"; os.makedirs(OUT, exist_ok=True)
LEVEL = 1
CHANNELS = ["DAPI", "MBP", "NeuN", "GFAP", "IBA1"]


def main():
    chip = np.asarray(Image.open(DB + "tissue_hires_image.png").convert("L"))
    Image.fromarray(chip).save(f"{OUT}/dbit_chip_gray.png")
    print("chip", chip.shape)

    names = json.load(open(f"{CX}/channels_LPC10_S1.json"))
    tf = tifffile.TiffFile(f"{CX}/Codex_LPC10_S1.qptiff")
    p0 = tf.series[0].pages[0]
    xres = p0.tags["XResolution"].value
    um0 = 1e4 / (xres[0] / xres[1])                       # px per cm -> um per px
    lvl = tf.series[0].levels[LEVEL]
    um = um0 * tf.series[0].levels[0].shape[-1] / lvl.shape[-1]
    acc = np.zeros(lvl.shape[1:], np.float32)
    for c in CHANNELS:
        x = lvl.pages[names.index(c)].asarray().astype(np.float32)
        lo, hi = np.percentile(x[::4, ::4], [1, 99.7])
        acc += np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
        print(" ", c, "done", flush=True)
    comp = (255 * acc / len(CHANNELS)).astype(np.uint8)
    tifffile.imwrite(f"{OUT}/codex_lvl{LEVEL}_tissue.tif", comp, compression="zlib")
    json.dump(dict(level=LEVEL, um_per_px=um, um_per_px_level0=um0, shape=list(comp.shape), channels=CHANNELS),
              open(f"{OUT}/codex_meta.json", "w"), indent=1)
    print("codex", comp.shape, f"{um:.4f} um/px")


if __name__ == "__main__":
    main()
