"""iStar inputs for super-resolving the melanoma Saturn isodepths from the ADJACENT H&E.

Geometry (all derived, nothing hand-tuned):
  DBiT spots -> brightfield   tissue_positions_list.csv, x = pxl_col, y = pxl_row
                              (verified: 93% of fitted spots on BF tissue, 0% of out-of-tissue
                              spots; the transposed convention puts 82% of out-of-tissue spots on
                              tissue)
  brightfield px size          DBiT pitch 40 um (20 um channels, user-confirmed) / BF pitch in px
  brightfield -> H&E           registration A, refined with free scale (reg_refined.json)
  H&E px size                  BF px size / registration scale
  iStar frame                  H&E crop around the DBiT window, resampled here to 0.5 um/px, so
                               iStar's rescale step is the identity for the image. Do NOT run
                               `rescale.py --locs`: the local copy swaps x/y (see below); locs.tsv
                               is written here instead.
  square spot                  20 um side -> half-width 10 um = 20 px at 0.5 um/px

Values imputed: rho_RNA and rho_ATAC of the `melanoma_tissue` fit (iStar min-max scales each
column, so signed values are fine). Written to `cnts_ab.tsv` because the local iStar copy
(impute_by_basic.get_gene_counts) reads that file name.

Usage: python scripts/melanoma_istar_prep.py <reg_refined.json> [similarity|affine] [--holdout 0.1]
"""
import json, os, sys
import numpy as np, pandas as pd, cv2, tifffile, zarr
from PIL import Image

H = "/mnt/HDD2B/Users/yancui"
M = f"{H}/Yufan_Melanoma"
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
PITCH_UM, CHANNEL_UM, TARGET_UM = 40.0, 20.0, 0.5
Image.MAX_IMAGE_PIXELS = None


def main(reg_json, model="similarity", holdout=0.0, seed=0):
    tag = "" if holdout == 0 else f"_holdout{int(round(holdout * 100))}"
    OUT = f"{R}/melanoma_tissue_istar{tag}/"
    os.makedirs(OUT, exist_ok=True)
    reg = json.load(open(reg_json))
    key = "similarity-free-scale" if model == "similarity" else "affine"
    F = np.array(reg[key]["bf_to_he_full"])                      # BF full px -> HE full px (3x3)

    pos = pd.read_csv(f"{M}/spatial/tissue_positions_list.csv", header=None,
                      names=["bc", "in", "r", "c", "py", "px"]).set_index("bc")
    # BF pixel size from the lattice itself
    pitch_bf = np.median(np.diff(np.sort(pos.drop_duplicates("c").sort_values("c").px.values)))
    um_bf = PITCH_UM / pitch_bf
    scale = np.sqrt(abs(np.linalg.det(F[:2, :2])))
    um_he = um_bf / scale
    print(f"BF pitch {pitch_bf:.2f} px -> {um_bf:.4f} um/px | registration scale {scale:.4f} "
          f"-> H&E {um_he:.4f} um/px", flush=True)

    # DBiT window corners in H&E full px, plus one pitch of margin
    n = 6382
    corners = np.array([[0, 0, 1], [n, 0, 1], [n, n, 1], [0, n, 1]], float) @ F.T
    pad = PITCH_UM / um_he
    x0, y0 = np.floor(corners[:, :2].min(0) - pad).astype(int)
    x1, y1 = np.ceil(corners[:, :2].max(0) + pad).astype(int)
    z = zarr.open(tifffile.TiffFile(f"{M}/HE_HMelanoma.tif").aszarr(), mode="r")
    Hh, Wh = z.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, Wh), min(y1, Hh)
    crop = np.asarray(z[y0:y1, x0:x1])
    f = um_he / TARGET_UM
    img = cv2.resize(crop, (int(round(crop.shape[1] * f)), int(round(crop.shape[0] * f))),
                     interpolation=cv2.INTER_AREA)
    Image.fromarray(img).save(OUT + "he-raw.jpg", quality=95)
    print(f"H&E crop x {x0}-{x1}, y {y0}-{y1} ({crop.shape[1]}x{crop.shape[0]} px) -> "
          f"{img.shape[1]}x{img.shape[0]} at {TARGET_UM} um/px", flush=True)

    # DBiT window + H&E tissue mask in the iStar frame (predictions are only read inside it)
    win = (corners[:, :2] - [x0, y0]) * f
    wmask = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(wmask, [win.astype(np.int32)], 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    tissue = ((hsv[..., 1] > 40) | (img.mean(2) < 190)).astype(np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    Image.fromarray(((tissue & wmask) * 255).astype(np.uint8)).save(OUT + "mask-window.png")

    # spots: fitted melanoma_tissue spots, BF -> HE full -> iStar frame
    ax = pd.read_csv(f"{R}/melanoma_tissue/axes.csv", index_col=0)
    p = pos.loc[ax.index]
    xy_bf = np.c_[p.px.values, p.py.values, np.ones(len(p))]
    xy_he = xy_bf @ F.T
    loc = (xy_he[:, :2] - [x0, y0]) * f
    locs = pd.DataFrame({"x": loc[:, 0], "y": loc[:, 1]}, index=ax.index); locs.index.name = "Spot ID"
    cn = pd.DataFrame({"rho_RNA": ax.d_RNA.values, "rho_ATAC": ax.d_ATAC.values}, index=ax.index)
    cn.index.name = "Spot ID"
    held = np.zeros(len(ax), bool)
    if holdout > 0:
        rng = np.random.default_rng(seed)
        held[rng.choice(len(ax), int(round(holdout * len(ax))), replace=False)] = True
        pd.Series(ax.index[held]).to_csv(OUT + "heldout_spots.txt", index=False, header=False)
    locs[~held].to_csv(OUT + "locs-raw.tsv", sep="\t")
    # write iStar's locs.tsv directly: istar/rescale.py --locs SWAPS x and y (a Hi-C-specific edit in the
    # local checkout), which would transpose these already-correct coordinates. Scale is 1 here.
    locs[~held].round().astype(int).to_csv(OUT + "locs.tsv", sep="\t")
    cn[~held].to_csv(OUT + "cnts_ab.tsv", sep="\t")
    locs.to_csv(OUT + "locs-all.tsv", sep="\t"); cn.to_csv(OUT + "cnts-all.tsv", sep="\t")
    open(OUT + "gene-names.txt", "w").write("rho_RNA\nrho_ATAC\n")
    open(OUT + "pixel-size-raw.txt", "w").write(str(TARGET_UM))
    open(OUT + "pixel-size.txt", "w").write(str(TARGET_UM))
    open(OUT + "radius-raw.txt", "w").write(str(CHANNEL_UM / 2 / TARGET_UM))
    json.dump(dict(model=key, um_bf=um_bf, um_he=um_he, scale=scale, crop=[int(x0), int(y0), int(x1), int(y1)],
                   resample=f, window_iStar=win.tolist(), n_train=int((~held).sum()), n_heldout=int(held.sum()),
                   registration=reg[key]), open(OUT + "geometry.json", "w"), indent=1)
    print(f"-> {OUT}  ({(~held).sum()} training spots, {held.sum()} held out)", flush=True)
    return OUT


if __name__ == "__main__":
    ho = float(sys.argv[sys.argv.index("--holdout") + 1]) if "--holdout" in sys.argv else 0.0
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "similarity", ho)
