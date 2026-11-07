"""Does H&E super-resolution make the tumour / stroma boundary clearer?

Contrast map  TS = mean z(melanocytic markers) - mean z(collagens), z-scored on spot values.
Compared on the SAME 8 um superpixel grid, over H&E tissue inside the DBiT window:
  iStar          super-resolved features (full run), linearly calibrated per gene on spots
  interpolated   spot values linearly interpolated onto the grid (no histology)

Sharpness metrics
  ambiguous area   fraction of tissue with |TS| < 0.5 (mixed / undecided zone)
  separability     Otsu between-class variance / total variance of TS (1 = two clean classes)
  transition width (TS p90 - TS p10) / median |grad TS| on the TS = 0 boundary, in um

Correctness (non-circular) is the held-out test in the same run log / manifest; sharpness alone
cannot show correctness because the H&E itself has sharp edges.

Usage: python scripts/melanoma_istar_boundary.py
"""
import json, pickle
import numpy as np, pandas as pd, cv2
from PIL import Image
from scipy.interpolate import griddata
from skimage.filters import threshold_otsu

P = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/melanoma_tissue_istar_feat/"
TUM = ["TYR", "PMEL", "MLANA", "DCT", "SOX10", "MITF"]
STR = ["COL1A1", "COL1A2", "COL3A1"]
SP_UM = 8.0                                     # superpixel = 16 px at 0.5 um/px
Image.MAX_IMAGE_PIXELS = None


def maps():
    L = pd.read_csv(P + "locs-all.tsv", sep="\t", index_col=0)
    C = pd.read_csv(P + "cnts-all.tsv", sep="\t", index_col=0).loc[L.index]
    he = np.asarray(Image.open(P + "he.jpg")); G = he.shape[0] // 16
    m = np.asarray(Image.open(P + "mask-window.png")) > 0
    tissue = cv2.resize(m.astype(np.uint8)[:G * 16, :G * 16], (G, G), interpolation=cv2.INTER_AREA) > 0.5
    i = (L.y.values / 16).astype(int); j = (L.x.values / 16).astype(int)
    gy, gx = np.mgrid[0:G, 0:G]; grid = np.c_[gx.ravel() * 16 + 8, gy.ravel() * 16 + 8]
    Zi, Zl, Zs = {}, {}, {}
    for g in TUM + STR:
        n = "RNA_" + g; v = C[n].values; mu, sd = v.mean(), v.std()
        y = pickle.load(open(P + f"cnts-super/{n}.pickle", "rb"))
        p = np.array([y[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2].mean() for a, b in zip(i, j)])
        a_, b_ = np.linalg.lstsq(np.c_[p, np.ones_like(p)], v, rcond=None)[0]
        Zi[g] = ((a_ * y + b_) - mu) / sd
        li = griddata(L[["x", "y"]].values, v, grid, method="linear").reshape(G, G)
        Zl[g] = (li - mu) / sd
        Zs[g] = (v - mu) / sd
    ts = lambda Z: np.mean([Z[g] for g in TUM], 0) - np.mean([Z[g] for g in STR], 0)
    return ts(Zi), ts(Zl), tissue, he, L, ts(Zs)


def sharpness(T, tissue):
    ok = tissue & np.isfinite(T)
    v = T[ok]
    amb = float((np.abs(v) < 0.5).mean())
    th = threshold_otsu(v); a, b = v[v <= th], v[v > th]
    eta = float((len(a) * len(b) / len(v) ** 2) * (a.mean() - b.mean()) ** 2 / v.var())
    Tf = np.where(ok, T, 0.0)
    gy, gx = np.gradient(Tf); gm = np.hypot(gx, gy) / SP_UM             # per um
    sign = np.sign(Tf)
    edge = ok & ((np.roll(sign, 1, 0) != sign) | (np.roll(sign, 1, 1) != sign))
    edge &= np.roll(ok, 1, 0) & np.roll(ok, 1, 1)
    span = np.percentile(v, 90) - np.percentile(v, 10)
    width = float(span / np.median(gm[edge])) if edge.any() else np.nan
    return dict(ambiguous_frac=amb, separability=eta, transition_width_um=width, n_edge=int(edge.sum()))


def main():
    Ti, Tl, tissue, he, L, ts_spot = maps()
    out = dict(iStar=sharpness(Ti, tissue), interpolated=sharpness(Tl, tissue))
    for k, v in out.items():
        print(f"{k:13s} ambiguous area {v['ambiguous_frac']:.3f} | separability {v['separability']:.3f} | "
              f"transition width {v['transition_width_um']:.1f} um  (boundary superpixels {v['n_edge']})")
    np.savez_compressed(P + "tumour_stroma_maps.npz", iStar=Ti, interpolated=Tl, tissue=tissue)
    json.dump(out, open(P + "tumour_stroma_sharpness.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
