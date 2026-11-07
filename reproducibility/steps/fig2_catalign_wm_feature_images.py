"""White-matter structure images for Catalign (run 2).

The lumen-only inputs (02) are degenerate here: the DBiT chip is almost all tissue, so a small template fits
anywhere inside the brain. White matter is the strongest structure both modalities share, so both images are
rebuilt as 'WM bright, grey matter and empty space dark'. Catalign's fluorescence lumen mask then equals
'not white matter', and its intensity channel sees the graded myelin signal.

chip : DBiT Mbp/Plp1/Mobp/Mog log-CPM per spot, painted into the chip pixel frame (GEO positions, x = pxl_col)
codex: MBP channel, CODEX level 1 (1.0133 um/px)
Each: Otsu threshold between grey and white matter inside tissue, soft ramp of +-w around it.
Microglia and neuron markers are NOT used, so 04_validate_markers.py stays independent for them.
"""
import json
import numpy as np, pandas as pd, tifffile, cv2, anndata as ad
from scipy import sparse
from skimage.filters import threshold_otsu

import os; H = os.environ["SATURN_ROOT"]
D = f"{H}/catalign_lpc10/inputs"
POS = f"{H}/external_data/dbit_lpc10_spatial/GSM9247593_07_LPC10S1_tissue_positions_list.csv"
RNA = f"{H}/claude_code_tutorial_lpc/results/timepoints/arcseq_LPC10S1_rna_adata.h5ad"
CX = f"{H}/external_data/codex_lpc"
GENES = ["Mbp", "Plp1", "Mobp", "Mog"]


def ramp(v, tissue):
    t = threshold_otsu(v[tissue])
    w = 0.5 * np.std(v[tissue])
    return np.where(tissue, np.clip((v - (t - w)) / (2 * w), 0, 1), 0), t


def main():
    # --- chip: paint per-spot myelin score as square spots of one pitch
    p = pd.read_csv(POS, header=None, names=["bc", "t", "row", "col", "py", "px"]).set_index("bc")
    a = ad.read_h5ad(RNA)
    X = sparse.csr_matrix(a.X).astype(np.float32)
    X = sparse.diags(1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
    X.data = np.log1p(X.data)
    vn = list(a.var_names)
    s = np.asarray(X[:, [vn.index(g) for g in GENES]].mean(1)).ravel()
    q = p.loc[a.obs_names]
    G = np.zeros((96, 96), np.float32); T = np.zeros((96, 96), bool)
    G[q.row, q.col] = s; T[q.row, q.col] = True
    G = cv2.GaussianBlur(G * T, (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(T.astype(np.float32), (0, 0), 1.0), 1e-3)
    f, thr = ramp(G, T)
    # grid cell (row, col) -> chip px (y, x): fit pxl = a*idx + b on all positions
    ax_, bx = np.polyfit(p.col, p.px, 1); ay, by = np.polyfit(p.row, p.py, 1)
    from PIL import Image
    hw = Image.open(f"{D}/dbit_chip_gray.png").size[::-1]
    M = np.float32([[1 / ax_, 0, -bx / ax_], [0, 1 / ay, -by / ay]])           # chip px -> grid index
    chip = cv2.warpAffine(f.astype(np.float32), M, (hw[1], hw[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Image.fromarray((255 * chip).astype(np.uint8)).save(f"{D}/dbit_chip_wm.png")
    print(f"chip: pitch {ax_:.2f}/{ay:.2f} px, Otsu {thr:.2f}, WM frac {np.mean(f[T] > .5):.2f}")

    # --- CODEX MBP at level 1
    names = json.load(open(f"{CX}/channels_LPC10_S1.json"))
    lvl = tifffile.TiffFile(f"{CX}/Codex_LPC10_S1.qptiff").series[0].levels[1]
    m = lvl.pages[names.index("MBP")].asarray().astype(np.float32)
    lo, hi = np.percentile(m[m > 0], [1, 99.5])
    m = np.clip((m - lo) / (hi - lo), 0, 1)
    mb = cv2.GaussianBlur(m, (0, 0), 8)                                          # ~8 um, below the spot size
    tis = cv2.GaussianBlur(tifffile.imread(f"{D}/codex_lvl1_feat.tif").astype(np.float32), (0, 0), 4) > 40
    fc, tc = ramp(mb, tis)
    tifffile.imwrite(f"{D}/codex_lvl1_wm.tif", (255 * fc).astype(np.uint8), compression="zlib")
    print(f"codex: Otsu {tc:.3f}, WM frac {np.mean(fc[tis] > .5):.2f}")


if __name__ == "__main__":
    main()
