"""Step 03 — the FOV Saturn is fitted on: lesion + distal-WM focus, cortex excluded.

Each hotspot from step 02 is reduced to its largest connected component (satellites dropped);
the FOV is their joint bounding box padded by 3 spot pitches, minus the cortex CellCharter
clusters (1, 5, 2). The result carries the lesion / distal index sets that become
`ann_region` in the Saturn targets.

Recovered verbatim (logic unchanged) from the interactive session of 2026-09-20 that produced
results/real_saturn/lpc10_fov.{npz,json}. Only the paths were made absolute and the diagnostic
PNG dropped.
"""
import json
import numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"
CORTEX = [1, 5, 2]


def main():
    z = np.load(f"{R}/lpc10_hotspots.npz", allow_pickle=True)
    xy, zmg, myz, lab, bc = z["xy"], z["zmg"], z["myz"], z["lab"], z["barcodes"]

    def main_component(idx):
        c = xy[idx]; k = min(7, len(c)); knn = cKDTree(c).query(c, k=k)[1]; kk = knn.shape[1] - 1
        adj = sp.csr_matrix((np.ones(len(c) * kk, np.int8), (np.repeat(np.arange(len(c)), kk), knn[:, 1:].ravel())),
                            shape=(len(c), len(c)))
        n, cl = connected_components(adj + adj.T, directed=False)
        return idx[cl == np.bincount(cl).argmax()]

    A = main_component(z["comp0"]); B = main_component(z["comp1"])        # lesion, distal WM focus
    pitch = np.median(np.diff(np.unique(xy[:, 0]))); pad = 3 * pitch
    x0, x1 = min(xy[A, 0].min(), xy[B, 0].min()) - pad, max(xy[A, 0].max(), xy[B, 0].max()) + pad
    y0, y1 = min(xy[A, 1].min(), xy[B, 1].min()) - pad, max(xy[A, 1].max(), xy[B, 1].max()) + pad
    inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    keep = inside & (~np.isin(lab, CORTEX))
    json.dump(dict(fov=[float(x0), float(x1), float(y0), float(y1)], n_in_fov=int(inside.sum()),
                   n_keep=int(keep.sum()), lesion_n=len(A), distal_n=len(B), cortex_excluded=CORTEX),
              open(f"{R}/lpc10_fov.json", "w"), indent=1)
    np.savez(f"{R}/lpc10_fov.npz", fov_mask=keep, lesion=A, distal=B, barcodes=bc, xy=xy, lab=lab, zmg=zmg, myz=myz)
    print(f"-> {R}/lpc10_fov.npz  kept {int(keep.sum())} spots (lesion {len(A)}, distal {len(B)})")


if __name__ == "__main__":
    main()
