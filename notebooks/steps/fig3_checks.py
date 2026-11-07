"""Comparisons with the published Fig 3 results that nbutil.compare_axes / compare_table do not cover."""
import json
import numpy as np, pandas as pd, scipy.sparse as sp
import nbutil as nb

LAB = "Hole/gaston_mix_mm/results"


def _pub(rel): return nb.PUB / LAB / rel


def upstream_objects():
    """Recovered RNA and ATAC objects vs the published objects: same spots, features and counts."""
    import anndata as ad, hdf5plugin  # noqa: F401
    rows = []
    for f in ("melanoma/melanoma_rna.h5ad", "melanoma/atac_snap.h5ad"):
        a, b = ad.read_h5ad(nb.ROOT / LAB / f), ad.read_h5ad(_pub(f))
        same_obs = list(a.obs_names) == list(b.obs_names); same_var = list(a.var_names) == list(b.var_names)
        Xa, Xb = sp.csr_matrix(a.X), sp.csr_matrix(b.X)
        rows.append(dict(file=f, shape=str(a.shape), pub_shape=str(b.shape), same_obs=same_obs, same_var=same_var,
                         n_count_diffs=int((Xa != Xb).nnz) if Xa.shape == Xb.shape else -1, dtype=f"{Xa.dtype}/{Xb.dtype}"))
        if "spatial" in b.obsm and "spatial" in a.obsm:
            rows[-1]["spatial_max_diff"] = float(np.abs(a.obsm["spatial"] - b.obsm["spatial"]).max())
        if "mel_score" in b.obs:
            rows[-1]["mel_score_max_diff"] = float(np.abs(a.obs["mel_score"].values - b.obs["mel_score"].values).max())
    T = pd.DataFrame(rows); print(T.T.to_string()); return T


def csv_abs(rel, cat=None):
    """CSV under the lab root, this run vs published: largest difference per numeric column"""
    a = pd.read_csv(nb.ROOT / rel, index_col=0); b = pd.read_csv(nb.PUB / rel, index_col=0)
    print(rel, a.shape, b.shape, "same spots:", list(a.index) == list(b.index))
    print(pd.Series({c: float(np.nanmax(np.abs(a[c].astype(float) - b[c].astype(float)))) for c in b.select_dtypes("number")}).to_string())


def npz(rel, keys=None):
    """largest difference per array of an .npz under results/real_saturn"""
    a = np.load(nb.RES / rel, allow_pickle=True); b = np.load(_pub("real_saturn/" + rel), allow_pickle=True); rows = []
    for k in (keys or b.files):
        x, y = a[k], b[k]
        if x.shape != y.shape: rows.append(dict(key=k, diff=f"shape {x.shape} vs {y.shape}")); continue
        if x.dtype.kind in "fiu":
            rows.append(dict(key=k, diff=float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0))
        else:
            rows.append(dict(key=k, diff=int((x.astype(str) != y.astype(str)).sum())))
    T = pd.DataFrame(rows); print(rel); print(T.to_string(index=False)); return T


def target_recheck(ds):
    """Rerun of the target script vs the published target; for Y_RNA also the correlation of each component with its best published match."""
    a = np.load(nb.RES / f"target_recheck/targets_{ds}.npz", allow_pickle=True); b = np.load(_pub(f"real_saturn/targets_{ds}.npz"), allow_pickle=True)
    rows = []
    for k in b.files:
        x, y = a[k], b[k]
        same = bool(np.array_equal(x, y, equal_nan=True)) if x.dtype.kind in "fiu" else bool((x.astype(str) == y.astype(str)).all())
        rows.append(dict(key=k, identical=same, max_diff=float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.dtype.kind in "fiu" else np.nan))
    print(pd.DataFrame(rows).to_string(index=False))
    x, y = a["Y_RNA"], b["Y_RNA"]; C = np.abs(np.corrcoef(x.T, y.T)[:x.shape[1], x.shape[1]:])
    print("Y_RNA, best |r| of each new component with the published ones:", C.max(1).round(2))
    print("Y_RNA sd per component, new:", x.std(0).round(2), "published:", y.std(0).round(2))


def json_numbers(rel, keys):
    a = json.load(open(nb.RES / rel)); b = json.load(open(_pub("real_saturn/" + rel)))
    for k in keys: print(f"{rel} {k}: this run {a[k]} | published {b[k]} | {'SAME' if a[k] == b[k] else 'DIFFERS'}")


def istar_stats():
    """Fig 3j numbers from the builder's stats file, this run vs published (iStar is not deterministic)."""
    a = json.load(open(nb.PF / "fig3/fig3_saturn_stats.json"))["j"]; b = json.load(open(nb.PUB / "Saturn/published_figures/fig3/fig3_parts_stats.json"))["j"]
    rows = [dict(metric="width_um iStar", this=a["width_um"]["iStar"], published=b["width_um"]["iStar"]),
            dict(metric="width_um interpolated", this=a["width_um"]["interpolated"], published=b["width_um"]["interpolated"]),
            dict(metric="width p", this=a["width_p"], published=b["width_p"])]
    for s in ("all", "boundary"):
        for k in ("n", "iStar", "knn6"): rows.append(dict(metric=f"held-out {s} {k}", this=a["heldout"][s][k], published=b["heldout"][s][k]))
    T = pd.DataFrame(rows); print(T.to_string(index=False)); return T


def png_identical(*names):
    """pixel comparison of workspace vs published PNG (paths relative to published_figures)"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    for n in names:
        a = np.asarray(Image.open(nb.PF / n)); b = np.asarray(Image.open(nb.PUB / "Saturn/published_figures" / n))
        if a.shape != b.shape: print(n, f"shape {a.shape} vs {b.shape}"); continue
        d = (a != b).any(-1) if a.ndim == 3 else a != b
        print(f"{n}: {a.shape}, differing pixels {int(d.sum())} ({100 * d.mean():.3f}%)")
