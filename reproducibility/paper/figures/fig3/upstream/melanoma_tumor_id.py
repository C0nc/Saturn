"""
Melanoma tumor vs non-tumor identification via 4 orthogonal signals:
  1. CellCharter spatial domains (scVI + Delaunay kNN + GMM K sweep)
  2. Melanoma marker score (MLANA/TYR/PMEL/MITF/DCT/S100B/SOX10/TYRP1)
  3. TACCO-equivalent: Starfysh deconvolution `mel_frac` (sum of 4 melanoma
     states: Melanocytic/Transitory/Invasive/NeuralCrest)
  4. CNV burden per spot from infercnv X_cnv matrix

Then combine per-CellCharter-cluster averages to call each cluster tumor / non-tumor,
and save a per-spot tumor_mask.
"""
import os, sys, pathlib
import numpy as np, pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RNA_H5 = "/mnt/HDD2B/Users/yancui/Saturn/paper_figures/figure2/melanoma_in_tissue/melanoma_with_cnv_in_tissue.h5ad"
SF_H5  = "/mnt/HDD2B/Users/yancui/Saturn/paper_figures/figure2/melanoma_starfysh/starfysh_decon.h5ad"
OUT    = pathlib.Path("/mnt/HDD2B/Users/yancui/Saturn/paper_figures/figure2/melanoma_tumor_id")
OUT.mkdir(exist_ok=True, parents=True)

# ================================================================
# 1. load + merge
# ================================================================
print("[1] load RNA h5ad + Starfysh decon")
a  = ad.read_h5ad(RNA_H5)          # 1999 × 62700; has X_cnv + mel_score + cnv_burden
sf = ad.read_h5ad(SF_H5)           # 2000 × 2544; has sf_* deconvolution fractions
print(f"  RNA: {a.shape}   Starfysh: {sf.shape}")

# align on spatial coords (both have Visium spot barcodes as obs_names ideally)
if not a.obs_names.isin(sf.obs_names).all():
    # fallback: align by rounded spatial coords
    print("  obs_name mismatch — aligning by spatial coords")
    key_a  = pd.MultiIndex.from_arrays(a.obsm["spatial"].round().astype(int).T,  names=["x","y"])
    key_sf = pd.MultiIndex.from_arrays(sf.obsm["spatial"].round().astype(int).T, names=["x","y"])
    sf_reindex = pd.DataFrame(index=key_sf).reset_index()
    sf_reindex["ix"] = np.arange(len(sf_reindex))
    match = pd.DataFrame(index=key_a).reset_index().merge(sf_reindex, on=["x","y"], how="left")
    ok = match["ix"].notna()
    print(f"  aligned {ok.sum():,} / {len(a):,} spots by coords")
    sf_cols = [c for c in sf.obs.columns if c.startswith("sf_") or c.endswith("_frac")]
    for c in sf_cols:
        vals = np.full(len(a), np.nan, dtype=float)
        vals[ok] = sf.obs[c].values[match.loc[ok, "ix"].astype(int)]
        a.obs[c] = vals
else:
    common = a.obs_names.intersection(sf.obs_names)
    a = a[common].copy()
    sf_ord = sf[common]
    for c in sf.obs.columns:
        if c.startswith("sf_") or c.endswith("_frac"):
            a.obs[c] = sf_ord.obs[c].values
    print(f"  aligned by obs_names: {len(a):,} spots")

# ================================================================
# 2. preprocess RNA + scVI + CellCharter K sweep
# ================================================================
print("[2] scVI on RNA + CellCharter K sweep")
sc.pp.filter_genes(a, min_cells=10)
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat")
adata_hv = a[:, a.var.highly_variable].copy()
adata_hv.X = adata_hv.layers["counts"]

import scvi
scvi.settings.seed = 0
scvi.model.SCVI.setup_anndata(adata_hv)
model = scvi.model.SCVI(adata_hv, n_latent=15, n_layers=2, gene_likelihood="nb")
model.train(max_epochs=200, early_stopping=True,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            check_val_every_n_epoch=5, plan_kwargs=dict(lr=1e-3))
a.obsm["X_scVI"] = model.get_latent_representation()

import cellcharter as cc
import squidpy as sq
sq.gr.spatial_neighbors(a, coord_type="generic", delaunay=True)
cc.gr.remove_long_links(a)
cc.gr.aggregate_neighbors(a, n_layers=3, use_rep="X_scVI", out_key="X_cc")

# CellCharter's torchgmm barks on small N (1999 spots) → sklearn GMM directly on X_cc
from sklearn.mixture import GaussianMixture
Ks = [5, 7, 9]
CLS = {}
Xcc = np.asarray(a.obsm["X_cc"])
for K in Ks:
    print(f"  fit K={K} (sklearn GMM on X_cc)")
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=0, max_iter=200)
    labs = gmm.fit_predict(Xcc).astype(int)
    a.obs[f"cc_K{K}"] = pd.Categorical(labs)
    CLS[K] = labs

# ================================================================
# 3. Melanoma marker score (sc.tl.score_genes)
# ================================================================
print("[3] melanoma marker score")
MEL = ["MLANA","TYR","PMEL","MITF","DCT","S100B","SOX10","TYRP1","MC1R","GPNMB","APOD","MLPH"]
IMM = ["CD3D","CD3E","CD8A","CD4","CD68","CD163","CD79A","MS4A1","PTPRC"]
STROM = ["ACTA2","FAP","COL1A1","VIM","PECAM1"]
def resolve(gs, U): return [g for g in gs if g in U]
mel_hit = resolve(MEL, a.var_names); imm_hit = resolve(IMM, a.var_names); str_hit = resolve(STROM, a.var_names)
print(f"  melanoma markers found: {mel_hit}")
print(f"  immune markers found:   {imm_hit}")
print(f"  stromal markers found:  {str_hit}")
sc.tl.score_genes(a, mel_hit, score_name="mel_marker",   ctrl_size=min(50, a.n_vars-len(mel_hit)))
sc.tl.score_genes(a, imm_hit, score_name="immune_marker",ctrl_size=min(50, a.n_vars-len(imm_hit)))
sc.tl.score_genes(a, str_hit, score_name="stromal_marker",ctrl_size=min(50, a.n_vars-len(str_hit)))

# ================================================================
# 4. CNV burden per spot (use existing X_cnv from infercnv)
# ================================================================
print("[4] CNV burden per spot")
Xcnv = a.obsm["X_cnv"]
if hasattr(Xcnv, "toarray"): Xcnv = Xcnv.toarray()
a.obs["cnv_burden_v2"] = np.abs(Xcnv).sum(1)          # sum absolute deviation
a.obs["cnv_std"]        = Xcnv.std(axis=1)             # variability across genome bins

# ================================================================
# 5. Per-CellCharter-cluster tumor call
# ================================================================
print("[5] per-cluster tumor call")
score_cols = ["mel_marker","immune_marker","stromal_marker","cnv_burden_v2"]
# add starfysh mel_frac / immune_frac / other_frac if present
for c in ["mel_frac","immune_frac","other_frac"]:
    if c in a.obs.columns: score_cols.append(c)

summary = {}
for K in Ks:
    df = a.obs.groupby(f"cc_K{K}", observed=True)[score_cols].mean()
    df["n"] = a.obs.groupby(f"cc_K{K}", observed=True).size()
    df.to_csv(OUT / f"cluster_summary_K{K}.csv")
    summary[K] = df

# tumor call rule: rank each cluster by z-scored (mel_marker + cnv_burden_v2 + mel_frac)
K_pub = 7
df = summary[K_pub].copy()
def z(x): return (x - x.mean()) / (x.std() + 1e-9)
tumor_score = z(df["mel_marker"]) + z(df["cnv_burden_v2"])
if "mel_frac" in df.columns: tumor_score = tumor_score + z(df["mel_frac"])
if "immune_marker" in df.columns: tumor_score = tumor_score - z(df["immune_marker"])
df["tumor_score"] = tumor_score
df["call"] = np.where(tumor_score > 0, "TUMOR", "non-tumor")
df = df.sort_values("tumor_score", ascending=False)
df.to_csv(OUT / f"tumor_call_K{K_pub}.csv")
print(f"\nK={K_pub} per-cluster tumor call:")
print(df[["n","mel_marker","cnv_burden_v2","mel_frac","immune_marker","tumor_score","call"]].to_string())

# per-spot tumor mask
tumor_clusters = df[df["call"]=="TUMOR"].index.tolist()
a.obs["tumor_mask"] = a.obs[f"cc_K{K_pub}"].isin(tumor_clusters).astype(int)
print(f"\ntumor spots (K={K_pub}, clusters {tumor_clusters}): {int(a.obs['tumor_mask'].sum())} / {len(a)} ({100*a.obs['tumor_mask'].mean():.1f}%)")

# ================================================================
# 6. Plots
# ================================================================
print("[6] plots")
xy = a.obsm["spatial"]

fig, axes = plt.subplots(3, 3, figsize=(18, 17))

# (a) CellCharter K=7
ax = axes[0,0]
ax.scatter(xy[:,0], -xy[:,1], c=CLS[K_pub], s=18, cmap="tab10", linewidths=0)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"CellCharter K={K_pub}")
for c in np.unique(CLS[K_pub]):
    m = CLS[K_pub] == c
    if m.sum() < 5: continue
    cx, cy = xy[m,0].mean(), -xy[m,1].mean()
    ax.text(cx, cy, str(c), fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# (b) mel_marker
def _cbar(ax, xy, v, title, cmap="magma"):
    lo, hi = np.percentile(v, [2, 98])
    im = ax.scatter(xy[:,0], -xy[:,1], c=v, s=18, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7)

_cbar(axes[0,1], xy, a.obs["mel_marker"].values, "melanoma marker score")
_cbar(axes[0,2], xy, a.obs["cnv_burden_v2"].values, "CNV burden (Σ|X_cnv|)")
_cbar(axes[1,0], xy, a.obs["mel_frac"].values if "mel_frac" in a.obs else np.zeros(len(a)),
      "Starfysh mel_frac (TACCO)", cmap="viridis")
_cbar(axes[1,1], xy, a.obs["immune_marker"].values, "immune marker", cmap="Blues")
_cbar(axes[1,2], xy, a.obs["stromal_marker"].values, "stromal marker", cmap="Greens")

# tumor mask
ax = axes[2,0]
tm = a.obs["tumor_mask"].values.astype(bool)
ax.scatter(xy[~tm,0], -xy[~tm,1], s=16, c="#dddddd", linewidths=0, label=f"non-tumor n={int((~tm).sum())}")
ax.scatter(xy[ tm,0], -xy[ tm,1], s=18, c="crimson", linewidths=0, label=f"tumor n={int(tm.sum())}")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"TUMOR call (K={K_pub} clusters {tumor_clusters})")
ax.legend(loc="lower left", fontsize=9, frameon=False)

# per-cluster tumor score bars
ax = axes[2,1]
df_sorted = df.sort_values("tumor_score")
colors = ["crimson" if c=="TUMOR" else "#7f7f7f" for c in df_sorted["call"]]
ax.barh(range(len(df_sorted)), df_sorted["tumor_score"], color=colors)
ax.set_yticks(range(len(df_sorted))); ax.set_yticklabels([f"cl{i}" for i in df_sorted.index])
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("z-sum tumor_score"); ax.set_title("per-cluster tumor score")

# concordance heatmap: cluster × signal (z-scored)
ax = axes[2,2]
show = ["mel_marker","cnv_burden_v2","mel_frac","immune_marker","stromal_marker"]
show = [c for c in show if c in df.columns]
Zc = np.column_stack([z(df[c]) for c in show])
Zc_sorted = Zc[np.argsort(-df["tumor_score"].values)]
im = ax.imshow(Zc_sorted, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_yticks(range(len(df))); ax.set_yticklabels([f"cl{i}" for i in np.argsort(-df['tumor_score'].values)])
ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=45, ha="right")
plt.colorbar(im, ax=ax, shrink=0.7, label="z-score across clusters")
ax.set_title("cluster × signal z-score")

plt.tight_layout()
plt.savefig(OUT / "tumor_id_overview.png", dpi=150, bbox_inches="tight")
plt.close()

# save minimal h5ad
save_cols = ["mel_marker","immune_marker","stromal_marker","cnv_burden_v2","cnv_std",
             "mel_frac" if "mel_frac" in a.obs else "compartment",
             "immune_frac" if "immune_frac" in a.obs else "compartment",
             f"cc_K{K_pub}", "tumor_mask"]
save_cols = [c for c in save_cols if c in a.obs.columns]
a.obs[save_cols].to_csv(OUT / "per_spot_tumor_call.csv")
print(f"done → {OUT}")
