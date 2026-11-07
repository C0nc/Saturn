"""Post-hoc explanation of the LPC10 FOV fit for inflammatory-response genes.

For every inflammatory gene: GASTON per-domain Poisson fits along BOTH axes
(d_RNA = lesion-distance axis, d_ATAC = inflammation axis), reporting which axis
explains it better, plus its levels in lesion / distal WM / other.
"""
import sys, os, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from prep_real2 import _read_h5ad_safe, LPC_TP, INFL_GENES, H
from lpc_gene_isodepth import gaston_bins, pois_fit
from lpc_gaston_fig4l import gaston_domains
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "svg.fonttype": "none"})
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/lpc10_fov_posthoc"; os.makedirs(OUT, exist_ok=True)
NB, KDOM = 25, 4
ax = pd.read_csv(f"{R}/lpc10_fov/axes.csv", index_col=0)
z = dict(np.load(f"{R}/targets_lpc10_fov.npz", allow_pickle=True))
dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
reg = z["ann_region"].astype(str)
r = _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['rna']}")
r = r[pd.Index([str(b) for b in ax.index])].copy()
X = sp.csr_matrix(r.X); vn = list(map(str, r.var_names)); lib = np.asarray(X.sum(1)).ravel().clip(1)
GRP = {g: k for k, gs in INFL_GENES.items() for g in gs}


def best_fit(cnt, d):
    idx, cen, keep = gaston_bins(d, NB)
    yb = np.array([cnt[idx == b].sum() for b in keep]); eb = np.array([lib[idx == b].sum() for b in keep])
    best = None
    for bp in np.quantile(cen, np.linspace(0.2, 0.8, 10)):
        f = pois_fit(yb, eb, cen, bp)
        if best is None or f["R2"] > best["R2"]:
            best = f
    return best["R2"], cen, yb / np.maximum(eb, 1), best["rate"]


rows = []
for g in [x for grp in INFL_GENES.values() for x in grp]:
    if g not in vn:
        continue
    c = np.asarray(X[:, vn.index(g)].todense()).ravel()
    if c.sum() < 50:
        continue
    r2R, cR, obsR, fitR = best_fit(c, dR)
    r2A, cA, obsA, fitA = best_fit(c, dA)
    ln = np.log1p(c / lib * 1e4)
    rows.append(dict(gene=g, group=GRP[g], umi=int(c.sum()), R2_dRNA=round(r2R, 3), R2_dATAC=round(r2A, 3),
                     better="d_ATAC" if r2A > r2R else "d_RNA",
                     lesion=round(float(ln[reg == "lesion"].mean()), 2),
                     distal_WM=round(float(ln[reg == "distal_WM"].mean()), 2),
                     other=round(float(ln[reg == "other"].mean()), 2)))
df = pd.DataFrame(rows).sort_values("R2_dATAC", ascending=False)
df["lesion_vs_other"] = (df.lesion - df.other).round(2)
df["distal_vs_other"] = (df.distal_WM - df.other).round(2)
df.to_csv(f"{OUT}/infl_gene_axis_fits.csv", index=False)
print(df.head(15).to_string(index=False))
print(f"\nbetter explained by d_ATAC: {int((df.better=='d_ATAC').sum())}/{len(df)} genes; "
      f"median R2 d_ATAC {df.R2_dATAC.median():.2f} vs d_RNA {df.R2_dRNA.median():.2f}")
print("\nenriched in BOTH regions (lesion and distal vs other, >0.1):")
both = df[(df.lesion_vs_other > 0.1) & (df.distal_vs_other > 0.1)]
print(both[["gene", "group", "lesion_vs_other", "distal_vs_other", "R2_dATAC"]].to_string(index=False))
top = df.head(6)
fig, AX = plt.subplots(2, 6, figsize=(19, 6))
for c_, (_, row) in enumerate(top.iterrows()):
    cnt = np.asarray(X[:, vn.index(row.gene)].todense()).ravel()
    for k, (d, nm) in enumerate([(dR, "d_RNA"), (dA, "d_ATAC")]):
        r2, cen, obs, fit = best_fit(cnt, d)
        a_ = AX[k, c_]
        a_.plot(cen, np.log(obs * 1e6 + 1), "o", ms=3, color="#999999")
        a_.plot(cen, np.log(fit * 1e6 + 1), color="#1f77b4" if k == 0 else "#d62728", lw=1.6)
        a_.set_xlabel(nm); a_.set_ylabel("log(rate x 1e6 + 1)")
        a_.set_title(f"{row.gene} on {nm}: R2 {r2:.2f}")
        a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("LPC10 FOV: inflammatory genes along the lesion-distance axis (top) and the inflammation axis (bottom)",
             fontsize=9)
fig.tight_layout()
for e in ("png", "svg"):
    fig.savefig(f"{OUT}/infl_genes_both_axes.{e}", dpi=250 if e == "png" else None)
print("\nwritten", OUT)
