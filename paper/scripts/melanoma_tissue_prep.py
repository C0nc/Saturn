"""Saturn targets for the melanoma slide, IN-TISSUE ONLY, both reductions recomputed.

`melanoma_whole_prep.py` kept 2,489 of the 2,500 DBiT pixels -- effectively the whole
grid, with no in-tissue filter (there is no `in_tissue` column in the object to filter
on). 490 of those spots had `tumor_mask = nan`, i.e. the per-spot tumour caller refused
them, and their median RNA depth was 767 against 8,274 for assigned spots. That
population made rho_ATAC a partly depth-driven axis (Spearman vs log depth +0.43) and
drove CellRank's terminal states (fate vs log depth +0.56 against +0.04 for the tumour
mask).

QC here:
    tumour call assigned  AND  RNA depth >= 500  AND  ATAC depth >= 2000
and -- the part that matters -- GLM-PCA and the SnapATAC2 spectral embedding are
recomputed on the surviving spots, per the standing rule that region selection is only
worth anything if the reductions follow it.
"""
import sys
import hdf5plugin  # noqa: F401
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, snapatac2 as snap
from scipy.stats import spearmanr
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import lognorm, save, H
from saturn.preprocess import rna_glmpc_real, scale_target

R = f"{H}/Hole/gaston_mix_mm/results/melanoma"
CALL = f"{H}/Saturn/paper_figures/figure2/melanoma_tumor_id/per_spot_tumor_call.csv"
MARKERS = ["MITF", "SOX10", "PMEL", "MLANA", "AXL", "PTPRC", "CD74", "COL1A1", "CD3E", "CD68"]
MIN_RNA, MIN_ATAC = 500, 2000

r = ad.read_h5ad(f"{R}/melanoma_rna.h5ad")
a = ad.read_h5ad(f"{R}/atac_snap.h5ad")
bc = [b for b in map(str, r.obs_names) if b in set(map(str, a.obs_names))]
r = r[pd.Index(bc)].copy(); a = a[pd.Index(bc)].copy()

t = pd.read_csv(CALL, index_col=0); t.index = t.index.astype(str); t = t.reindex(bc)
dr = np.asarray(sp.csr_matrix(r.X).sum(1)).ravel().astype(float)
da0 = np.asarray(sp.csr_matrix(a.X).sum(1)).ravel().astype(float)
keep = t["tumor_mask"].notna().values & (dr >= MIN_RNA) & (da0 >= MIN_ATAC)
print(f"QC: {len(bc)} spots -> {keep.sum()} in-tissue "
      f"({(~t['tumor_mask'].notna().values).sum()} unassigned, "
      f"{(dr < MIN_RNA).sum()} low RNA, {(da0 < MIN_ATAC).sum()} low ATAC)", flush=True)
print(f"    median RNA depth {np.median(dr):.0f} -> {np.median(dr[keep]):.0f}; "
      f"ATAC {np.median(da0):.0f} -> {np.median(da0[keep]):.0f}", flush=True)
bc = [b for b, k in zip(bc, keep) if k]
r = r[pd.Index(bc)].copy(); a = a[pd.Index(bc)].copy(); t = t.reindex(bc)

Yr, ir = rna_glmpc_real(r.X, K=10)                      # recomputed on kept spots
# After QC, log RNA depth and the CNV call are nearly the same variable (rho -0.88) and
# the unregressed axis came out more depth (partial rho 0.45) than CNV (0.19). GLM-PCA's
# Poisson offset does not remove it, so depth is regressed out of the targets the way
# methyl_pca(depth_mode="regress") already does on the methylation side.
_ld = np.log1p(np.asarray(sp.csr_matrix(r.X).sum(1)).ravel().astype(float))
_A = np.c_[np.ones_like(_ld), _ld]
Yr = np.asarray(Yr, float) - _A @ np.linalg.lstsq(_A, np.asarray(Yr, float), rcond=None)[0]
Yr = scale_target(Yr, "global").astype(np.float32)
print(f"   RNA GLM-PCA {Yr.shape}, log-depth regressed out", flush=True)
snap.pp.select_features(a, n_features=250000)
snap.tl.spectral(a, n_comps=15)                         # recomputed on kept spots
Ysp = np.asarray(a.obsm["X_spectral"])
dep = np.asarray(sp.csr_matrix(a.X).sum(1)).ravel().astype(np.float32)
print(f"   spectral comp1 |corr| with depth {abs(spearmanr(Ysp[:, 0], dep)[0]):.2f} -> dropped", flush=True)
Ya = scale_target(Ysp[:, 1:], "global")

ex = dict(Y_RNA=Yr, Y_ATAC=Ya.astype(np.float32), depth_RNA=ir["depth"], depth_ATAC=dep)
for c in ["cnv_burden_v2", "mel_marker", "immune_marker", "stromal_marker", "mel_frac", "immune_frac"]:
    if c in t: ex[f"ann_{c}"] = t[c].values.astype(np.float32)
for c in ["cc_K7", "tumor_mask"]:
    if c in t: ex[f"ann_{c}"] = t[c].astype(str).values
ex.update(lognorm(sp.csr_matrix(r.X), MARKERS, r.var_names))
save("melanoma_tissue", np.asarray(r.obsm["spatial"], float), bc, ["RNA", "ATAC"], ex)
