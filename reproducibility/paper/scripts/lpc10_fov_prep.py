"""Saturn targets for the LPC10 FOV (injection site + distal WM, cortex excluded).

Both reductions are recomputed on THIS spot set only:
  RNA  = GLM-PCA (Poisson, Townes) over HVGs of the FOV spots
  ATAC = SnapATAC2 5-kb tile matrix built from fragments for these barcodes ->
         select_features -> spectral embedding, component 1 (depth) dropped
Annotations carried through: region (lesion / distal WM / other), microglia and myelin
scores, CellCharter cluster, distance to the lesion, and inflammatory marker levels.
"""
import os, sys, numpy as np, pandas as pd, scipy.sparse as sp, snapatac2 as snap
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import _read_h5ad_safe, LPC_TP, INFL_GENES, lognorm, save, H
from saturn.preprocess import rna_glmpc_real, scale_target
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
FRAG = f"{H}/mouse_atlas/mouse_aging/GSM9249009_07_LPC10S1_atac_fragments.tsv.gz"
TILE = f"{H}/Hole/gaston_mix_mm/results/atac_tiles"

z = np.load(f"{R}/lpc10_fov.npz", allow_pickle=True)
mask = z["fov_mask"].astype(bool); bc_all = np.asarray(z["barcodes"], dtype=object)
xy_all, lab, zmg, myz = z["xy"], z["lab"], z["zmg"], z["myz"]
les, dis = set(z["lesion"].tolist()), set(z["distal"].tolist())
idx = np.where(mask)[0]
region = np.array(["lesion" if i in les else ("distal_WM" if i in dis else "other") for i in idx])
bc = [str(b) for b in bc_all[idx]]
print(f"FOV spots {len(idx)}: {pd.Series(region).value_counts().to_dict()}", flush=True)

r = _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['rna']}")
r = r[pd.Index(bc)].copy()
Yr, ir = rna_glmpc_real(r.X, K=10)
print(f"   RNA GLM-PCA on FOV: {Yr.shape}", flush=True)

wl = set(bc) | {f"{b}-1" for b in bc}
tmp = f"{TILE}/_fov_tmp.h5ad"
data = snap.pp.import_fragments(FRAG, chrom_sizes=snap.genome.mm10, file=tmp,
                               sorted_by_barcode=False, min_num_fragments=0, whitelist=wl)
snap.pp.add_tile_matrix(data, bin_size=5000)
A = data.to_memory(); data.close(); os.remove(tmp)
A.obs_names = [str(b)[:-2] if str(b).endswith("-1") else str(b) for b in A.obs_names]
A = A[[b for b in bc if b in set(A.obs_names)]].copy()
snap.pp.select_features(A, n_features=250000)
snap.tl.spectral(A, n_comps=15)
Ysp = np.asarray(A.obsm["X_spectral"])
dep = np.asarray(sp.csr_matrix(A.X).sum(1)).ravel()
from scipy.stats import spearmanr
print(f"   ATAC tiles {A.shape}; spectral comp1 |corr| with depth "
      f"{abs(spearmanr(Ysp[:,0], dep)[0]):.2f} -> dropped", flush=True)
Ya = scale_target(Ysp[:, 1:], "global")

les_xy = xy_all[np.array(sorted(les))]
d_les = cKDTree(les_xy).query(xy_all[idx], k=1)[0]
ex = dict(Y_RNA=Yr, Y_ATAC=Ya.astype(np.float32), depth_RNA=ir["depth"], depth_ATAC=dep.astype(np.float32),
          ann_region=region, ann_domain=lab[idx].astype(str),
          ann_microglia_z=zmg[idx].astype(np.float32), ann_myelin_z=myz[idx].astype(np.float32),
          ann_dist_to_lesion=d_les.astype(np.float32))
genes = [g for grp in INFL_GENES.values() for g in grp][:14] + ["Mbp", "Plp1", "Snap25"]
ex.update(lognorm(sp.csr_matrix(r.X), genes, r.var_names))
save("lpc10_fov", xy_all[idx], bc, ["RNA", "ATAC"], ex)
