"""CellCharter labels of LPC10 S1, rerun with the producer, as a check only.

Recovered from Saturn/paper_figures/figure2/lpc_timepoint_cellcharter.py (2026-08-06), unchanged except that it runs one time
point (the original ran LPC5 then LPC10 in one process) and writes next to the published labels instead of over them.
The producer does not seed scVI, so the labels change between runs, and the cortex clusters (K = 10: 1, 5, 2) were picked by eye
from the published labels. So the chain keeps LPC10_cc_labels.csv as an input; this step reports how close a rerun gets.
"""
import os, pathlib, warnings, h5py
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as ssp
warnings.filterwarnings("ignore")
import scvi, squidpy as sq, cellcharter as cc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ari

H = os.environ["SATURN_ROOT"]
TP = os.environ.get("TP_DIR", f"{H}/claude_code_tutorial_lpc/results/timepoints")
PUB = pathlib.Path(f"{H}/Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison")
OUT = pathlib.Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn/lpc10_cellcharter_rerun"); OUT.mkdir(exist_ok=True, parents=True)
TIMEPOINTS = {t: f"{TP}/arcseq_{t}S1_rna_adata.h5ad" for t in os.environ.get("TPS", "LPC10").split(",")}
K_LIST = [3, 5, 7, 10]; N_HVG = 3000; SCVI_EPOCHS = 200


def load_h5ad_minimal(path):
    with h5py.File(path, "r") as f:
        Xg = f["X"]
        if isinstance(Xg, h5py.Dataset):
            X = ssp.csr_matrix(Xg[:])
        else:
            data = Xg["data"][:]; ind = Xg["indices"][:]; ptr = Xg["indptr"][:]
            shp = tuple(Xg.attrs.get("shape", (len(ptr) - 1, ind.max() + 1)))
            enc = Xg.attrs.get("encoding-type", "csr_matrix")
            X = (ssp.csc_matrix((data, ind, ptr), shape=shp).tocsr() if enc == "csc_matrix" else ssp.csr_matrix((data, ind, ptr), shape=shp))
        return X, np.array([v.decode() for v in f["obs/_index"][:]]), np.array([v.decode() for v in f["var/_index"][:]]), f["obsm/spatial"][:]


for tp, rna_path in TIMEPOINTS.items():
    X, obs, var, coords = load_h5ad_minimal(rna_path)
    ad = sc.AnnData(X=X, obs=pd.DataFrame(index=obs), var=pd.DataFrame(index=var))
    ad.obsm['spatial'] = coords; ad.layers['counts'] = ad.X.copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor='seurat_v3', layer='counts')
    ad_hvg = ad[:, ad.var['highly_variable']].copy()
    scvi.model.SCVI.setup_anndata(ad_hvg, layer='counts')
    model = scvi.model.SCVI(ad_hvg, n_layers=2, n_latent=10, gene_likelihood='nb')
    model.train(max_epochs=SCVI_EPOCHS, early_stopping=True, check_val_every_n_epoch=1, plan_kwargs=dict(lr=1e-3))
    ad.obsm['X_scVI'] = model.get_latent_representation()
    sq.gr.spatial_neighbors(ad, coord_type='generic', delaunay=True)
    cc.gr.remove_long_links(ad)
    cc.gr.aggregate_neighbors(ad, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter')
    autok = cc.tl.ClusterAutoK(n_clusters=(min(K_LIST), max(K_LIST)), max_runs=3, model_params=dict(random_state=0))
    autok.fit(ad, use_rep='X_cellcharter')
    for K in K_LIST:
        ad.obs[f'cc_K{K}'] = np.asarray(autok.predict(ad, use_rep='X_cellcharter', k=K)).astype(int)
    ad.obs[[f'cc_K{K}' for K in K_LIST]].to_csv(OUT / f'{tp}_cc_labels.csv')
    if tp != "LPC10":
        continue
    a = ad.obs; b = pd.read_csv(PUB / f'{tp}_cc_labels.csv', index_col=0).loc[a.index]
    rows = [dict(K=K, ARI=ari(a[f'cc_K{K}'], b[f'cc_K{K}']), identical=bool(a[f'cc_K{K}'].values.tolist() == b[f'cc_K{K}'].values.tolist())) for K in K_LIST]
    # cortex decision used by the FOV: match rerun clusters to the published K = 10 ids, then compare the mask
    u, v = a['cc_K10'].astype(int).values, b['cc_K10'].astype(int).values
    C = np.zeros((10, 10), int); np.add.at(C, (u, v), 1)
    r, c = linear_sum_assignment(-C); mp = dict(zip(r, c)); mapped = np.array([mp[x] for x in u])
    T = pd.DataFrame(rows); T["cortex_mask_agreement_K10"] = float((np.isin(mapped, [1, 5, 2]) == np.isin(v, [1, 5, 2])).mean())
    T.to_csv(OUT / "rerun_vs_published.csv", index=False); print(T.to_string(index=False))
