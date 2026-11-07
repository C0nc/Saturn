"""Run SCIGMA in its own env (sat-scigma, py3.8); dump PER-MODALITY latents.

SCIGMA (Ma lab) integrates spatial multi-omics with uncertainty-based
contrastive learning. Unlike most integration methods it exposes per-modality
latents (`emb_latent_omics{i}`) alongside the joint one, so modality m maps
directly to its own latent — no cross-modality selection needed.

`obsm['feat']` is the model input: PCA of log-normalised counts, matching
SCIGMA's own `preprocessing()` (data.py:215-238).

Usage:  python scigma_helper.py in.npz out.npz [epochs]
  out.npz : emb_0..emb_{M-1} per-modality latents, emb_combined joint latent
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/SCIGMA")
import numpy as np, torch, scanpy as sc, anndata

def prep(X, n_comps=30):
    a = anndata.AnnData(np.asarray(X, dtype=np.float32))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=min(n_comps, min(a.shape) - 1))
    return a.obsm["X_pca"].astype(np.float32)

def main(inp, outp, epochs=200):
    z = np.load(inp)
    coords = z["coords"].astype(np.float32)
    M = sum(1 for k in z.files if k.startswith("counts_"))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from SCIGMA import SCIGMA
    ads = {}
    for m in range(M):
        a = anndata.AnnData(np.asarray(z[f"counts_{m}"], dtype=np.float32))
        a.obsm["spatial"] = coords
        a.obsm["feat"] = prep(z[f"counts_{m}"])
        ads[f"adata_omics{m+1}"] = a
    # SCIGMA's preprocessing() bundles HVG/scaling (tuned for real assays) with
    # the graph construction the model requires. Our simulated counts already
    # have a sensible `feat`, so call only the graph half directly.
    from data import build_network, feature_graph
    for a in ads.values():
        a.uns["adj_spatial"] = build_network(a.obsm["spatial"], n_neighbors=8)
    fg1, fg2 = feature_graph(ads["adata_omics1"], ads["adata_omics2"], k=8)
    ads["adata_omics1"].obsm["adj_feature"] = fg1
    ads["adata_omics2"].obsm["adj_feature"] = fg2

    # batch_size defaults to 6000 (SCIGMA targets large assays); its forest-fire
    # sampler raises "Sample larger than population" on our 900-spot grids, so
    # cap it below N.
    N = ads["adata_omics1"].n_obs
    model = SCIGMA(ads, seed_num=42, dim_output=40, device=dev,
                    batch_size=max(64, min(512, N // 2)))
    out = model.train(epochs)
    res = {}
    for m in range(M):
        k = f"emb_latent_omics{m+1}"
        if k in out:
            res[f"emb_{m}"] = np.asarray(out[k], dtype=np.float32)
    res["emb_combined"] = np.asarray(out["emb_latent_combined"], dtype=np.float32)
    np.savez_compressed(outp, **res)
    print(f"SCIGMA ok: {[(k, v.shape) for k, v in res.items()]}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
