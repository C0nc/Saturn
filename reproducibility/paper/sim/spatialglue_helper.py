"""Run SpatialGlue in env sat-scigma; dump per-modality and joint latents.

SpatialGlue (Long et al., Nat Methods 2024) integrates two spatial modalities
with within- and cross-modality graph attention. Like SCIGMA it exposes a latent
PER modality (`emb_latent_omics1/2`) as well as the fused one, so modality m's
axis is chosen from its own latent (best variance-explained dimension).

Preprocessing follows SpatialGlue's own RNA + ATAC tutorial
("Spatial-epigenome-transcriptome"): RNA -> normalise/log1p/scale -> PCA;
ATAC -> LSI (its `lsi`, which drops the depth-tracking first component).

Usage:  python spatialglue_helper.py in.npz out.npz
  in.npz  : counts_0 (RNA), counts_1 (ATAC) int (N, G), coords (N, 2)
  out.npz : emb_0, emb_1 per-modality latents, emb_combined fused latent
"""
import sys, os, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/SpatialGlue")
import numpy as np, torch, scanpy as sc, anndata


def main(inp, outp, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from SpatialGlue.preprocess import pca, lsi, construct_neighbor_graph
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
    z = np.load(inp)
    coords = z["coords"].astype(np.float32)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rna = anndata.AnnData(np.asarray(z["counts_0"], dtype=np.float32))
    rna.obsm["spatial"] = coords
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna); sc.pp.scale(rna)
    rna.obsm["feat"] = pca(rna, n_comps=30)

    atac = anndata.AnnData(np.asarray(z["counts_1"], dtype=np.float32))
    atac.obsm["spatial"] = coords
    lsi(atac, use_highly_variable=False, n_components=31)
    atac.obsm["feat"] = atac.obsm["X_lsi"].copy()

    data = construct_neighbor_graph(rna, atac, datatype="Spatial-epigenome-transcriptome")
    model = Train_SpatialGlue(data, datatype="Spatial-epigenome-transcriptome", device=dev)
    out = model.train()
    np.savez_compressed(outp,
                        emb_0=np.asarray(out["emb_latent_omics1"], dtype=np.float32),
                        emb_1=np.asarray(out["emb_latent_omics2"], dtype=np.float32),
                        emb_combined=np.asarray(out["SpatialGlue"], dtype=np.float32))
    print(f"SpatialGlue ok: emb_0 {out['emb_latent_omics1'].shape} "
          f"emb_1 {out['emb_latent_omics2'].shape}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
