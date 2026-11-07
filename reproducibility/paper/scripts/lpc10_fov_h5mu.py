"""LPC10 lesion FOV (Fig 2) as one MuData object: RNA + ATAC, lesion annotation, Saturn axes and
named domains, and cell2location cell-type proportions.

  1. cell2location on the WHOLE LPC10S1 section (8,180 spots) with the cached mouse-cortex
     signatures (results/cell2location/mouse_cortex_inf_aver.csv, 26 types) and the settings of
     run_cell2location.py (N_cells_per_location=10, detection_alpha=20, 10,000 epochs), cached to
     results/cell2location/LPC10_proportions.h5ad; then subset to the FOV.
  2. MuData on the FOV spots (the exact barcodes of the Saturn fit `lpc10_fov`):
       mod['rna']   raw UMI counts, all genes        (arcseq_LPC10S1_rna_adata.h5ad)
                    obsm['X_glmpca']  the Saturn RNA target (GLM-PCA, 10 comps, FOV-only)
       mod['atac']  peak counts                      (arcseq_LPC10S1_atac_adata.h5ad)
                    obsm['X_spectral_5kb']  the Saturn ATAC target (5 kb tiles -> spectral, comp 1 dropped)
       obs          region (lesion / distal WM focus / other), microglia_z, myelin_z,
                    dist_to_lesion, CellCharter domain, Saturn rho_RNA / rho_ATAC and their named
                    domains (Fig 2 labels), sequencing depths, dominant cell type
       obsm         spatial; c2l_proportions / c2l_abundance_q05 (DataFrames, 26 types)
       uns          provenance, Saturn R2, domain palette, label maps

Usage:
  conda activate c2l;            python scripts/lpc10_fov_h5mu.py --deconvolve   (step 1, GPU)
  conda activate spatialmultivelo; python scripts/lpc10_fov_h5mu.py              (step 2)
cell2location runs in its own env `c2l` (python 3.10, pip cell2location): in spatialmultivelo its
scvi import needs a newer anndata than the one every Saturn fit was made with, so that env is left alone.
"""
import json, os, sys
import numpy as np, pandas as pd, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")

H = "<workspace>"
RNA_H5AD = f"{H}/claude_code_tutorial_lpc/results/timepoints/arcseq_LPC10S1_rna_adata.h5ad"
ATAC_H5AD = f"{H}/claude_code_tutorial_lpc/results/timepoints/arcseq_LPC10S1_atac_adata.h5ad"
R = f"{H}/Hole/gaston_mix_mm/results"
RS = f"{R}/real_saturn"
C2L = f"{R}/cell2location"
OUT = f"{RS}/lpc10_fov_mudata"
DS = "lpc10_fov"
N_MARK = 150


def deconvolve_whole_section():
    out = f"{C2L}/LPC10_proportions.h5ad"
    if os.path.exists(out):
        print(f"cell2location cached: {out}", flush=True)
        return ad.read_h5ad(out)
    if "--deconvolve" not in sys.argv:
        sys.exit(f"{out} missing: run step 1 first (conda activate c2l; python {__file__} --deconvolve)")
    import torch
    from cell2location.models import Cell2location
    inf = pd.read_csv(f"{C2L}/mouse_cortex_inf_aver.csv", index_col=0)
    a = ad.read_h5ad(RNA_H5AD)
    a.X = a.X.astype(np.float32)
    shared = [g for g in a.var_names if g in set(inf.index)]
    # informative genes only: union of the N_MARK most type-specific genes per reference type
    # (specificity = a type's signature / its sum over types); cuts memory and time ~4x
    sig = inf.loc[shared]; spec = sig.div(sig.sum(1) + 1e-9, axis=0)
    mark = set().union(*[set(spec[c].nlargest(N_MARK).index) for c in sig.columns])
    shared = [g for g in shared if g in mark]
    a = a[:, shared].copy(); inf = inf.loc[shared]
    print(f"LPC10S1 {a.n_obs} spots, {len(shared)} genes shared with the {inf.shape[1]}-type reference", flush=True)
    Cell2location.setup_anndata(adata=a, batch_key=None)
    mod = Cell2location(a, cell_state_df=inf, N_cells_per_location=10, detection_alpha=20)
    use_gpu = torch.cuda.is_available() and "--cpu" not in sys.argv     # the shared GPU is often saturated
    if not use_gpu: torch.set_num_threads(32)
    acc = {"accelerator": "gpu"} if use_gpu else {"accelerator": "cpu"}
    mod.train(max_epochs=10000, batch_size=None, train_size=1.0, **acc)
    a = mod.export_posterior(a, sample_kwargs={"num_samples": 1000, "batch_size": mod.adata.n_obs, **acc})
    abund = a.obsm["q05_cell_abundance_w_sf"]
    abund.columns = [c.replace("q05cell_abundance_w_sf_", "") for c in abund.columns]
    prop = abund.values / (abund.values.sum(1, keepdims=True) + 1e-12)
    o = ad.AnnData(X=prop.astype(np.float32), obs=pd.DataFrame(index=a.obs_names), var=pd.DataFrame(index=abund.columns))
    o.layers["abundance_q05"] = abund.values.astype(np.float32)
    o.uns["settings"] = dict(reference="mouse_cortex_inf_aver.csv", N_cells_per_location=10, detection_alpha=20,
                             max_epochs=10000, batch_size="full", genes=len(shared), section="LPC10S1 whole",
                             gene_selection=f"union of top {N_MARK} type-specific reference genes",
                             device=acc["accelerator"],
                             versions=dict(cell2location=str(__import__("cell2location").__version__),
                                           scvi=str(__import__("scvi").__version__), torch=str(torch.__version__)))
    mod.save(f"{C2L}/LPC10_c2lmodel", overwrite=True)                 # model first: a write error must not lose it
    o.write_h5ad(out)
    print(f"-> {out}", flush=True)
    return o


def saturn_domains(ax, z):
    """Fig 2's domain cut and its composition-asserted labels (fig2_lpc.label_domains)"""
    sys.path.insert(0, "<workspace>/Saturn/published_figures/fig2")
    from lpc_gaston_fig4l import gaston_domains
    import fig2_lpc as F2
    from domain_k import kdom
    D = dict(d={m: ax[f"d_{m}"].values for m in ("RNA", "ATAC")},
             lab={m: gaston_domains(ax[f"d_{m}"].values, z[f"Y_{m}"], L_=kdom(DS, m)) for m in ("RNA", "ATAC")},
             region=pd.Series(z["ann_region"].astype(str)).replace({"distal_WM": "distal WM focus"}).values,
             micro=z["ann_microglia_z"].astype(float), myelin=z["ann_myelin_z"].astype(float))
    names = F2.label_domains(D)
    return D, names


def main():
    import mudata as md
    from paper_style import DOMAIN, REGION
    from prep_real2 import _read_h5ad_safe
    os.makedirs(OUT, exist_ok=True)
    ax = pd.read_csv(f"{RS}/{DS}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{RS}/targets_{DS}.npz", allow_pickle=True))
    assert [str(b) for b in z["barcodes"]] == bc, "targets and axes disagree on spot order"
    summ = json.load(open(f"{RS}/{DS}/summary.json"))

    rna = _read_h5ad_safe(RNA_H5AD)[pd.Index(bc)].copy()
    atac = _read_h5ad_safe(ATAC_H5AD)[pd.Index(bc)].copy()
    for a, nm in ((rna, "rna"), (atac, "atac")):
        a.obs = a.obs[[c for c in ("in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
                       if c in a.obs]].copy()
        a.var_names_make_unique()
    rna.obsm["X_glmpca"] = np.asarray(z["Y_RNA"], np.float32)
    atac.obsm["X_spectral_5kb"] = np.asarray(z["Y_ATAC"], np.float32)
    rna.obs["n_counts"] = np.asarray(rna.X.sum(1)).ravel().astype(np.float32)
    atac.obs["n_fragments_peaks"] = np.asarray(atac.X.sum(1)).ravel().astype(np.float32)

    D, names = saturn_domains(ax, z)
    obs = pd.DataFrame(index=bc)
    obs["region"] = pd.Categorical(D["region"], categories=["lesion", "distal WM focus", "other"])
    obs["microglia_z"] = D["micro"].astype(np.float32); obs["myelin_z"] = D["myelin"].astype(np.float32)
    obs["dist_to_lesion"] = z["ann_dist_to_lesion"].astype(np.float32)
    obs["cellcharter_domain"] = pd.Categorical(z["ann_domain"].astype(str))
    for m in ("RNA", "ATAC"):
        obs[f"rho_{m}"] = ax[f"d_{m}"].values.astype(np.float32)
        obs[f"domain_{m}"] = pd.Categorical([names[m][k] for k in D["lab"][m]],
                                            categories=[names[m][k] for k in sorted(names[m])])
    obs["depth_RNA"] = z["depth_RNA"].astype(np.float32); obs["depth_ATAC"] = z["depth_ATAC"].astype(np.float32)

    c2l = deconvolve_whole_section()
    miss = set(bc) - set(c2l.obs_names)
    assert not miss, f"{len(miss)} FOV spots absent from the deconvolution"
    c = c2l[pd.Index(bc)]
    prop = pd.DataFrame(np.asarray(c.X), index=bc, columns=c.var_names.astype(str))
    abund = pd.DataFrame(np.asarray(c.layers["abundance_q05"]), index=bc, columns=c.var_names.astype(str))
    obs["dominant_cell_type"] = pd.Categorical(prop.idxmax(1))

    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.obs = mdata.obs.join(obs)
    mdata.obsm["spatial"] = np.c_[ax.x.values, ax.y.values].astype(np.float32)
    mdata.obsm["c2l_proportions"] = prop
    mdata.obsm["c2l_abundance_q05"] = abund
    mdata.uns["provenance"] = dict(
        dataset="LPC10S1 ARC-seq (mouse LPC demyelination, day 10), GEO GSM9247593 (RNA) / GSM9249009 (ATAC)",
        fov="lesion + distal white-matter FOV, cut from DBiT microglia hotspots; cortex excluded (Saturn fit lpc10_fov)",
        rna=RNA_H5AD, atac=ATAC_H5AD,
        saturn_fit=f"{RS}/{DS}", saturn_seed=42, saturn_R2_full={m: v["full"] for m, v in summ["R2_whole"].items()},
        deconvolution=dict(c2l.uns["settings"]),
        script=os.path.abspath(__file__))
    mdata.uns["domain_labels"] = {m: {str(k): v for k, v in names[m].items()} for m in names}
    mdata.uns["palette"] = dict(region={k: REGION.get(k, "#D5D5D5") for k in ("lesion", "distal WM focus", "other")},
                                domain={v: DOMAIN[v] for m in names for v in names[m].values()})
    mdata.update()
    f = f"{OUT}/LPC10_lesion_FOV.h5mu"
    mdata.write(f)
    mdata.write(f.replace(".h5mu", ".gz.h5mu"), compression="gzip")       # the copy downstream scripts read
    print(mdata)
    print("\nregion:", obs.region.value_counts().to_dict())
    print("dominant cell type by region:\n",
          pd.crosstab(obs.region, obs.dominant_cell_type).loc[:, lambda t: t.sum() > 10].to_string())
    print("mean proportion of microglia / macrophage / oligodendrocyte by region:\n",
          prop.groupby(obs.region.values)[[c for c in ("microglial cell", "macrophage", "oligodendrocyte",
                                                       "oligodendrocyte precursor cell", "astrocyte")
                                         if c in prop]].mean().round(3).to_string())
    print(f"\n-> {f}")


if __name__ == "__main__":
    if "--deconvolve" in sys.argv:
        deconvolve_whole_section()
    else:
        main()
