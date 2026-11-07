"""Why the ATAC cross term explains an RNA gene, and why it helps where rho_ATAC is steep (Fig 7 g/h follow-up).

Datasets: the two RNA+ATAC figure fits (lpc10, melanoma). Genes = cross-explained genes (FDR 5%, dR2 >= 0.01).
  absorb     for each gene: y ~ B(rho_RNA) + K  vs  + K + B(rho_ATAC), K = one candidate source of the cross signal:
               cis    the gene's own +-50 kb accessibility (hinge basis of it)
               TF     top-6 PCs of per-spot chromVAR TF activity (23/24 panel TFs)
               comp   top-6 PCs of cell-type composition (cell2location 26 types / Starfysh 28 states)
               all    cis + TF + comp
               null   the TF block or the composition block spatially rotated (5 rotations each, mean) = what any smooth 6-column block absorbs
             absorbed share = 1 - dR2_ATAC|K / dR2_ATAC. All blocks 15-NN smoothed; equal width (6 columns).
  drivers    per gene: correlation of its cross profile with each cell type and each TF (top 3 of each)
  boundary   per spot: local gradient magnitude (25-NN linear fit, standardised feature) of every cell type and TF;
             each feature's change in the rho_ATAC-steep quintile vs the rho_RNA-steep quintile (log2 ratio);
             and the gain trend split into its two parts (|grad rho_ATAC| up, |grad rho_RNA| down)
-> results/real_saturn/saturn_cross_term/<key>_mech_{genes.csv, drivers.csv, boundary.csv, summary.json}
Usage: python scripts/saturn_cross_mechanism.py [lpc10 melanoma]
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saturn_cross_term as SC
from saturn import mechanism as MP

R, OUT = SC.R, SC.OUT
DR2_MIN, NPC, KS, KG, NROT = 0.01, 6, 15, 25, 5


def composition(key, bc):
    if key == "lpc10":
        import mudata as md
        m = md.read_h5mu(f"{R}/lpc10_fov_mudata/LPC10_lesion_FOV.h5mu"); C = m.obsm["c2l_proportions"]
        C.index = m.obs_names.astype(str)
    else:
        C = pd.read_csv(f"{SC.H}/melanoma_multiseek/data/starfysh_composition.csv", index_col=0); C.index = C.index.astype(str)
    return C.reindex(bc)


def tf_activity(key, bc):
    z = np.load(f"{R}/{SC.FITS[key]['ds']}_tf_activity/tf_z_per_spot.npz", allow_pickle=True)
    return pd.DataFrame(z["z"], index=z["barcodes"].astype(str), columns=z["tfs"].astype(str)).reindex(bc)


smooth = MP.smooth


pcs = MP.pcs


local_grad = MP.local_grad


r2 = MP.r2


dR2 = MP.dR2


def run(key):
    T = pd.read_csv(f"{OUT}/{key}_spots.csv", index_col=0); T.index = T.index.astype(str); bc = list(T.index)
    xy = T[["x", "y"]].values; N = len(T)
    G = pd.read_csv(f"{OUT}/{key}_genes.csv"); hit = G[(G.q < 0.05) & (G.dR2_cross >= DR2_MIN)].gene.values
    Xm, genes = SC.rna(key, bc); col = np.isin(genes, hit) & ~pd.Series(genes).duplicated().values
    names = genes[col]; Y = SC.logcpm(Xm, col)
    C, Z = composition(key, bc), tf_activity(key, bc)
    ok = C.notna().all(1).values & Z.notna().all(1).values
    print(f"[{key}] {len(names)} cross genes, {ok.sum()}/{N} spots with composition + TF", flush=True)
    T, xy, Y, C, Z = T[ok], xy[ok], Y[ok], C[ok], Z[ok]; bc = list(T.index); N = len(T)
    keep, W, _ = SC.cis_test(key, bc, names, np.zeros((N, len(names))), np.ones(len(names), bool))
    D = MP.absorption(Y, names, T.rho_RNA.values, T.rho_X.values, xy, {"TF": Z.values, "comp": C.values},
                      cis=(keep, W), n_rot=NROT)
    D = D.rename(columns={"dR2_X": "dR2_ATAC"})[
        ["gene", "dR2_ATAC", "dR2_given_TF", "dR2_given_comp", "dR2_given_null", "dR2_given_cis", "dR2_given_all"]
        + [f"absorbed_{b}" for b in ("cis", "TF", "comp", "all", "null")]]
    D.to_csv(f"{OUT}/{key}_mech_genes.csv", index=False)
    MP.drivers(Y, names, T.rho_RNA.values, T.rho_X.values, xy, {"type": C, "tf": Z}).to_csv(
        f"{OUT}/{key}_mech_drivers.csv", index=False)
    # boundary: what changes where rho_ATAC is steep
    lg = SC.CT.steepness(T)
    top = lg >= np.quantile(lg, 0.8)
    Bd = MP.boundary_switch({"type": C, "TF": Z}, xy, lg).rename(
        columns={"grad_Xsteep": "grad_ATACsteep", "level_Xsteep": "level_ATACsteep"})
    Bd.to_csv(f"{OUT}/{key}_mech_boundary.csv", index=False)
    g = T.gain_RNA.values
    S = dict(n_genes=len(names), n_spots=N, n_cis=len(keep),
             median_absorbed={b: float(np.nanmedian(D[f"absorbed_{b}"])) for b in ("cis", "TF", "comp", "all", "null")},
             gain_vs_loggradATAC=float(spearmanr(np.log(T.grad_X), g)[0]),
             gain_vs_neg_loggradRNA=float(spearmanr(-np.log(T.grad_RNA), g)[0]),
             gradRNA_in_steep_rel=float(T.grad_RNA[top].mean() / T.grad_RNA.mean()),
             gradATAC_in_steep_rel=float(T.grad_X[top].mean() / T.grad_X.mean()))
    json.dump(S, open(f"{OUT}/{key}_mech_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1)); print(Bd.head(8).round(3).to_string())


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["lpc10", "melanoma"]):
        run(k)
