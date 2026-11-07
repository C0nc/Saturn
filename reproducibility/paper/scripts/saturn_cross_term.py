"""What the Saturn cross pathways explain, per figure dataset (Fig 7).

For each fit (RNA + one other modality X), from the saved model:
  decomposition  per spot: own term (pathway from rho_m), cross term (pathway from the other axis), full prediction
  local gain     per spot: ||Y_i - P_own_i||^2 - ||Y_i - P_full_i||^2, divided by the mean ||Y_i - Ybar||^2 and averaged
                 over the 25 nearest spots = where the other axis explains this modality's variance
                 (P_own = full prediction with the cross term held at its mean, as R2_whole.own_axis_only)
  rate diff      |log ratio| of |grad rho_X| to |grad rho_RNA| (each over its median) = where one axis changes
                 fast and the other is flat (boundaries placed differently)
  misalignment   1 - |cos| between grad rho_RNA and grad rho_X (autograd through the isodepth nets, 25-NN averaged
                 vectors) = where the two axes run in different directions
  genes          every RNA gene detected in >= 5% of spots: log1p CPM ~ B(rho_RNA)  vs  ~ B(rho_RNA) + B(rho_X),
                 B = linear + 5 ReLU hinges at the 10-90% quantiles (Saturn's pathway family); cross dR2, partial F,
                 BH; the gene's cross profile = B(rho_X) part of the full fit
  cis (ATAC)     for genes with cross FDR < 0.05: the gene's +-50 kb accessibility (summed peaks / tiles, log1p CPM)
                 fitted the same way; concordance = corr(gene cross profile, cis-window cross profile) across spots,
                 vs the same gene paired with a random other gene's window (200 draws)
Axes are oriented as in the figures (sign matched to the figure's stored axis).
-> results/real_saturn/saturn_cross_term/<key>_{spots.csv, genes.csv, summary.json, cis.csv}
  steeper        RNA's local gain vs log(|grad rho_X| / |grad rho_RNA|) (and X's gain vs the reverse): Spearman over
                 spots, p against 200 rotations of the gain field
  length         gene body length vs cross gain (top 200 vs all) and vs own-axis R2 (specificity)
Usage: python scripts/saturn_cross_term.py [key ...] [--null-only]      (keys: lpc10 melanoma merfish sprox hic)
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.spatial import cKDTree
from scipy.stats import f as fdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "<workspace>/saturn_repo/src")
import saturn
from saturn import crossterm as CT
from saturn.core import Saturn  # noqa: F401

H = "<workspace>"; R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/saturn_cross_term"
KNN, MIN_DET, NDRAW, WIN = 25, 0.05, 200, 50_000


def _fig_axes(path, cols):
    def f(bc):
        a = pd.read_csv(path, index_col=0); a.index = a.index.astype(str); return a.loc[bc, cols].values
    return f


FITS = {   # key -> targets, weights dir, figure axis file + columns (RNA, X) for orientation
    "lpc10": dict(ds="lpc10_fov", fit="lpc10_fov", X="ATAC", fig=_fig_axes(f"{R}/lpc10_fov/axes.csv", ["d_RNA", "d_ATAC"])),
    "melanoma": dict(ds="melanoma_tissue", fit="melanoma_tissue", X="ATAC",
                     fig=_fig_axes(f"{R}/melanoma_tissue/axes.csv", ["d_RNA", "d_ATAC"])),
    "merfish": dict(ds="rtt_merfish_s5_iso100_sv_pca_shrunk", fit="rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3_weights",
                    X="Isoform", fig=_fig_axes(f"{R}/rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3/axes.csv",
                                               ["d_RNA", "d_Isoform"])),
    "sprox": dict(ds="sproxseq_B1_p46_dr", fit="sproxseq_B1_p46_dr", X="Protein",
                  fig=_fig_axes(f"{R}/sproxseq_B1_fig5/spots.csv", ["rho_RNA", "rho_Protein"])),
    "hic": dict(ds="mel_hic_abins_dr50_51_rnadr", fit="mel_hic_abins_dr50_51_rnadr", X="HiC",
                fig=_fig_axes(f"{R}/mel_hic_abins_dr50_51_rnadr/axes.csv", ["d_RNA", "d_HiC"])),
}


def rna(key, bc):
    """RNA counts (spots x genes, csr) and gene names for the fit's spots"""
    if key in ("lpc10", "melanoma"):
        from domain_annot import CFG
        r = CFG[FITS[key]["ds"]]["rna"](); r.obs_names = r.obs_names.astype(str); r = r[pd.Index(bc)]
        return sp.csr_matrix(r.X), np.array([str(v) for v in r.var_names])
    if key == "merfish":
        from isoform_marker_panels import load_rtt_merfish
        Xg, genes, *_, codes = load_rtt_merfish(5, bin_um=100.0)   # the fit uses 100 um bins; the default is 50 um
        ix = pd.Index(codes).get_indexer(pd.Index(bc)); assert (ix >= 0).all(), "fit barcodes missing from the RNA bins"
        return sp.csr_matrix(Xg)[ix], genes
    from continuous_marker_panels import sprox_rna, mel_rna
    X, g = sprox_rna(bc, "B1") if key == "sprox" else mel_rna(bc)
    return sp.csr_matrix(X), np.asarray(g).astype(str)


hinge = CT.hinge


fit_split = CT.fit_split


bh = CT.bh


logcpm = CT.logcpm


def cis_test(key, bc, genes, prof, sig):
    """+-50 kb accessibility of each significant gene vs its cross profile (saturn.crossterm.cis_windows)"""
    from domain_annot import CFG
    cfg = CFG[FITS[key]["ds"]]; A = cfg["atac"](); A.obs_names = A.obs_names.astype(str); A = A[pd.Index(bc)]
    keep, W, _ = CT.cis_windows(A.X, A.var_names, genes[sig], cfg["gtf"], window=WIN)
    gi = {g: i for i, g in enumerate(genes)}
    return keep, W, np.column_stack([prof[:, gi[g]] for g in keep])


corr_cols = CT.corr_cols


GTF = {"lpc10": "mouse_atlas/gencode.vM25.primary_chr.annotation.gtf", "merfish": "mouse_atlas/gencode.vM25.primary_chr.annotation.gtf",
       "melanoma": "gencode.v21.annotation.gtf", "sprox": "gencode.v21.annotation.gtf", "hic": "gencode.v21.annotation.gtf"}


def gene_length(key, ntop=200):
    """gene body length vs cross gain (Spearman) and vs own-axis R2 (specificity); top ntop cross genes vs all"""
    from saturn.annotate import gene_lengths
    L = gene_lengths(f"{H}/{GTF[key]}")
    G = pd.read_csv(f"{OUT}/{key}_genes.csv"); G["length"] = G.gene.map(L); G = G.dropna(subset=["length"])
    G[["gene", "length", "dR2_cross", "R2_own", "q"]].to_csv(f"{OUT}/{key}_length.csv", index=False)
    S = json.load(open(f"{OUT}/{key}_summary.json"))
    S["length"] = CT.gene_length_test(G, L, ntop)
    json.dump(S, open(f"{OUT}/{key}_summary.json", "w"), indent=1, default=float); print(key, S["length"], flush=True)


rotate = CT.rotate


def rate_null(key, nrot=200):
    """signed test: RNA's cross gain vs log(|grad rho_X| / |grad rho_RNA|) (X steeper), and X's cross gain vs the
    reverse; Spearman over spots, two-sided p against nrot rotations (saturn.crossterm.steepness_test)"""
    T = pd.read_csv(f"{OUT}/{key}_spots.csv", index_col=0)
    S = json.load(open(f"{OUT}/{key}_summary.json")); S.update(CT.steepness_test(T, nrot=nrot))
    json.dump(S, open(f"{OUT}/{key}_summary.json", "w"), indent=1, default=float)
    print(key, {k: S[k] for k in ("steeper_RNA", "steeper_X")}, flush=True)


def run(key):
    c = FITS[key]; os.makedirs(OUT, exist_ok=True)
    z = dict(np.load(f"{R}/targets_{c['ds']}.npz", allow_pickle=True)); bc = [str(b) for b in z["barcodes"]]
    mods = [str(m) for m in z["modalities"]]; K = [z[f"Y_{m}"].shape[1] for m in mods]
    iR, iX = mods.index("RNA"), mods.index(c["X"])
    model = saturn.load_model(f"{R}/{c['fit']}/saturn_best.pt", K)
    D = saturn.decompose(model, z["coords_z"])[0]; fa = c["fig"](bc)       # orientation as in the figures
    sgn = [np.sign(np.corrcoef(D[:, i], fa[:, k])[0, 1]) for k, i in enumerate((iR, iX))]
    # gradient by the chain rule through zscore_coords; PER_AXIS (--per-axis) = the pre-2026-10-22 run that divided by
    # the per-axis coordinate std (see saturn docs)
    T, S0 = CT.spot_decomposition(model, z["coords_z"], z["coords"], [z[f"Y_{m}"] for m in mods], iR, iX, sgn,
                                  knn=KNN, per_axis_sd=PER_AXIS)
    T.index = bc
    S = dict(key=key, dataset=c["ds"], X=c["X"], coupling=S0["coupling"], R2_RNA=S0["R2_RNA"], R2_X=S0["R2_X"])
    for nm in ("RNA", "X"):
        S[f"spearman_misalign_gain_{nm}"] = S0[f"spearman_misalign_gain_{nm}"]
        S[f"spearman_ratediff_gain_{nm}"] = S0[f"spearman_ratediff_gain_{nm}"]
        dec = pd.qcut(T.rate_diff.rank(method="first"), 10, labels=False)
        S[f"gain_by_ratediff_decile_{nm}"] = T.groupby(dec)[f"gain_{nm}"].mean().round(4).tolist()
        dec = pd.qcut(T.misalign.rank(method="first"), 10, labels=False)
        S[f"gain_by_misalign_decile_{nm}"] = T.groupby(dec)[f"gain_{nm}"].mean().round(4).tolist()
    T.to_csv(f"{OUT}/{key}_spots.csv")
    # ---- genes ----------------------------------------------------------------
    X, genes = rna(key, bc)
    G, prof, genes = CT.gene_cross_test(X, genes, T.rho_RNA.values, T.rho_X.values, min_det=MIN_DET)
    G.to_csv(f"{OUT}/{key}_genes.csv", index=False)
    S.update(n_genes=int(len(G)), n_cross_fdr05=int((G.q < 0.05).sum()))
    np.savez_compressed(f"{OUT}/{key}_profiles.npz", genes=genes, prof=prof.astype(np.float32))
    # ---- cis windows (ATAC datasets) ------------------------------------------
    if c["X"] == "ATAC":
        sig = G.sort_index().q.values < 0.05                                   # in tested-gene order
        keep, W, Pg = cis_test(key, bc, genes, prof, sig)
        cis, S["cis"] = CT.cis_concordance(prof, genes, W, keep, T.rho_RNA.values, T.rho_X.values, ndraw=NDRAW)
        cis.to_csv(f"{OUT}/{key}_cis.csv", index=False)
    json.dump(S, open(f"{OUT}/{key}_summary.json", "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float), flush=True)


PER_AXIS = "--per-axis" in sys.argv


if __name__ == "__main__":
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FITS)
    for k in keys:
        if "--null-only" not in sys.argv: run(k)
        rate_null(k); gene_length(k)
