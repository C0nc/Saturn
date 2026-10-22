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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/src")
from saturn.core import Saturn

H = "/mnt/HDD2B/Users/yancui"; R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/saturn_cross_term"
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
        Xg, genes, *_, codes = load_rtt_merfish(5)
        return sp.csr_matrix(Xg)[pd.Index(codes).get_indexer(pd.Index(bc))], genes
    from continuous_marker_panels import sprox_rna, mel_rna
    X, g = sprox_rna(bc, "B1") if key == "sprox" else mel_rna(bc)
    return sp.csr_matrix(X), np.asarray(g).astype(str)


def hinge(d):
    q = np.quantile(d, np.linspace(0.1, 0.9, 5)); z = (d - d.mean()) / d.std()
    return np.column_stack([z] + [np.maximum(d - k, 0) / d.std() for k in q])


def fit_split(Y, d_own, d_x):
    """Y (N x G). Returns R2 own-only, R2 own + X, partial-F p, and the B(d_x) part of the full fit (N x G)"""
    N = len(Y); Bo, Bx = hinge(d_own), hinge(d_x)
    A0 = np.c_[np.ones(N), Bo]; A1 = np.c_[A0, Bx]
    tss = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    c0 = np.linalg.lstsq(A0, Y, rcond=None)[0]; r0 = ((Y - A0 @ c0) ** 2).sum(0)
    c1 = np.linalg.lstsq(A1, Y, rcond=None)[0]; r1 = ((Y - A1 @ c1) ** 2).sum(0)
    q, df2 = Bx.shape[1], N - A1.shape[1]
    F = ((r0 - r1) / q) / (r1 / df2 + 1e-12); p = fdist.sf(F, q, df2)
    prof = Bx @ c1[A0.shape[1]:]
    return 1 - r0 / tss, 1 - r1 / tss, p, prof - prof.mean(0)


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.clip(q, 0, 1); return out


def logcpm(X, cols):
    """log1p CP10k of the selected genes, library size from all genes"""
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    return np.log1p(np.asarray(sp.csc_matrix(X)[:, np.where(cols)[0]].astype(float).todense()) / lib[:, None] * 1e4)


def cis_test(key, bc, genes, prof, sig):
    """+-50 kb accessibility of each significant gene vs its cross profile; control = random other gene's window"""
    from domain_annot import CFG, all_tss
    cfg = CFG[FITS[key]["ds"]]; A = cfg["atac"](); A.obs_names = A.obs_names.astype(str); A = A[pd.Index(bc)]
    Xa = sp.csc_matrix(A.X).astype(float); lib = np.asarray(Xa.sum(1)).ravel().clip(1)
    pk = pd.Series([str(v) for v in A.var_names]); pch = pk.str.split(":").str[0].values
    pmid = pk.str.split(":").str[1].str.split("-").apply(lambda s: (int(s[0]) + int(s[1])) / 2).values
    ch, tss, nm = all_tss(cfg["gtf"]); first = pd.Series(range(len(nm)), index=nm).groupby(level=0).first()
    win, keep = {}, []
    for g in genes[sig]:
        if g not in first.index: continue
        j = first[g]; idx = np.where((pch == ch[j]) & (np.abs(pmid - tss[j]) <= WIN))[0]
        if len(idx) and Xa[:, idx].sum() >= 50: win[g] = idx; keep.append(g)
    gi = {g: i for i, g in enumerate(genes)}
    W = np.column_stack([np.log1p(np.asarray(Xa[:, win[g]].sum(1)).ravel() / lib * 1e4) for g in keep])
    return keep, W, np.column_stack([prof[:, gi[g]] for g in keep])


def corr_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


GTF = {"lpc10": "mouse_atlas/gencode.vM25.primary_chr.annotation.gtf", "merfish": "mouse_atlas/gencode.vM25.primary_chr.annotation.gtf",
       "melanoma": "gencode.v21.annotation.gtf", "sprox": "gencode.v21.annotation.gtf", "hic": "gencode.v21.annotation.gtf"}


def gene_length(key, ntop=200):
    """gene body length vs cross gain (Spearman) and vs own-axis R2 (specificity); top ntop cross genes vs all"""
    import re
    from scipy.stats import spearmanr, mannwhitneyu
    L = {}
    for line in open(f"{H}/{GTF[key]}"):
        if line[0] == "#": continue
        p = line.split("\t", 9)
        if p[2] != "gene": continue
        m = re.search(r'gene_name "([^"]+)"', p[8])
        if m and m.group(1) not in L: L[m.group(1)] = int(p[4]) - int(p[3])
    G = pd.read_csv(f"{OUT}/{key}_genes.csv"); G["length"] = G.gene.map(L); G = G.dropna(subset=["length"])
    G[["gene", "length", "dR2_cross", "R2_own", "q"]].to_csv(f"{OUT}/{key}_length.csv", index=False)
    top = G.nlargest(ntop, "dR2_cross").length; lg = np.log10(G.length)
    S = json.load(open(f"{OUT}/{key}_summary.json"))
    S["length"] = dict(rho_cross=float(spearmanr(lg, G.dR2_cross).statistic), rho_own=float(spearmanr(lg, G.R2_own).statistic),
                       median_top_kb=float(top.median() / 1e3), median_all_kb=float(G.length.median() / 1e3),
                       p=float(mannwhitneyu(top, G.length).pvalue))
    json.dump(S, open(f"{OUT}/{key}_summary.json", "w"), indent=1, default=float); print(key, S["length"], flush=True)


def rotate(xy, v, rng):
    c = xy.mean(0); th = rng.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return v[cKDTree((xy - c) @ Rm.T + c).query(xy, k=1)[1]]


def rate_null(key, nrot=200):
    """signed test: RNA's cross gain vs log(|grad rho_X| / |grad rho_RNA|) (X steeper), and X's cross gain vs the
    reverse; Spearman over spots, two-sided p against nrot rotations of the gain field"""
    from scipy.stats import spearmanr
    T = pd.read_csv(f"{OUT}/{key}_spots.csv", index_col=0); xy = T[["x", "y"]].values
    s = np.log((T.grad_X / T.grad_X.median()) / (T.grad_RNA / T.grad_RNA.median())).values
    S = json.load(open(f"{OUT}/{key}_summary.json")); rng = np.random.default_rng(0)
    for nm, lead in (("RNA", s), ("X", -s)):
        g = T[f"gain_{nm}"].values; obs = float(spearmanr(lead, g).statistic)
        null = np.array([spearmanr(lead, rotate(xy, g, rng)).statistic for _ in range(nrot)])
        q = pd.qcut(pd.Series(lead).rank(method="first"), 5, labels=False)
        S[f"steeper_{nm}"] = dict(rho=obs, p_rot=float((1 + (np.abs(null) >= abs(obs)).sum()) / (nrot + 1)),
                                  null_sd=float(null.std()), gain_by_quintile=pd.Series(g).groupby(q).mean().round(4).tolist())
    json.dump(S, open(f"{OUT}/{key}_summary.json", "w"), indent=1, default=float)
    print(key, {k: S[k] for k in ("steeper_RNA", "steeper_X")}, flush=True)


def run(key):
    c = FITS[key]; os.makedirs(OUT, exist_ok=True)
    z = dict(np.load(f"{R}/targets_{c['ds']}.npz", allow_pickle=True)); bc = [str(b) for b in z["barcodes"]]
    mods = [str(m) for m in z["modalities"]]; K = [z[f"Y_{m}"].shape[1] for m in mods]
    iR, iX = mods.index("RNA"), mods.index(c["X"])
    model = Saturn(K); model.load_state_dict(torch.load(f"{R}/{c['fit']}/saturn_best.pt", map_location="cpu")); model.eval()
    xy = torch.tensor(np.asarray(z["coords_z"], np.float32), requires_grad=True)
    d, full, own, cross = model(xy)
    raw = np.asarray(z["coords"], float); sdv = raw.std(0); nn_ = cKDTree(raw).query(raw, k=KNN)[1]
    # orientation as in the figures
    fa = c["fig"](bc); D = d.detach().numpy().copy(); sgn = np.sign([np.corrcoef(D[:, i], fa[:, k])[0, 1] for k, i in enumerate((iR, iX))])
    sg = {iR: sgn[0], iX: sgn[1]}
    grads = {}
    for i in (iR, iX):
        g = torch.autograd.grad(d[:, i].sum(), xy, retain_graph=True)[0].numpy() / sdv * sg[i]
        grads[i] = g[nn_].mean(1)
    u = {i: grads[i] / (np.linalg.norm(grads[i], axis=1, keepdims=True) + 1e-12) for i in grads}
    cosv = (u[iR] * u[iX]).sum(1); mis = 1 - np.abs(cosv)
    # rate difference: |log| ratio of the two gradient magnitudes, each relative to its own median
    gn = {i: np.linalg.norm(grads[i], axis=1) for i in grads}
    rate = np.abs(np.log((gn[iX] / np.median(gn[iX]) + 1e-6) / (gn[iR] / np.median(gn[iR]) + 1e-6)))
    out = dict(x=raw[:, 0], y=raw[:, 1], rho_RNA=D[:, iR] * sg[iR], rho_X=D[:, iX] * sg[iX], cos=cosv, misalign=mis,
               grad_RNA=gn[iR], grad_X=gn[iX], rate_diff=rate)
    S = dict(key=key, dataset=c["ds"], X=c["X"], coupling=np.round(model.coupling(xy.detach()), 3).tolist())
    for i, nm in ((iR, "RNA"), (iX, "X")):
        Y = np.asarray(z[f"Y_{mods[i]}"], float); F = full[i].detach().numpy(); C = cross[i].detach().numpy(); O = own[i].detach().numpy()
        Po = F - C + C.mean(0); den = ((Y - Y.mean(0)) ** 2).sum(1).mean()
        gain = (((Y - Po) ** 2).sum(1) - ((Y - F) ** 2).sum(1)) / den
        out[f"gain_{nm}"] = gain[nn_].mean(1)
        vo, vc = ((O - O.mean(0)) ** 2).sum(1), ((C - C.mean(0)) ** 2).sum(1)
        out[f"crossshare_{nm}"] = (vc / (vo + vc + 1e-12))[nn_].mean(1)
        tss = ((Y - Y.mean(0)) ** 2).sum()
        S[f"R2_{nm}"] = dict(full=1 - ((Y - F) ** 2).sum() / tss, own_only=1 - ((Y - Po) ** 2).sum() / tss)
    T = pd.DataFrame(out, index=bc)
    from scipy.stats import spearmanr
    for nm in ("RNA", "X"):
        S[f"spearman_misalign_gain_{nm}"] = float(spearmanr(T.misalign, T[f"gain_{nm}"]).statistic)
        S[f"spearman_ratediff_gain_{nm}"] = float(spearmanr(T.rate_diff, T[f"gain_{nm}"]).statistic)
        dec = pd.qcut(T.rate_diff.rank(method="first"), 10, labels=False)
        S[f"gain_by_ratediff_decile_{nm}"] = T.groupby(dec)[f"gain_{nm}"].mean().round(4).tolist()
        dec = pd.qcut(T.misalign.rank(method="first"), 10, labels=False)
        S[f"gain_by_misalign_decile_{nm}"] = T.groupby(dec)[f"gain_{nm}"].mean().round(4).tolist()
    T.to_csv(f"{OUT}/{key}_spots.csv")
    # ---- genes ----------------------------------------------------------------
    X, genes = rna(key, bc)
    det = np.asarray((X > 0).mean(0)).ravel(); ok = det >= MIN_DET
    Lg = logcpm(X, ok); genes = genes[ok]
    r0, r1, p, prof = fit_split(Lg, T.rho_RNA.values, T.rho_X.values)
    q = bh(p); G = pd.DataFrame(dict(gene=genes, detect=det[ok], R2_own=r0, R2_own_plus_X=r1, dR2_cross=r1 - r0, p=p, q=q,
                                     cross_sd=prof.std(0), cross_r_rhoX=corr_cols(prof, np.repeat(T.rho_X.values[:, None], prof.shape[1], 1))))
    G.sort_values("dR2_cross", ascending=False).to_csv(f"{OUT}/{key}_genes.csv", index=False)
    S.update(n_genes=int(len(G)), n_cross_fdr05=int((q < 0.05).sum()))
    np.savez_compressed(f"{OUT}/{key}_profiles.npz", genes=genes, prof=prof.astype(np.float32))
    # ---- cis windows (ATAC datasets) ------------------------------------------
    if c["X"] == "ATAC":
        sig = (q < 0.05)
        keep, W, Pg = cis_test(key, bc, genes, prof, sig)
        _, _, _, Pw = fit_split(W, T.rho_RNA.values, T.rho_X.values)
        obs = corr_cols(Pg, Pw); rng = np.random.default_rng(0)
        ctrl = np.array([corr_cols(Pg, Pw[:, rng.permutation(len(keep))]).mean() for _ in range(NDRAW)])
        pd.DataFrame(dict(gene=keep, concordance=obs)).to_csv(f"{OUT}/{key}_cis.csv", index=False)
        S["cis"] = dict(n_genes=len(keep), mean_obs=float(obs.mean()), ctrl_mean=float(ctrl.mean()), ctrl_sd=float(ctrl.std()),
                        z=float((obs.mean() - ctrl.mean()) / ctrl.std()), p=float((1 + (ctrl >= obs.mean()).sum()) / (NDRAW + 1)),
                        frac_pos=float((obs > 0).mean()))
    json.dump(S, open(f"{OUT}/{key}_summary.json", "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float), flush=True)


if __name__ == "__main__":
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FITS)
    for k in keys:
        if "--null-only" not in sys.argv: run(k)
        rate_null(k); gene_length(k)
