"""Per section (B1, D1): do Saturn's two isodepths carry the cell-cell communication (CCC) gradients that other methods' axes miss?

The test set is the held-out PLA layer (never in any method's input): the authors' per-spot Pearson residual of every
complex, rank-residualised on log RNA depth, mitochondrial share and log PLA depth. CCC complexes = the trans-cellular
ligand-receptor / adhesion / synapse pairs measurable with this 29-target panel (both probe orientations):
  LFA1:ICAM1, CD11a:ICAM1 (LFA-1-ICAM-1 adhesion) | ITGA4:VCAM1, CD29:VCAM1 (VLA-4-VCAM-1, B-FDC) |
  CD28:CD86 (T costimulation) | CD279:PDL2 (PD-1-PD-L2, Tfh-B) | CD4:HLADPRQ, CD3:HLADPRQ (T-cell : MHC-II synapse) |
  CD23:CD21 (FCER2-CR2) | CD21:CD35 (CR2-CR1, B-FDC immune-complex handover)
Reference sets: B-cell receptor cis complexes (IgM/IgD/CD19/CD20/CD21/CD81 among themselves) and every complex.

Predictors (one per method; each method's RNA axis + Protein axis, additive cubic splines on the rank of each axis):
  Saturn, GASTON (default), GASTON (global-scaled), MEFISTO, MISO, SCIGMA, SpatialGlue, PCA (non-spatial),
  DPT (our whole-tissue diffusion pseudotime, RNA + Protein), Domains (marker compartments, one-hot; the discrete
  alternative to a gradient).
Score = spatially blocked cross-validated R2 (10 k-means blocks, leave-one-block-out): the complex must be the SAME
function of the axis in a part of the tissue the fit never saw. A gradient transfers; a patchwork does not.
Significance = the same CV R2 for 200 joint rotations of the method's two fields (alignment destroyed, autocorrelation
kept); nominal p < 0.05 reported against the 5% chance rate.
-> results/real_saturn/sproxseq_<sec>_ccc[_k<smooth>][_loo<k>]/{cv_r2.csv, p_rot.csv, summary.csv, profiles_*.csv, plane.npz}
Usage: python scripts/sproxseq_ccc.py [--sec D1] [--nrot 200] [--smooth 7] [--loo 6]
Needs sproxseq_baselines.py <sec> first.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from sklearn.preprocessing import SplineTransformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sproxseq_B1_rna_axis import resid, tr

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"; G = f"{R}/sproxseq_gc_trajectory"
NBLOCK = 10
CCC = {"LFA-1:ICAM-1": [("LFA1", "ICAM1"), ("CD11a", "ICAM1")], "VLA-4:VCAM-1": [("ITGA4", "VCAM1"), ("CD29", "VCAM1")],
       "CD28:CD86": [("CD28", "CD86")], "PD-1:PD-L2": [("CD279", "PDL2")],
       "CD4:MHC-II": [("CD4", "HLADPRQ")], "CD3:MHC-II": [("CD3", "HLADPRQ")],
       "CD23:CD21": [("CD23", "CD21")], "CD21:CD35": [("CD21", "CD35")]}
BCR = ["IgM", "IgD", "CD19", "CD20", "CD21", "CD81"]
METHODS = ["Saturn", "GASTON (default)", "GASTON (global-scaled)", "MEFISTO", "MISO", "SCIGMA", "SpatialGlue", "PCA", "DPT",
           "Domains"]


def ccc_index(prod):
    lab = {}
    for k, pairs in CCC.items():
        for a, b in pairs:
            for p in (f"{a}:{b}", f"{b}:{a}"):
                if p in prod: lab[p] = k
    return lab


def spline(v, n_knots=5):
    return SplineTransformer(n_knots=n_knots, degree=3, include_bias=False).fit_transform(tr(v)[:, None])


def design(cols):
    return np.column_stack([np.ones(len(cols[0]))] + [spline(c) for c in cols])


def cv_r2(X, Y, blk):
    """leave-one-block-out R2 per column of Y (pooled out-of-block predictions vs the grand mean)."""
    pred = np.empty_like(Y)
    for b in np.unique(blk):
        te = blk == b; tr_ = ~te
        beta = np.linalg.lstsq(X[tr_], Y[tr_], rcond=None)[0]; pred[te] = X[te] @ beta
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def split_half(t, Y, blk, n=50, nb=10, seed=0):
    """profile reproducibility: per complex, Pearson between the binned profiles (nb quantile bins of the axis) from two
    disjoint halves of the tissue (random 5 vs 5 of the spatial blocks), averaged over n splits"""
    b = np.minimum((tr(t) * nb).astype(int), nb - 1); H = np.eye(nb)[b]; ub = np.unique(blk); rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h = np.isin(blk, rng.choice(ub, len(ub) // 2, replace=False))
        prof = [(H[m].T @ Y[m]) / H[m].sum(0).clip(1)[:, None] for m in (h, ~h)]
        a, c = [(p_ - p_.mean(0)) / p_.std(0).clip(1e-12) for p_ in prof]
        out.append((a * c).mean(0))
    return np.mean(out, 0)


def smooth(xy, Y, k):
    from scipy.spatial import cKDTree
    nn = cKDTree(xy).query(xy, k=k)[1]
    return Y[nn].mean(1)


def rotate(xy, F, rng):
    c = xy.mean(0); th = rng.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return griddata((xy - c) @ Rm.T + c, F, xy, method="nearest")


def spots(sec):
    """x, y, covariates, marker compartment and the oriented Saturn axes (t_RNA: epithelium = 0; t_Prot: T zone = 0)"""
    f = f"{R}/sproxseq_{sec}_rna_axis/spots.csv"
    if os.path.exists(f): return pd.read_csv(f, index_col=0)
    from sproxseq_gc_trajectory import rna
    E = pd.read_csv(f"{R}/sproxseq_{sec}_explain/spots.csv", index_col=0); bc = list(E.index)
    a = rna(sec, bc); cnt = a.X.tocsr(); tot = np.asarray(cnt.sum(1)).ravel()
    mt = np.asarray(cnt[:, a.var_names.str.startswith("MT-")].sum(1)).ravel() / tot.clip(1)
    d, dp, z = E.rho_RNA_dr.values, E.rho_Prot_dr.values, E.compartment.values.copy()
    ep = E["sig_epithelium"].values; z[(ep - ep.mean()) / ep.std() > 1.5] = "epithelium"   # as in sproxseq_B1_rna_axis
    d = d if np.corrcoef(rankdata(d), rankdata(E.dist_epi))[0, 1] > 0 else -d
    dp = dp if np.median(dp[z == "T"]) < np.median(dp[np.isin(z, ["NBC", "DZ", "LZ"])]) else -dp
    return pd.DataFrame(dict(x=E.x, y=E.y, t_RNA=tr(d), t_Prot=tr(dp), compartment=z, dist_epi=E.dist_epi,
                             log_depth=E.log_depth, mito=mt), index=bc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sec", default="B1"); ap.add_argument("--nrot", type=int, default=200)
    ap.add_argument("--loo", type=int, default=0, help="replace every axis by the mean of its k nearest OTHER spots: a "
                    "per-spot axis built from the spot's own protein reads shares Poisson noise with every complex at that spot")
    ap.add_argument("--smooth", type=int, default=0, help="average each complex over its k nearest spots first")
    A = ap.parse_args(); SEC = A.sec; out = f"{R}/sproxseq_{SEC}_ccc" + (f"_k{A.smooth}" if A.smooth else "") + (f"_loo{A.loo}" if A.loo else ""); os.makedirs(out, exist_ok=True)
    sp = spots(SEC); bc = list(sp.index)
    xy = sp[["x", "y"]].values.astype(float)
    B = pd.read_csv(f"{R}/sproxseq_{SEC}_baselines/axes.csv", index_col=0).loc[bc]
    D = pd.read_csv(f"{R}/sproxseq_dpt_whole/spots.csv", index_col=0); D = D[D.section == SEC].loc[bc]
    pla = np.load(f"{R}/sproxseq_{SEC}_p46_pla.npz", allow_pickle=True)
    pidx = {b: i for i, b in enumerate(pla["barcodes"].astype(str))}
    lpla = np.log1p(pla["counts"][[pidx[b] for b in bc]].sum(1).astype(float))
    Rz = np.load(f"{G}/residuals_{SEC}.npy"); prod = pd.read_csv(f"{G}/residual_products_{SEC}.csv").iloc[:, 0].values
    S0 = pd.read_csv(f"{G}/spots_{SEC}.csv", index_col=0); assert list(S0.index) == bc
    Y = resid(Rz.astype(float), [sp.log_depth.values, sp.mito.values, lpla])
    if A.smooth: Y = smooth(xy, Y, A.smooth)
    lab = ccc_index(set(prod)); is_ccc = np.array([p in lab for p in prod])
    is_bcr = np.array([(p.split(":")[0] in BCR) and (p.split(":")[1] in BCR) for p in prod])
    print(f"{len(prod)} complexes | CCC {is_ccc.sum()} ({sorted(set(lab.values()))}) | BCR cis {is_bcr.sum()}")
    fields = {m: np.column_stack([B[f"{m}_RNA"], B[f"{m}_Protein"]]) for m in METHODS[:-3] if f"{m}_RNA" in B}
    fields["PCA"] = np.column_stack([B.PCA_RNA, B.PCA_Protein])
    fields["DPT"] = np.column_stack([D.DPT_RNA_dr, D.DPT_Prot])
    if A.loo:
        from scipy.spatial import cKDTree
        nn = cKDTree(xy).query(xy, k=A.loo + 1)[1][:, 1:]
        fields = {m: F[nn].mean(1) for m, F in fields.items()}
    comp = sp.compartment.values; levels = sorted(set(comp))
    onehot = np.column_stack([np.ones(len(comp))] + [(comp == l).astype(float) for l in levels[1:]])
    blk = KMeans(NBLOCK, n_init=10, random_state=0).fit_predict((xy - xy.mean(0)) / xy.std(0))
    cv, pr, rng = {}, {}, np.random.default_rng(0)
    for m, F in fields.items():
        cv[m] = cv_r2(design([F[:, 0], F[:, 1]]), Y, blk)
        null = np.array([cv_r2(design(list(rotate(xy, F, rng).T)), Y, blk) for _ in range(A.nrot)])
        pr[m] = ((null >= cv[m][None]).sum(0) + 1) / (A.nrot + 1)
        for k, col in (("RNA", 0), ("Protein", 1)):
            cv[f"{m} [{k} axis]"] = cv_r2(design([F[:, col]]), Y, blk)
        print(f"{m:24s} CCC median CV R2 {np.median(cv[m][is_ccc]):+.3f} | sig {np.mean(pr[m][is_ccc] < 0.05):.0%} "
              f"| all {np.median(cv[m]):+.3f} sig {np.mean(pr[m] < 0.05):.0%}", flush=True)
    sh = {f"{m} [{k} axis]": split_half(F[:, c], Y, blk) for m, F in fields.items() for k, c in (("RNA", 0), ("Protein", 1))}
    Sh = pd.DataFrame(sh, index=prod); Sh.insert(0, "ccc", [lab.get(p, "") for p in prod]); Sh.to_csv(f"{out}/split_half.csv")
    print("\nsplit-half profile reproducibility, median (CCC | all):")
    for m in sh: print(f"  {m:40s} {np.median(Sh[m][is_ccc]):+.3f} | {np.median(Sh[m]):+.3f}")
    # unique information: CV R2 gain of each method's pair over a composition baseline (PCA pair + DPT pair)
    base_cols = [fields["PCA"][:, 0], fields["PCA"][:, 1], fields["DPT"][:, 0], fields["DPT"][:, 1]]
    r_base = cv_r2(design(base_cols), Y, blk); inc = {}
    for m, F in fields.items():
        if m in ("PCA", "DPT"): continue
        inc[m] = cv_r2(design(base_cols + [F[:, 0], F[:, 1]]), Y, blk) - r_base
    inc["Domains"] = cv_r2(np.column_stack([design(base_cols), onehot[:, 1:]]), Y, blk) - r_base
    I = pd.DataFrame(inc, index=prod); I.insert(0, "ccc", [lab.get(p, "") for p in prod]); I.to_csv(f"{out}/gain_over_composition.csv")
    print("\nCV R2 gain over PCA+DPT, median / mean (CCC | all) | complexes gaining > 0.01:")
    for m in inc:
        print(f"  {m:24s} {np.median(I[m][is_ccc]):+.4f} / {I[m][is_ccc].mean():+.4f} | {np.median(I[m]):+.4f} / "
              f"{I[m].mean():+.4f} | {(I[m] > 0.01).sum()}")
    cv["Domains"] = cv_r2(onehot, Y, blk)
    cv["x,y (thin spline)"] = cv_r2(design([xy[:, 0], xy[:, 1]]), Y, blk)
    C = pd.DataFrame(cv, index=prod); C.insert(0, "ccc", [lab.get(p, "") for p in prod]); C.insert(1, "bcr", is_bcr)
    C.to_csv(f"{out}/cv_r2.csv"); P = pd.DataFrame(pr, index=prod); P.to_csv(f"{out}/p_rot.csv")
    rows = []
    for m in [k for k in C.columns[2:]]:
        for s, msk in (("CCC", is_ccc), ("BCR cis", is_bcr), ("all", np.ones(len(prod), bool))):
            v = C[m].values[msk]; row = dict(method=m, set=s, n=int(msk.sum()), median=np.median(v),
                                             mean=v.mean(), frac_pos=np.mean(v > 0))
            if m in P: row["frac_sig"] = np.mean(P[m].values[msk] < 0.05)
            if m != "Saturn": row["p_saturn_better"] = wilcoxon(C.Saturn.values[msk], v, alternative="greater").pvalue
            rows.append(row)
    Sm = pd.DataFrame(rows); Sm.to_csv(f"{out}/summary.csv", index=False)
    pd.set_option("display.width", 250); print(Sm[Sm.set != "BCR cis"].round(4).to_string(index=False))
    # per CCC complex, every method
    print("\nper CCC complex CV R2 (p_rot):")
    T = C[is_ccc].copy()
    for m in fields: T[m] = [f"{a:+.3f}{'*' if b < 0.05 else ' '}" for a, b in zip(C[m][is_ccc], P[m][is_ccc])]
    print(T[["ccc"] + list(fields)].sort_values("ccc").to_string())
    # Saturn biology: profiles of CCC complexes along each oriented axis + the (t_RNA, t_Prot) plane
    t_r, t_p = sp.t_RNA.values, sp.t_Prot.values
    for k, t in (("RNA", t_r), ("Protein", t_p)):
        pd.DataFrame(Y[:, is_ccc], columns=prod[is_ccc]).groupby(np.minimum((t * 10).astype(int), 9)).mean() \
            .to_csv(f"{out}/profiles_saturn_{k}.csv")
    nb = 6; ir, ip = np.minimum((t_r * nb).astype(int), nb - 1), np.minimum((t_p * nb).astype(int), nb - 1)
    plane = np.full((is_ccc.sum(), nb, nb), np.nan); cnt = np.zeros((nb, nb), int)
    for i in range(nb):
        for j in range(nb):
            m_ = (ir == i) & (ip == j); cnt[i, j] = m_.sum()
            if m_.sum() >= 15: plane[:, i, j] = Y[m_][:, is_ccc].mean(0)
    np.savez_compressed(f"{out}/plane.npz", plane=plane, count=cnt, complexes=prod[is_ccc],
                        comp_med=pd.DataFrame(dict(c=comp, r=t_r, p=t_p)).groupby("c").median().reset_index().values)
    print("->", out)


if __name__ == "__main__":
    main()
