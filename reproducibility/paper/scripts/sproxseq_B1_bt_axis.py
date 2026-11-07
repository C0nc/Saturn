"""B1 rho_Protein (depth-regressed fit): beyond 'B cells vs T cells', what happens ALONG the T zone -> B follicle axis?

Axis oriented T zone = 0 -> B follicle = 1 (rank). Everything is depth-controlled (log RNA UMI regressed from ranks).
  1 guidance cues   chemokine / oxysterol / S1P genes that position B and T cells (CXCL13-CXCR5, CCL19/21-CCR7,
                    EBI2 = GPR183 + CH25H/CYP7B1/HSD3B7, S1PR1/2) and T-B help (CD40LG, ICOS, IL21, BCL6, PDCD1, CD69, CD83);
                    profile per ventile of the axis
  2 border genes    genome-wide: is a gene highest in the MIDDLE of the axis (the T-B interface) rather than at an end?
                    border score = depth-residual mean in the middle 40% minus the higher end-30% mean; spatial null =
                    the same score along NROT rotated axes (two-sided not needed: the statistic is one-directional);
                    BH q per gene
  3 geometry        T-B border spots (B-follicle spot with a T-zone spot within 1.6 spot steps, or vice versa): where do
                    they sit on the axis?
  4 complexes       every PLA product's Pearson residual (held out from the fit): monotone tracking (Spearman, depth-
                    controlled) and border peaking (same score as 2), each vs the rotation null
-> results/real_saturn/sproxseq_B1_bt_axis/
Usage: python scripts/sproxseq_B1_bt_axis.py
"""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sproxseq_gc_trajectory import rna
from sproxseq_validate2 import rotated_fields

R = "<workspace>/Hole/gaston_mix_mm/results/real_saturn"; G = f"{R}/sproxseq_gc_trajectory"
SEC, NROT, SEED = "B1", 200, 0; OUT = f"{R}/sproxseq_{SEC}_bt_axis"
CUES = ["CXCL13", "CXCR5", "CCL19", "CCL21", "CCR7", "GPR183", "CH25H", "CYP7B1", "HSD3B7", "S1PR1", "S1PR2", "P2RY8",
        "CD40LG", "ICOS", "IL21", "BCL6", "PDCD1", "CD69", "CD83", "CD40", "CD86", "MS4A1", "CD3E", "MKI67"]


def resid(M, ld):
    A = np.c_[np.ones(len(ld)), rankdata(ld)]
    R_ = np.apply_along_axis(rankdata, 0, M) if M.ndim == 2 else rankdata(M)
    R_ = R_ - A @ np.linalg.lstsq(A, R_, rcond=None)[0]
    return (R_ - R_.mean(0)) / R_.std(0).clip(1e-9)


def border_score(Y, t):
    mid = (t > 0.3) & (t < 0.7); lo = t <= 0.3; hi = t >= 0.7
    return Y[mid].mean(0) - np.maximum(Y[lo].mean(0), Y[hi].mean(0))


def mono(Y, t):
    tr = (rankdata(t) - rankdata(t).mean()); tr = tr / tr.std()
    return (Y * tr[:, None]).mean(0)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = pd.read_csv(f"{G}/spots_{SEC}.csv", index_col=0); bc = list(S.index)
    ax = pd.read_csv(f"{R}/sproxseq_{SEC}_p46_dr/axes.csv", index_col=0).loc[bc]
    xy = np.c_[ax.x.values, ax.y.values].astype(float); d = ax.d_Protein.values
    z = S.zone.values.copy(); z[(z == "other") & (S.score_T.values > 0.3)] = "T"; z[z == "other"] = "EF"
    if np.median(d[z == "T"]) > np.median(d[z == "NBC"]): d = -d
    t = (rankdata(d) - 1) / (len(d) - 1)
    a = rna(SEC, bc); ld = np.log1p(np.asarray(a.X.sum(1)).ravel())
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    keep = np.asarray((a.X > 0).mean(0)).ravel() >= 0.03
    cue = [g for g in CUES if g in a.var_names]; keep |= a.var_names.isin(cue)
    names = a.var_names[keep]; Y = resid(np.asarray(a.X[:, keep].todense()), ld)
    # 1 cue profiles
    vent = np.minimum((t * 20).astype(int), 19)
    P = pd.DataFrame(Y[:, [names.get_loc(g) for g in cue]], columns=cue).groupby(vent).mean()
    P.to_csv(f"{OUT}/cue_profiles.csv")
    # 2 border genes with rotation null
    rot = rotated_fields(xy, d, NROT, np.random.default_rng(SEED))
    trs = [(rankdata(r) - 1) / (len(r) - 1) for r in rot]
    obs_b, obs_m = border_score(Y, t), mono(Y, t)
    nb = np.array([border_score(Y, q) for q in trs]); nm = np.array([mono(Y, q) for q in trs])
    pb = ((nb >= obs_b[None]).sum(0) + 1) / (NROT + 1); pm = ((np.abs(nm) >= np.abs(obs_m)[None]).sum(0) + 1) / (NROT + 1)
    Gt = pd.DataFrame(dict(border=obs_b, p_border=pb, q_border=multipletests(pb, method="fdr_bh")[1],
                           mono=obs_m, p_mono=pm, q_mono=multipletests(pm, method="fdr_bh")[1]), index=names)
    Gt.to_csv(f"{OUT}/genes.csv")
    # 3 geometry: T-B border spots
    step = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]); Bf = np.isin(z, ["NBC", "DZ", "LZ"]); Tz = z == "T"
    nT = cKDTree(xy[Tz]).query(xy)[0] <= 1.6 * step; nB = cKDTree(xy[Bf]).query(xy)[0] <= 1.6 * step
    border = (Bf & nT) | (Tz & nB)
    geo = pd.DataFrame(dict(t=t, compartment=np.where(border, "T-B border", z)))
    print("median axis position:", geo.groupby("compartment").t.median().round(2).to_dict(),
          "| n border", int(border.sum()))
    # 4 complexes
    Rz = np.load(f"{G}/residuals_{SEC}.npy"); prod = pd.read_csv(f"{G}/residual_products_{SEC}.csv").iloc[:, 0].values
    Yc = resid(Rz, ld)
    cb, cm = border_score(Yc, t), mono(Yc, t)
    ncb = np.array([border_score(Yc, q) for q in trs]); ncm = np.array([mono(Yc, q) for q in trs])
    pcb = ((ncb >= cb[None]).sum(0) + 1) / (NROT + 1); pcm = ((np.abs(ncm) >= np.abs(cm)[None]).sum(0) + 1) / (NROT + 1)
    Ct = pd.DataFrame(dict(complex=prod, border=cb, p_border=pcb, q_border=multipletests(pcb, method="fdr_bh")[1],
                           mono=cm, p_mono=pcm, q_mono=multipletests(pcm, method="fdr_bh")[1]))
    Ct.to_csv(f"{OUT}/complexes.csv", index=False)
    pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], t=t, compartment=geo.compartment.values, log_depth=ld), index=bc).to_csv(f"{OUT}/spots.csv")
    pd.set_option("display.width", 250)
    print("\ncue profile (depth-residual z, 5 bins T->B):")
    print(P.groupby(P.index // 4).mean().T.round(2).to_string())
    print("\ncue monotone / border:\n", Gt.loc[cue].round(3).to_string())
    for k, col in (("border", "q_border"), ("mono", "q_mono")):
        sig = Gt[Gt[col] < 0.05].sort_values(k)
        print(f"\ngenes {k} q<0.05: {len(sig)}")
        if k == "border": print(", ".join(f"{g} {v:+.2f}" for g, v in sig[k][::-1][:40].items()))
        else:
            print("B end:", ", ".join(f"{g} {v:+.2f}" for g, v in sig[k][::-1][:30].items()))
            print("T end:", ", ".join(f"{g} {v:+.2f}" for g, v in sig[k][:30].items()))
    for k, col in (("border", "q_border"), ("mono", "q_mono")):
        sig = Ct[Ct[col] < 0.1].sort_values(k)
        print(f"\ncomplexes {k} q<0.1: {len(sig)} of {len(Ct)}")
        if k == "border": print(", ".join(f"{c} {v:+.2f}" for c, v in zip(sig.complex[::-1][:30], sig[k][::-1][:30])))
        else:
            print("B end:", ", ".join(f"{c} {v:+.2f}" for c, v in zip(sig.complex[::-1][:25], sig[k][::-1][:25])))
            print("T end:", ", ".join(f"{c} {v:+.2f}" for c, v in zip(sig.complex[:25], sig[k][:25])))


if __name__ == "__main__":
    main()
