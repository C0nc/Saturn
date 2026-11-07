"""B1 rho_RNA (depth-regressed fit): epithelial surface -> interior. Same tests as sproxseq_B1_bt_axis.py, plus the
joint two-axis picture with rho_Protein.

Axis oriented surface = 0 (near crypt epithelium) -> interior = 1 (rank). Depth-controlled throughout (log RNA UMI
regressed from ranks); mitochondrial share also checked because the surface end carries MT- genes.
  1 modules     pre-declared tonsil programmes (below): profile per decile, monotone trend, mid-axis peak,
                each vs 1000 rotated axes
  2 genes       genome-wide monotone / mid-peak, 200 rotations; nominal p < 0.05 counts vs the 5% chance rate
  3 complexes   every held-out PLA product's Pearson residual: monotone / mid-peak, 200 rotations; repeated with
                mitochondrial share regressed too (does the complex signal survive the quality covariate?)
  4 joint       compartment medians in the (t_RNA, t_Protein) plane; each complex classed by which axis (and which
                shape) it follows; overlap vs independence
-> results/real_saturn/sproxseq_B1_rna_axis/
Usage: python scripts/sproxseq_B1_rna_axis.py
"""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import rankdata, spearmanr
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sproxseq_gc_trajectory import rna
from sproxseq_validate2 import rotated_fields
from sproxseq_B1_bt_axis import border_score, mono, R, G, SEC

OUT = f"{R}/sproxseq_{SEC}_rna_axis"; NROT, NROT_MOD, SEED = 200, 1000, 0
MOD = {"epithelium": ["KRT5", "KRT13", "KRT4", "KRT14", "SPRR3"], "CCL20 (epithelial chemokine)": ["CCL20"],
       "CCR6 (memory B)": ["CCR6"], "IgA plasma": ["IGHA1", "JCHAIN", "MZB1", "XBP1"],
       "myeloid": ["S100A8", "S100A9", "LYZ"], "naive mantle": ["IGHD", "FCER2", "TCL1A", "SELL"],
       "GC": ["BCL6", "RGS13", "SERPINA9", "MEF2B", "AICDA", "MKI67"], "T zone": ["CCL19", "CCL21", "CCR7", "CD3E"],
       "mito share": None}


def resid(M, covs):
    A = np.column_stack([np.ones(len(covs[0]))] + [rankdata(c) for c in covs])
    R_ = np.apply_along_axis(rankdata, 0, M) if M.ndim == 2 else rankdata(M)
    R_ = R_ - A @ np.linalg.lstsq(A, R_, rcond=None)[0]
    return (R_ - R_.mean(0)) / R_.std(0).clip(1e-9)


def tr(v):
    return (rankdata(v) - 1) / (len(v) - 1)


def test(Y, t, trs):
    ob, om = border_score(Y, t), mono(Y, t)
    nb = np.array([border_score(Y, q) for q in trs]); nm = np.array([mono(Y, q) for q in trs])
    n = len(trs)
    return ob, ((nb >= ob[None]).sum(0) + 1) / (n + 1), om, ((np.abs(nm) >= np.abs(om)[None]).sum(0) + 1) / (n + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = pd.read_csv(f"{G}/spots_{SEC}.csv", index_col=0); bc = list(S.index)
    ax = pd.read_csv(f"{R}/sproxseq_{SEC}_p46_dr/axes.csv", index_col=0).loc[bc]
    xy = np.c_[ax.x.values, ax.y.values].astype(float)
    z = S.zone.values.copy(); z[(z == "other") & (S.score_T.values > 0.3)] = "T"; z[z == "other"] = "EF"
    a = rna(SEC, bc); cnt = a.X.tocsr(); tot = np.asarray(cnt.sum(1)).ravel(); ld = np.log1p(tot)
    mt = np.asarray(cnt[:, a.var_names.str.startswith("MT-")].sum(1)).ravel() / tot.clip(1)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    epi_v = np.asarray(a[:, MOD["epithelium"]].X.todense()).mean(1).ravel(); epi_z = (epi_v - epi_v.mean()) / epi_v.std()
    epi = epi_z > 1.5; dep = cKDTree(xy[epi]).query(xy)[0]
    d = ax.d_RNA.values; d = d if spearmanr(d, dep)[0] > 0 else -d; t = tr(d)
    dp = ax.d_Protein.values; dp = dp if np.median(dp[z == "T"]) < np.median(dp[z == "NBC"]) else -dp; tp = tr(dp)
    comp = np.where(epi, "epithelium", z)
    print(f"t_RNA vs distance to epithelium r {spearmanr(t, dep)[0]:.2f} | vs log depth {spearmanr(t, ld)[0]:.2f} | "
          f"vs mito share {spearmanr(t, mt)[0]:.2f} | vs t_Protein {spearmanr(t, tp)[0]:.2f}")
    print("compartment medians (t_RNA, t_Prot):",
          pd.DataFrame(dict(c=comp, tR=t, tP=tp)).groupby("c").median().round(2).T.to_dict())
    # 1 modules
    gs = sorted({g for v in MOD.values() if v for g in v})
    Yg = pd.DataFrame(resid(np.asarray(a[:, gs].X.todense()), [ld]), columns=gs)
    M = pd.DataFrame({k: (Yg[v].mean(1) if v else resid(mt, [ld])) for k, v in MOD.items()})
    rot = rotated_fields(xy, d, NROT_MOD, np.random.default_rng(1)); trs_m = [tr(r) for r in rot]
    ob, pb, om, pm = test(M.values, t, trs_m)
    Mt = pd.DataFrame(dict(monotone=om, p_mono=pm, mid_peak=ob, p_mid=pb), index=M.columns)
    prof = M.groupby(np.minimum((t * 10).astype(int), 9)).mean()
    Mt.to_csv(f"{OUT}/modules.csv"); prof.to_csv(f"{OUT}/module_profiles.csv")
    pd.set_option("display.width", 250)
    print("\nmodules (monotone > 0 = rises toward the interior):\n", Mt.round(3).to_string())
    print("\nprofile surface (0) -> interior (9):\n", prof.T.round(2).to_string())
    # 2 genes
    trs = trs_m[:NROT]
    keep = np.asarray((a.X > 0).mean(0)).ravel() >= 0.03; names = a.var_names[keep]
    Y = resid(np.asarray(a.X[:, keep].todense()), [ld])
    ob, pb, om, pm = test(Y, t, trs)
    Gt = pd.DataFrame(dict(mid_peak=ob, p_mid=pb, monotone=om, p_mono=pm), index=names); Gt.to_csv(f"{OUT}/genes.csv")
    for k, p in (("monotone", "p_mono"), ("mid_peak", "p_mid")):
        s = Gt[Gt[p] < 0.05].sort_values(k)
        print(f"\ngenes {k} p<0.05: {len(s)}/{len(Gt)} ({len(s) / len(Gt):.1%}; chance 5%)")
        if k == "monotone":
            print(" surface:", ", ".join(f"{g} {v:+.2f}" for g, v in s[k][:40].items()))
            print(" interior:", ", ".join(f"{g} {v:+.2f}" for g, v in s[k][::-1][:40].items()))
        else: print(" mid:", ", ".join(f"{g} {v:+.2f}" for g, v in s[k][::-1][:40].items()))
    # 3 complexes (depth; depth + mito)
    Rz = np.load(f"{G}/residuals_{SEC}.npy"); prod = pd.read_csv(f"{G}/residual_products_{SEC}.csv").iloc[:, 0].values
    out = dict(complex=prod)
    for tag, covs in (("", [ld]), ("_mt", [ld, mt])):
        Yc = resid(Rz, covs); ob, pb, om, pm = test(Yc, t, trs)
        out.update({f"mid{tag}": ob, f"p_mid{tag}": pb, f"mono{tag}": om, f"p_mono{tag}": pm})
    # protein-axis results for the joint classification
    P = pd.read_csv(f"{R}/sproxseq_{SEC}_bt_axis/complexes.csv")
    assert list(P.complex) == list(prod)
    out.update(prot_mid=P.border.values, p_prot_mid=P.p_border.values, prot_mono=P.mono.values, p_prot_mono=P.p_mono.values)
    C = pd.DataFrame(out); C.to_csv(f"{OUT}/complexes.csv", index=False)
    for k, p in (("mono", "p_mono"), ("mid", "p_mid"), ("mono_mt", "p_mono_mt"), ("mid_mt", "p_mid_mt")):
        s = C[C[p] < 0.05].sort_values(k)
        print(f"\ncomplexes {k} p<0.05: {len(s)}/{len(C)} ({len(s) / len(C):.1%}; chance 5%)")
        if k.startswith("mono"):
            print(" surface:", ", ".join(f"{c} {v:+.2f}" for c, v in zip(s.complex[:25], s[k][:25])))
            print(" interior:", ", ".join(f"{c} {v:+.2f}" for c, v in zip(s.complex[::-1][:25], s[k][::-1][:25])))
        else: print(" mid:", ", ".join(f"{c} {v:+.2f}" for c, v in zip(s.complex[::-1][:25], s[k][::-1][:25])))
    # 4 joint classification
    cls = {"RNA surface->interior": C.p_mono < 0.05, "RNA mid": C.p_mid < 0.05,
           "Prot T->B": C.p_prot_mono < 0.05, "Prot mantle band": C.p_prot_mid < 0.05}
    Kt = pd.DataFrame({a_: {b_: int((cls[a_] & cls[b_]).sum()) for b_ in cls} for a_ in cls})
    print("\ncomplex overlap (diagonal = n; expected overlap by independence = n_a*n_b/849):\n", Kt.to_string())
    pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], t_RNA=t, t_Prot=tp, compartment=comp, dist_epi=dep, log_depth=ld,
                      mito=mt), index=bc).to_csv(f"{OUT}/spots.csv")


if __name__ == "__main__":
    main()
