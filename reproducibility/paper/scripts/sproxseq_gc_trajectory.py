"""Does the Saturn isodepth recover germinal-centre development, and does the held-out
protein-complex layer change along it the way it changes along GC development?

Sprox-seq human tonsil (Wang, Xia ... Tay, Cell 2026), sections A1/B1/D1, Saturn p46 fits
(RNA + 46-probe protein marginals as input; the 1,678 PLA products never in the loss).

GC development reference = the authors' own analysis (paper_code/A1_B1_D1_code6.3):
  diffusion pseudotime on follicle + GC spots, rooted in the non-GC follicle, which orders
  non-GC follicle -> dark zone -> light zone. Their Leiden zone labels are not deposited,
  so zones are called here from THEIR marker lists (A1_annotation.ipynb):
      DZ  CXCR4 AICDA MME FOXP1      LZ  CD83 LMO2 BCL2A1      naive/follicle  IGHD FCER2 TCL1A
  a spot is B-follicle if the B score (MS4A1 CD19 CD79A, z) > 0.3, then takes its best zone.
  DPT is re-run the authors' way (log-norm, 4000 HVG, scale, PCA, kNN, diffmap, dpt).

Tests, each against the 200-rotation spatial null (rotate the tissue, re-interpolate rho):
  (1) rho vs DPT inside follicle/GC spots, both axes, all sections
  (2) complex dynamics: for every PLA product (isotype controls dropped, >= 200 reads), the
      authors' Pearson residual (obs - exp)/sqrt(exp), exp = A-row x B-col / total per spot,
      i.e. interaction beyond the abundance of the two proteins. Spearman with rho and with
      DPT inside follicle/GC; agreement = Spearman across complexes of those two vectors.
      The agreement itself gets the rotation null (a rotated rho that happens to correlate
      with DPT would agree too).
  (0) whole tissue: the same residual vs rho over ALL spots, per complex, rotation null
      (the held-out complex prediction, recomputed on the p46 axes with the authors' statistic)
  (3) per-follicle replication: connected follicle/GC components (>= MIN_FOL spots); within
      each, Spearman(rho, DPT) -- is the relation development INSIDE each GC, or only
      differences BETWEEN follicles?

Rho sign is arbitrary; every rho is oriented to correlate positively with DPT before any
cross-section statement (reported).

Usage: python scripts/sproxseq_gc_trajectory.py
"""
import gzip, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy.io import mmread
from scipy.stats import spearmanr, rankdata
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sproxseq_prep import D, GSM_RNA
from sproxseq_validate2 import rotated_fields

H = "<workspace>"
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/sproxseq_gc_trajectory"
SECS, NROT, MIN_READS, MIN_FOL, SEED = ["A1", "B1", "D1"], 200, 200, 15, 0
ZONES = {"DZ": ["CXCR4", "AICDA", "MME", "FOXP1"], "LZ": ["CD83", "LMO2", "BCL2A1"],
         "NBC": ["IGHD", "FCER2", "TCL1A"], "B": ["MS4A1", "CD19", "CD79A"], "T": ["CD3E", "CD3D", "CD2"]}
CTRL = ("mIgG", "HIgG")
AUTHORS = ["CD19:CD21", "CD20:CD32", "CD21:CD35", "CD147:CD38", "ITGA4:VCAM1"]   # their Fig 4f / 6g


def rna(sec, bcs_keep):
    g = GSM_RNA[sec]
    bcs = [l.strip() for l in gzip.open(f"{D}/{g}_{sec}_barcodes.tsv.gz", "rt")]
    feat = pd.read_csv(f"{D}/{g}_{sec}_features.tsv.gz", sep="\t", header=None)
    X = sp.csr_matrix(mmread(f"{D}/{g}_{sec}_matrix.mtx.gz").T)
    a = ad.AnnData(X, obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=feat[1].astype(str).values))
    a.var_names_make_unique()
    return a[bcs_keep].copy()


def zones_dpt(sec):
    ax = pd.read_csv(f"{R}/sproxseq_{sec}_p46/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    a = rna(sec, bc)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    S = {}
    for k, gs in ZONES.items():
        gs = [g for g in gs if g in a.var_names]
        v = np.asarray(a[:, gs].X.todense()).mean(1).ravel(); S[k] = (v - v.mean()) / v.std()
    S = pd.DataFrame(S, index=bc)
    z = np.where(~(S.B > 0.3), "other", S[["DZ", "LZ", "NBC"]].idxmax(1))
    keep = np.isin(z, ["NBC", "DZ", "LZ"])
    b = a[keep].copy(); sc.pp.highly_variable_genes(b, n_top_genes=4000)
    b = b[:, b.var.highly_variable].copy(); sc.pp.scale(b); sc.tl.pca(b, random_state=SEED)
    sc.pp.neighbors(b, random_state=SEED); sc.tl.diffmap(b, random_state=SEED)
    zz = z[keep]; root = np.where(zz == "NBC")[0][np.argmax(S.NBC.values[keep][zz == "NBC"])]
    b.uns["iroot"] = int(root); sc.tl.dpt(b)
    return ax, z, S, b.obs.dpt_pseudotime.values, keep


def spear_mat(Y, x):
    ry = np.apply_along_axis(rankdata, 0, Y); rx = rankdata(x)
    ry = (ry - ry.mean(0)) / ry.std(0).clip(1e-9); rx = (rx - rx.mean()) / rx.std()
    return (ry * rx[:, None]).mean(0)


def canon(p):
    strip = lambda t: t.rsplit("_", 1)[0] if t.endswith(("_A", "_B")) else t
    return ":".join(sorted(strip(t) for t in p.split(":")))


def residuals(sec):
    P = dict(np.load(f"{R}/sproxseq_{sec}_p46_pla.npz", allow_pickle=True))
    C = P["counts"].astype(float); prod = P["products"].astype(str)
    A = np.array([p.split(":")[0] for p in prod]); B = np.array([p.split(":")[1] for p in prod])
    rows = {t: C[:, A == t].sum(1) for t in set(A)}; cols = {t: C[:, B == t].sum(1) for t in set(B)}
    tot = C.sum(1).clip(1)
    E = np.column_stack([rows[a] * cols[b] / tot for a, b in zip(A, B)])
    Rz = (C - E) / np.sqrt(E.clip(1e-9))
    ok = (C.sum(0) >= MIN_READS) & ~np.array([any(c in p for c in CTRL) for p in prod])
    return Rz[:, ok], prod[ok], np.array([canon(p) for p in prod[ok]]), C[:, ok]


def follicles(xy, keep):
    idx = np.where(keep)[0]
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]; step = np.median(nn)
    pairs = cKDTree(xy[idx]).query_pairs(1.6 * step, output_type="ndarray")
    G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
    _, lab = connected_components(G, directed=False)
    out = np.full(len(xy), -1); out[idx] = lab
    return out


def run():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summ, cx_rows, fol_rows, nulls, whole = [], [], [], {}, []
    for sec in SECS:
        ax, z, S, dpt, keep = zones_dpt(sec)
        xy = np.c_[ax.x.values, ax.y.values]
        Rz, prod, cn, Cnt = residuals(sec)
        Rk = Rz[keep]
        sd = spear_mat(Rk, dpt)
        fol = follicles(xy, keep)
        spot = pd.DataFrame(dict(zone=z, dpt=np.nan, follicle=fol, **{f"score_{k}": S[k].values for k in S}),
                            index=ax.index)
        spot.loc[keep, "dpt"] = dpt
        for m in ("RNA", "Protein"):
            d0 = ax[f"d_{m}"].values
            r = spearmanr(d0[keep], dpt)[0]; sgn = 1.0 if r >= 0 else -1.0
            d = sgn * d0
            rot = rotated_fields(xy, d, NROT, rng)
            sw = spear_mat(Rz, d); nw = np.array([spear_mat(Rz, v) for v in rot])
            pw = ((np.abs(nw) >= np.abs(sw)).sum(0) + 1) / (NROT + 1)
            whole.append(dict(section=sec, axis=m, n_complex=int(len(sw)), n_sig=int((pw < 0.05).sum()),
                              frac_sig=float((pw < 0.05).mean())))
            null_r = np.array([spearmanr(v[keep], dpt)[0] for v in rot])
            sr = spear_mat(Rk, d[keep])
            null_sr = np.array([spear_mat(Rk, v[keep]) for v in rot])
            p_cx = ((np.abs(null_sr) >= np.abs(sr)).sum(0) + 1) / (NROT + 1)
            agree = spearmanr(sr, sd)[0]
            null_agree = np.array([spearmanr(s_, sd)[0] for s_ in null_sr])
            sig = p_cx < 0.05
            nulls[f"{sec}|{m}"] = null_agree
            # per-follicle replication
            fr = []
            for f in np.unique(fol[fol >= 0]):
                mf = fol == f
                if mf.sum() < MIN_FOL or len(set(z[mf])) < 2: continue
                rr = spearmanr(d[mf], spot.dpt.values[mf])[0]
                fr.append(rr)
                fol_rows.append(dict(section=sec, axis=m, follicle=int(f), n=int(mf.sum()), r=rr,
                                     zones="/".join(sorted(set(z[mf])))))
            fr = np.array(fr)
            summ.append(dict(section=sec, axis=m, sign=sgn, r_dpt=abs(r),
                             p_rot_dpt=float(((np.abs(null_r) >= abs(r)).sum() + 1) / (NROT + 1)),
                             median_by_zone={k: float(np.median(d[z == k])) for k in ("NBC", "DZ", "LZ", "other")},
                             n_complex=int(len(sr)), n_complex_sig=int(sig.sum()),
                             same_sign_as_dpt=float((np.sign(sr[sig]) == np.sign(sd[sig])).mean()) if sig.any() else None,
                             agreement=float(agree), agreement_null_p=float(((null_agree >= agree).sum() + 1) / (NROT + 1)),
                             agreement_null_median=float(np.median(null_agree)),
                             n_follicles=int(len(fr)), follicles_pos=int((fr > 0).sum()),
                             follicle_median_r=float(np.median(fr)) if len(fr) else None))
            spot[f"rho_{m}"] = d
            for j in range(len(sr)):
                cx_rows.append(dict(section=sec, axis=m, product=prod[j], complex=cn[j], reads=int(Cnt[:, j].sum()),
                                    r_rho=float(sr[j]), p_rot=float(p_cx[j]), r_dpt=float(sd[j]),
                                    r_rho_whole=float(sw[j]), p_rot_whole=float(pw[j])))
            s = summ[-1]
            print(f"   whole tissue: {whole[-1]['n_sig']}/{whole[-1]['n_complex']} complexes track rho (rotation p<0.05)")
            print(f"{sec} rho_{m} (x{sgn:+.0f}): r(DPT) {s['r_dpt']:.2f} p {s['p_rot_dpt']:.3f} | complexes "
                  f"{s['n_complex_sig']}/{s['n_complex']} sig, same sign {s['same_sign_as_dpt']} | agreement "
                  f"{agree:+.2f} (null median {s['agreement_null_median']:+.2f}, p {s['agreement_null_p']:.3f}) | "
                  f"follicles {s['follicles_pos']}/{s['n_follicles']} positive, median r {s['follicle_median_r']}",
                  flush=True)
        spot.to_csv(f"{OUT}/spots_{sec}.csv")
        np.save(f"{OUT}/residuals_{sec}.npy", Rz.astype(np.float32))
        pd.Series(prod).to_csv(f"{OUT}/residual_products_{sec}.csv", index=False)
    pd.DataFrame(cx_rows).to_csv(f"{OUT}/complex_dynamics.csv", index=False)
    pd.DataFrame(fol_rows).to_csv(f"{OUT}/follicles.csv", index=False)
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    pd.DataFrame(whole).to_csv(f"{OUT}/whole_tissue.csv", index=False)
    np.savez(f"{OUT}/agreement_null.npz", **nulls)
    print(f"-> {OUT}")


if __name__ == "__main__":
    run()
