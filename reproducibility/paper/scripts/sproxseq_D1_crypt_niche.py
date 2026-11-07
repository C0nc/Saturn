"""D1 (main) + B1 (replication): the held-out protein-complex layer re-organises along Saturn's rho_RNA, the mucosal
crypt-surface -> lymphoid-interior axis.

Specificity of a complex a:b in a set of spots = log2((O+1)/(E+1)) minus the same quantity for each partner's
mouse-IgG isotype-control products (a:mIgG*, mIgG*:a, averaged with b's), all pooled over the set. O = PLA reads of a:b,
E = per-spot independence expectation row_a * col_b / total (the authors' expected count). The isotype term removes
non-specific ligation, which rises wherever a partner is rare -- without it, T-cell complexes look "enriched" in
B-cell regions (CD3:mIgG does the same).

  1 axis profile   specificity in 8 quantile bins of t_RNA (crypt surface = 0); slope vs bin centre; spatial null =
                   200 rotations of the rho_RNA field (alignment destroyed, autocorrelation kept), two-sided
  2 replication    complexes whose slope has the same sign and rotation p < 0.05 in both sections
  3 geometry       method-free check: the same specificity by distance to the crypt epithelium (spot steps)
  4 programmes     module scores per t_RNA bin (what each end of the axis is)
-> results/real_saturn/sproxseq_crypt_niche/{complexes.csv, profiles_<sec>.csv, geometry_<sec>.csv, modules_<sec>.csv}
Usage: python scripts/sproxseq_D1_crypt_niche.py
"""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sproxseq_ccc import spots, rotate, R
from sproxseq_gc_trajectory import rna
from sproxseq_B1_rna_axis import MOD, resid

OUT = f"{R}/sproxseq_crypt_niche"; ISO = ["mIgG1", "mIgG2a", "mIgG2b"]; CTRL = ISO + ["HIgG"]
NB, NROT, MIN_READS = 8, 200, 2000
GEO = [("epithelium", None), ("< 2", (0, 2)), ("2-4", (2, 4)), ("4-7", (4, 7)), ("> 7", (7, 1e9))]


def layers(sec, bc):
    P = np.load(f"{R}/sproxseq_{sec}_p46_pla.npz", allow_pickle=True)
    pi = {b: i for i, b in enumerate(P["barcodes"].astype(str))}
    C = P["counts"][[pi[b] for b in bc]].astype(float); prod = P["products"].astype(str)
    A = np.array([p.split(":")[0] for p in prod]); B = np.array([p.split(":")[1] for p in prod])
    rows = {t: C[:, A == t].sum(1) for t in set(A)}; cols = {t: C[:, B == t].sum(1) for t in set(B)}
    tot = C.sum(1).clip(1)
    pairs = sorted({(a, b) for a, b in zip(A, B) if a not in CTRL and b not in CTRL})
    O = np.column_stack([C[:, (A == a) & (B == b)].sum(1) for a, b in pairs])
    E = np.column_stack([rows[a] * cols[b] / tot for a, b in pairs])
    tg = sorted({x for p in pairs for x in p})
    io, ie = {}, {}
    for x in tg:
        ps = {(a, b) for a, b in zip(A, B) if (a == x and b in ISO) or (b == x and a in ISO)}
        io[x] = sum(C[:, (A == a) & (B == b)].sum(1) for a, b in ps); ie[x] = sum(rows[a] * cols[b] / tot for a, b in ps)
    keep = O.sum(0) >= MIN_READS
    pairs = [p for p, k in zip(pairs, keep) if k]
    return pairs, O[:, keep], E[:, keep], np.column_stack([io[x] for x in tg]), np.column_stack([ie[x] for x in tg]), tg


def spec(H, pairs, O, E, IO, IE, tg):
    """H: (n_spots, n_groups) 0/1 membership -> (n_groups, n_complexes) isotype-corrected log2 O/E"""
    lo = np.log2((H.T @ O + 1) / (H.T @ E + 1)); li = np.log2((H.T @ IO + 1) / (H.T @ IE + 1))
    ia = [tg.index(a) for a, _ in pairs]; ib = [tg.index(b) for _, b in pairs]
    return lo - 0.5 * (li[:, ia] + li[:, ib])


def bins(t):
    return np.eye(NB)[np.minimum((np.argsort(np.argsort(t)) * NB) // len(t), NB - 1)]


def slope(S):
    c = np.arange(NB) - (NB - 1) / 2
    return (S * c[:, None]).sum(0) / (c ** 2).sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    res, nulls = {}, {}
    for sec in ("D1", "B1"):
        sp = spots(sec); bc = list(sp.index); xy = sp[["x", "y"]].values.astype(float); t = sp.t_RNA.values
        pairs, O, E, IO, IE, tg = layers(sec, bc); name = [f"{a}:{b}" for a, b in pairs]
        S = spec(bins(t), pairs, O, E, IO, IE, tg); obs = slope(S)
        rng = np.random.default_rng(0); null = np.array([slope(spec(bins(rotate(xy, t[:, None], rng)[:, 0]), pairs, O, E,
                                                                     IO, IE, tg)) for _ in range(NROT)])
        p = ((np.abs(null) >= np.abs(obs)[None]).sum(0) + 1) / (NROT + 1)
        pd.DataFrame(S, columns=name).to_csv(f"{OUT}/profiles_{sec}.csv")
        step = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]); de = sp.dist_epi.values / step
        epi = sp.compartment.values == "epithelium"
        Hg = np.column_stack([epi if r is None else (~epi) & (de >= r[0]) & (de < r[1]) for _, r in GEO]).astype(float)
        G_ = pd.DataFrame(spec(Hg, pairs, O, E, IO, IE, tg), columns=name, index=[g for g, _ in GEO])
        G_.insert(0, "n_spots", Hg.sum(0).astype(int)); G_.to_csv(f"{OUT}/geometry_{sec}.csv")
        a = rna(sec, bc); a.var_names_make_unique(); ld = np.log1p(np.asarray(a.X.sum(1)).ravel())
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        Mm = pd.DataFrame({k: resid(np.asarray(a[:, [g for g in v if g in a.var_names]].X.todense()), [ld]).mean(1)
                           for k, v in MOD.items() if v})
        Mm.groupby(np.argmax(bins(t), 1)).mean().to_csv(f"{OUT}/modules_{sec}.csv")
        res[sec] = pd.DataFrame(dict(slope=obs, p=p, reads=O.sum(0)), index=name)
        nulls[sec] = pd.DataFrame(null, columns=name)
        print(f"{sec}: {len(name)} complexes | slope p<0.05 {np.mean(p < 0.05):.1%} (chance 5%) | rho_RNA vs distance to "
              f"epithelium r {np.corrcoef(np.argsort(np.argsort(t)), np.argsort(np.argsort(de)))[0, 1]:.2f}", flush=True)
    k = res["D1"].index.intersection(res["B1"].index)
    T = res["D1"].loc[k].join(res["B1"].loc[k], lsuffix="_D1", rsuffix="_B1")
    sd, sb = T.p_D1 < 0.05, T.p_B1 < 0.05; same = np.sign(T.slope_D1) == np.sign(T.slope_B1)
    rc = np.corrcoef(T.slope_D1, T.slope_B1)[0, 1]
    rn = np.array([np.corrcoef(T.slope_D1, nulls["B1"].loc[i, k])[0, 1] for i in range(NROT)])
    print(f"slope agreement across sections r {rc:.2f} | B1 rotated: median {np.median(rn):+.2f}, 95th pct "
          f"{np.percentile(rn, 95):+.2f}, p {((rn >= rc).sum() + 1) / (NROT + 1):.3f}")
    T["replicated"] = np.where(sd & sb & same, np.where(T.slope_D1 < 0, "surface", "interior"), "")
    T.to_csv(f"{OUT}/complexes.csv")
    exp = sd.sum() * sb.sum() / len(k) / 2
    print(f"replicated (p<0.05 both, same sign): {int((sd & sb & same).sum())} | expected by independence ~{exp:.1f} | "
          "")
    pd.set_option("display.width", 250)
    for g in ("surface", "interior"):
        s = T[T.replicated == g].sort_values("slope_D1", ascending=g == "surface")
        print(f"\n{g} ({len(s)}):", ", ".join(f"{c} {v:+.2f}/{w:+.2f}" for c, v, w in zip(s.index, s.slope_D1, s.slope_B1)))
    print("->", OUT)


if __name__ == "__main__":
    main()
