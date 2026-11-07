"""Transition-state spots from GRADIENTS (melanoma RNA+ATAC, Fig 3 fit `melanoma_tissue`; no Mellon).

A transition state = a place where the cell state is actively changing ALONG the isodepth
gradient, gradually over several spots (a sharp one-spot jump is two populations meeting).

Per axis m (state space = that modality's Saturn input Y_m, GLM-PCA RNA / spectral ATAC):
  1. grad rho     local linear fit of rho_m on (x, y) over the KNB nearest spots -> g (per spot step)
  2. grad state   same fit for every state component -> B (d x 2)
     along  = |B g_hat|   state change per spot step in the rho direction
     across = |B g_perp|  state change per spot step perpendicular to it
     align  = along / (along + across)
  3. null         rho rotated about the tissue centre (re-read by nearest spot) NROT times; g_hat
                  recomputed; median align and the along share -> is state change aligned with rho?
  4. transition score T = along. Spots with T >= TOP quantile are 'active'; 6-neighbour connected
     components of active spots are zones. Zone width = range of rho in the zone / median |grad rho|
     of the zone = extent along the gradient in spot steps.
       sharp interface   width < WMIN spot steps
       gradual zone      width >= WMIN  -> transition-state spots
  5. each zone: flanking domains (Fig 3 names), size, width, and marker / CNV profile (z) against
     the rest of its domains; position relative to the rho breakpoints.

Outputs results/real_saturn/melanoma_tissue_gradient/{spots.csv, zones.csv, summary.json, gradient_transition.png}
Usage: python scripts/mel_gradient_transition.py
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig3")
sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")

H = "<workspace>"
OUT = f"{H}/Hole/gaston_mix_mm/results/real_saturn/melanoma_tissue_gradient"
MODS, KNB, NROT, TOP, WMIN, MINZ, SEED = ("RNA", "ATAC"), 12, 200, 0.9, 3.0, 5, 0
MARK = ["ann_mel_marker", "ann_immune_marker", "ann_stromal_marker", "ann_cnv_burden_v2",
        "mk_MITF", "mk_SOX10", "mk_PMEL", "mk_AXL", "mk_COL1A1", "mk_PTPRC", "mk_CD74"]


def local_grad(xy, V, nn):
    """per spot least-squares slope of V (n x d) on centred (x, y) over its neighbourhood -> (n, d, 2)"""
    P = xy[nn] - xy[nn].mean(1, keepdims=True)                       # n x k x 2
    Vc = V[nn] - V[nn].mean(1, keepdims=True)                        # n x k x d
    G = np.einsum("nki,nkj->nij", P, P) + 1e-9 * np.eye(2)            # n x 2 x 2
    C = np.einsum("nki,nkd->nid", P, Vc)                              # n x 2 x d
    return np.transpose(np.linalg.solve(G, C), (0, 2, 1))             # n x d x 2


def along_across(B, g):
    gh = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    gp = np.c_[-gh[:, 1], gh[:, 0]]
    return np.linalg.norm(np.einsum("ndi,ni->nd", B, gh), axis=1), np.linalg.norm(np.einsum("ndi,ni->nd", B, gp), axis=1)


def rotate(xy, v, rng):
    c = xy.mean(0); th = rng.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return v[cKDTree((xy - c) @ Rm.T + c).query(xy, k=1)[1]]


def run():
    import fig3_melanoma as F3
    from paper_style import DOMAIN
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(SEED)
    D = F3.load(); names = F3.label_domains(D); z = D["z"]
    xy0 = np.asarray(D["xy"], float)
    step = np.median(cKDTree(xy0).query(xy0, k=2)[0][:, 1])
    xy = xy0 / step                                                  # coordinates in spot steps (40 um)
    tree = cKDTree(xy); nn = tree.query(xy, k=KNB + 1)[1]; nn6 = tree.query(xy, k=7)[1][:, 1:]
    Zm = {c: (np.asarray(z[c], float) - np.mean(z[c])) / np.std(z[c]) for c in MARK if c in z}
    spots = pd.DataFrame(index=D["bc"]); spots["x"], spots["y"] = xy0[:, 0], xy0[:, 1]
    zones_all, summ = [], {}
    for m in MODS:
        d = np.asarray(D["d"][m], float); lab = np.asarray(D["lab"][m])
        X = np.asarray(z[f"Y_{m}"], float); X = X / X.std(0).mean()                   # one common scale
        g = local_grad(xy, d[:, None], nn)[:, 0, :]; B = local_grad(xy, X, nn)
        al, ac = along_across(B, g); align = al / (al + ac + 1e-12)
        nl_align, nl_share = [], []
        for _ in range(NROT):
            gr = local_grad(xy, rotate(xy, d, rng)[:, None], nn)[:, 0, :]
            a_, c_ = along_across(B, gr); nl_align.append(np.median(a_ / (a_ + c_ + 1e-12)))
        nl_align = np.array(nl_align)
        p_align = float(((nl_align >= np.median(align)).sum() + 1) / (NROT + 1))
        T = al; act = T >= np.quantile(T, TOP)
        r, c = np.where(act[:, None] & act[nn6])
        A = coo_matrix((np.ones(len(r)), (r, nn6[r, c])), shape=(len(d), len(d)))
        ncomp, comp = connected_components(A, directed=False)
        gn = np.linalg.norm(g, axis=1)
        trans = np.zeros(len(d), bool); zid = np.full(len(d), -1)
        order = sorted(set(lab), key=lambda k: d[lab == k].min())
        for q in range(ncomp):
            mq = (comp == q) & act
            if mq.sum() < MINZ: continue
            width = float(np.ptp(d[mq]) / max(np.median(gn[mq]), 1e-9))
            doms = pd.Series([names[m][k] for k in lab[mq]]).value_counts()
            kind = "gradual" if width >= WMIN else "sharp"
            prof = {c_: round(float(np.median(v[mq]) - np.median(v[np.isin(lab, lab[mq]) & ~mq])), 2) for c_, v in Zm.items()}
            zones_all.append(dict(axis=m, zone=q, n=int(mq.sum()), width_steps=round(width, 1), kind=kind,
                                  rho_lo=round(float(d[mq].min()), 3), rho_hi=round(float(d[mq].max()), 3),
                                  domains="; ".join(f"{k} {v}" for k, v in doms.items()), **prof))
            zid[mq] = q
            if kind == "gradual": trans[mq] = True
        spots[f"rho_{m}"] = d; spots[f"domain_{m}"] = [names[m][k] for k in lab]
        spots[f"grad_rho_{m}"] = gn; spots[f"along_{m}"] = al; spots[f"across_{m}"] = ac
        spots[f"align_{m}"] = align; spots[f"zone_{m}"] = zid; spots[f"transition_{m}"] = trans
        Zs = [zz for zz in zones_all if zz["axis"] == m]
        summ[m] = dict(median_align=round(float(np.median(align)), 3), null_align_median=round(float(np.median(nl_align)), 3),
                       p_align=p_align, n_active=int(act.sum()),
                       n_zones=len(Zs), n_gradual=sum(zz["kind"] == "gradual" for zz in Zs),
                       n_sharp=sum(zz["kind"] == "sharp" for zz in Zs), n_transition_spots=int(trans.sum()),
                       steep_vs_flat=dict(grad_rho_active=round(float(np.median(gn[act])), 3),
                                          grad_rho_rest=round(float(np.median(gn[~act])), 3)))
        print(f"\nrho_{m}: state change aligned with grad rho, median {np.median(align):.2f} vs rotated {np.median(nl_align):.2f} "
              f"(p {p_align:.3f}); active spots {act.sum()}, zones {len(Zs)} (gradual {summ[m]['n_gradual']}, sharp {summ[m]['n_sharp']}), "
              f"transition spots {trans.sum()}", flush=True)
        for zz in sorted(Zs, key=lambda q: -q["n"]):
            print(f"   zone {zz['zone']:3d} {zz['kind']:7s} n {zz['n']:3d} width {zz['width_steps']:5.1f} rho [{zz['rho_lo']:.2f},{zz['rho_hi']:.2f}] "
                  f"| {zz['domains']} | CNV {zz.get('ann_cnv_burden_v2', 0):+.2f} mel {zz.get('ann_mel_marker', 0):+.2f} "
                  f"imm {zz.get('ann_immune_marker', 0):+.2f} str {zz.get('ann_stromal_marker', 0):+.2f} AXL {zz.get('mk_AXL', 0):+.2f}")
    spots.to_csv(f"{OUT}/spots.csv"); pd.DataFrame(zones_all).to_csv(f"{OUT}/zones.csv", index=False)
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    fig, A = plt.subplots(2, 3, figsize=(15, 9.5))
    for i, m in enumerate(MODS):
        cols = np.array([DOMAIN.get(n, "#999999") for n in spots[f"domain_{m}"]])
        a = A[i, 0]; a.scatter(xy0[:, 0], xy0[:, 1], c=cols, s=4, lw=0); a.set_title(f"{m} domains")
        a = A[i, 1]; v = spots[f"along_{m}"]
        sc = a.scatter(xy0[:, 0], xy0[:, 1], c=v, s=4, lw=0, cmap="viridis", vmax=np.quantile(v, 0.98))
        plt.colorbar(sc, ax=a, shrink=0.6); a.set_title(f"state change per spot step along grad rho_{m}")
        a = A[i, 2]; a.scatter(xy0[:, 0], xy0[:, 1], c="#e6e6e6", s=4, lw=0)
        zz = spots[f"zone_{m}"].values; t = spots[f"transition_{m}"].values
        a.scatter(xy0[(zz >= 0) & ~t, 0], xy0[(zz >= 0) & ~t, 1], s=6, c="#9a9a9a", lw=0, label="sharp interface")
        a.scatter(xy0[t, 0], xy0[t, 1], s=7, c="#c4453c", lw=0, label="gradual transition zone")
        a.legend(loc="lower left", fontsize=8); a.set_title(f"rho_{m}: active zones")
        for a in A[i]: a.set_aspect("equal"); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/gradient_transition.png", dpi=110); print(f"-> {OUT}")


if __name__ == "__main__":
    run()
