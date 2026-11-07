"""Compare the SPATIAL GRADIENTS of a gene and its cis peak, per domain.

The slope comparison along rho is a scalar and its magnitude is confounded by depth
(the width/speed result collapsed 1.86 -> 1.10 once ATAC was depth-matched). The spatial
gradient is a VECTOR:

    grad f(x) = (df/dx, df/dy)     estimated by a local least-squares plane over the
                                   spatial kNN of each spot

and the quantity compared is the ANGLE between the gene's and the peak's gradient:

    cos_theta = <grad g, grad p> / (|grad g| . |grad p|)

which is invariant to scaling either field, so a depth difference between modalities
cannot create it. Noise still attenuates cos toward 0, but it does so for near and far
peaks alike, so the TSS-distance trend remains the honest internal calibration --
the same one that worked for slope-sign concordance.

Reported per (gene, cis peak, domain): mean cos_theta over the spots of that domain,
weighted by |grad g||grad p| so flat regions do not dominate with random angles.

Usage: python scripts/saturn_cis_gradient.py <dataset> [n_genes] [--downsample]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "<workspace>/saturn_repo/src")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_annot import CFG, R, all_tss
from saturn_cis_function_pairs import PANEL, smooth, WIN, MAXPK, KDOM, MINCNT, zz
KGRAD = 12


def spatial_gradients(V, xy, k=KGRAD, chunk=2000):
    """local least-squares plane per spot -> (n_spots, n_feat, 2) (saturn.gradient.spatial_gradients)"""
    from saturn.gradient import spatial_gradients as sg
    return sg(V, xy, k=k)


def run(ds, n_genes=1500, downsample=False, seed=0):
    OUT = f"{R}/{ds}_cis_pairs"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z_ = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    xy = np.c_[ax.x.values, ax.y.values]
    rna = cfg["rna"](); atac = cfg["atac"]()
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}; ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    Xr = sp.csr_matrix(rna.X)[[ri[b] for b in bc]].astype(np.float64)
    Xa = sp.csr_matrix(atac.X)[[ai[b] for b in bc]].astype(np.float64)
    gn = np.array([str(g) for g in rna.var_names]); pn = np.array([str(p) for p in atac.var_names])
    tg = np.asarray(Xr.sum(0)).ravel(); tp = np.asarray(Xa.sum(0)).ravel()
    if downsample:                     # ATAC is the DEEPER modality here -> bring it down
        rng = np.random.default_rng(seed)
        lr = np.asarray(Xr.sum(1)).ravel(); la = np.asarray(Xa.sum(1)).ravel()
        keep = np.clip(lr / np.maximum(la, 1), 0, 1)
        Xa = Xa.tocoo(); Xa.data = rng.binomial(Xa.data.astype(int), keep[Xa.row]).astype(np.float64)
        Xa = Xa.tocsr(); Xa.eliminate_zeros(); tp = np.asarray(Xa.sum(0)).ravel()
        print(f"   ATAC downsampled to median depth {np.median(np.asarray(Xa.sum(1)).ravel()):.0f}", flush=True)
    kg = list(np.argsort(-tg)[:n_genes]); gidx = {g: i for i, g in enumerate(gn)}
    kg = np.array(kg + [gidx[g] for g in PANEL.get(ds, [])
                        if g in gidx and tg[gidx[g]] >= MINCNT and gidx[g] not in set(kg)])
    ch, po, nm = all_tss(cfg["gtf"]); tss = {}
    for c_, p_, n_ in zip(ch, po, nm): tss.setdefault(n_, (c_, p_))
    pc = np.array([p.split(":")[0] for p in pn])
    pm = np.array([(lambda a, b: (float(a) + float(b)) / 2)(*p.split(":")[1].split("-")) for p in pn])
    pairs = []
    for gi in kg:
        g = gn[gi]
        if g not in tss: continue
        c_, t_ = tss[g]; s = np.where((pc == c_) & (np.abs(pm - t_) <= WIN) & (tp >= 50))[0]
        if len(s) == 0: continue
        if len(s) > MAXPK: s = s[np.argsort(-tp[s])[:MAXPK]]
        for s_ in s: pairs.append((gi, s_))
    gi_a = np.array([p[0] for p in pairs]); pk_a = np.array([p[1] for p in pairs])
    need = np.unique(pk_a)
    print(f"   {len(pairs)} cis pairs; gradients for {len(kg)} genes and {len(need)} peaks", flush=True)
    lr = np.maximum(np.asarray(Xr.sum(1)).ravel(), 1); la = np.maximum(np.asarray(Xa.sum(1)).ravel(), 1)
    G = zz(smooth(np.log1p(np.asarray(Xr[:, kg].todense(), np.float32) / lr[:, None] * 1e4), xy))
    P = zz(smooth(np.log1p(np.asarray(Xa[:, need].todense(), np.float32) / la[:, None] * 1e4), xy))
    gG = spatial_gradients(G, xy); gP = spatial_gradients(P, xy)
    print("   gradients done", flush=True)
    gpos = {v: i for i, v in enumerate(kg)}; ppos = {v: i for i, v in enumerate(need)}
    ga = np.array([gpos[v] for v in gi_a]); pb = np.array([ppos[v] for v in pk_a])
    rows = []
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values; lab = gaston_domains(d, z_[f"Y_{axis}"], L_=KDOM)
        for k in range(KDOM):
            m = np.where(lab == k)[0]
            if len(m) < 30: continue
            A = gG[m][:, ga]; B = gP[m][:, pb]                     # spots x pairs x 2
            dot = (A * B).sum(2); na = np.linalg.norm(A, axis=2); nb = np.linalg.norm(B, axis=2)
            w = na * nb
            cos = dot / np.maximum(w, 1e-12)
            mc = (cos * w).sum(0) / np.maximum(w.sum(0), 1e-12)     # magnitude-weighted mean cos
            rows.append(pd.DataFrame(dict(dataset=ds, axis=axis, domain=k,
                                          gene=gn[gi_a], peak=pn[pk_a],
                                          kb=(pm[pk_a] - np.array([tss[gn[i]][1] for i in gi_a])) / 1e3,
                                          cos_grad=np.round(mc, 4),
                                          mag_gene=np.round(na.mean(0), 5), mag_peak=np.round(nb.mean(0), 5))))
    T = pd.concat(rows, ignore_index=True)
    tag = "_ds" if downsample else ""
    T.to_csv(f"{OUT}/cis_gradient{tag}.csv", index=False)
    print(f"-> {OUT}/cis_gradient{tag}.csv ({len(T)} rows)")
    return T


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1500,
        downsample="--downsample" in sys.argv)
