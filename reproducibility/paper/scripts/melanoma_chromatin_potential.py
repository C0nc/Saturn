"""Melanoma: function-matched peaks and per-domain chromatin potential along each rho.

Fitted functions exactly as in saturn_cis_function_pairs.py: kNN-15 smoothed log CP10k,
z-scored, GASTON Gaussian piecewise fit per domain (LRT gate t = 0.1), evaluated on a
120-point rho grid and z-scored over the grid.

(1) FUNCTION MATCHING  for every process gene sloped on an axis, Pearson similarity of its
    curve with the curves of the top-N peaks genome-wide. Matches are labelled cis (same
    chromosome, <= 1 Mb from the TSS) or trans.

(2) CHROMATIN POTENTIAL  (SHARE-seq's definition, in rho units). Within domain k, at the
    domain midpoint:
        CP_k = (f_peak - f_gene) / f_gene'          (curves in z units, f' = slope over the domain)
    i.e. how far along rho the gene must move for its expression to reach the level its peak's
    accessibility already shows. If the peak curve is the gene curve shifted by D along rho,
    CP_k recovers D. Reported as a FRACTION of the domain's rho-width so domains compare.
    A lag between whole curves is not used per domain: inside one domain both fits are
    straight lines and any shift fits equally well (see saturn_cis_function_pairs.py).
    Pairs: process gene x cis peak, both sloped in domain k and sign-concordant.
    Controls: (a) the gene's top trans function-matched peaks, (b) random peaks, same domain,
    same gate. Whole-axis z-scoring can offset curves for reasons unrelated to a shift, which
    is exactly what the controls calibrate.

Usage: python scripts/melanoma_chromatin_potential.py [n_peaks]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from domain_annot import CFG, R, all_tss
from piecewise import fit_gauss
from saturn_cis_function_pairs import smooth, zz, curves, PANEL, KDOM, T_LRT, NGRID, WIN

DS = "melanoma_tissue"
TOPM, NRAND, MINCNT = 10, 20, 100


def main(n_peaks=30000):
    OUT = f"{R}/{DS}_chromatin_potential"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[DS]
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True))
    xy = np.c_[ax.x.values, ax.y.values]
    rna = cfg["rna"](); atac = cfg["atac"]()
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}; ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    Xr = sp.csr_matrix(rna.X)[[ri[b] for b in bc]]; Xa = sp.csr_matrix(atac.X)[[ai[b] for b in bc]]
    gn = np.array([str(g) for g in rna.var_names]); pn = np.array([str(p) for p in atac.var_names])
    tg = np.asarray(Xr.sum(0)).ravel().astype(float); tp = np.asarray(Xa.sum(0)).ravel().astype(float)
    gidx = {g: i for i, g in enumerate(gn)}
    genes = [g for g in PANEL[DS] if g in gidx and tg[gidx[g]] >= MINCNT]
    kp = np.argsort(-tp)[:n_peaks]
    print(f"{DS}: {len(genes)} process genes, {len(kp)} peaks (of {len(pn)})", flush=True)
    ch, po, nm = all_tss(cfg["gtf"]); tss = {}
    for c_, p_, n_ in zip(ch, po, nm): tss.setdefault(n_, (c_, p_))
    pc = np.array([p.split(":")[0] for p in pn[kp]])
    pm = np.array([(lambda a, b: (float(a) + float(b)) / 2)(*p.split(":")[1].split("-")) for p in pn[kp]])
    lib_r = np.maximum(np.asarray(Xr.sum(1)).ravel(), 1); lib_a = np.maximum(np.asarray(Xa.sum(1)).ravel(), 1)
    G = zz(smooth(np.log1p(np.asarray(Xr[:, [gidx[g] for g in genes]].todense(), np.float32) / lib_r[:, None] * 1e4), xy))
    P = zz(smooth(np.log1p(np.asarray(Xa[:, kp].todense(), np.float32) / lib_a[:, None] * 1e4), xy))
    rng = np.random.default_rng(0)

    match_rows, cp_rows = [], []
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values; lab = gaston_domains(d, z[f"Y_{axis}"], L_=kdom(DS, axis))
        grid = np.linspace(np.percentile(d, 1), np.percentile(d, 99), NGRID)
        kgrid = lab[np.argmin(np.abs(d[None, :] - grid[:, None]), axis=1)]
        sg, ig = fit_gauss(G, lab, d, t=T_LRT); spk, ipk = fit_gauss(P, lab, d, t=T_LRT)
        CG = curves(sg, ig, d, lab, grid); CPk = curves(spk, ipk, d, lab, grid)
        Az = (CG - CG.mean(0)) / np.maximum(CG.std(0), 1e-12)
        Bz = (CPk - CPk.mean(0)) / np.maximum(CPk.std(0), 1e-12)
        S = Az.T @ Bz / NGRID                                      # genes x peaks function similarity
        dom = {}
        for k in range(lab.max() + 1):
            gk = np.where(kgrid == k)[0]
            if len(gk) >= 3:
                dom[k] = (gk, grid[gk[-1]] - grid[gk[0]], gk[len(gk) // 2])

        def cp_value(a, b, k):
            gk, width, mid = dom[k]
            slope = (CG[gk[-1], a] - CG[gk[0], a]) / max(grid[gk[-1]] - grid[gk[0]], 1e-12)
            if abs(slope) < 1e-9: return np.nan
            return (CPk[mid, b] - CG[mid, a]) / slope / width

        for a, g in enumerate(genes):
            if not (np.abs(sg[a]) > 1e-12).any(): continue
            c_, t_ = tss.get(g, (None, None))
            cis_mask = (pc == c_) & (np.abs(pm - t_) <= WIN) if c_ is not None else np.zeros(len(kp), bool)
            order = np.argsort(-S[a])
            top = order[:TOPM]
            for rnk, b in enumerate(top):
                match_rows.append(dict(axis=axis, gene=g, rank=rnk + 1, peak=pn[kp[b]], similarity=round(float(S[a, b]), 4),
                                       relation="cis" if cis_mask[b] else ("same_chrom" if pc[b] == c_ else "trans"),
                                       kb=round((pm[b] - t_) / 1e3, 1) if pc[b] == c_ else np.nan))
            trans_top = [b for b in order if not cis_mask[b]][:TOPM]
            cis_idx = np.where(cis_mask)[0]
            for k in dom:
                if abs(sg[a][k]) <= 1e-12: continue
                groups = {"cis": [b for b in cis_idx if abs(spk[b][k]) > 1e-12 and np.sign(spk[b][k]) == np.sign(sg[a][k])],
                          "trans_matched": [b for b in trans_top if abs(spk[b][k]) > 1e-12],
                          "random": [b for b in rng.choice(len(kp), NRAND * 5, replace=False)
                                     if abs(spk[b][k]) > 1e-12 and np.sign(spk[b][k]) == np.sign(sg[a][k])][:NRAND]}
                for grp, bs in groups.items():
                    for b in bs:
                        v = cp_value(a, b, k)
                        if np.isfinite(v):
                            cp_rows.append(dict(axis=axis, domain=int(k), gene=g, peak=pn[kp[b]], group=grp,
                                                cp_frac=round(float(v), 4), similarity=round(float(S[a, b]), 4),
                                                kb=round((pm[b] - t_) / 1e3, 1) if grp == "cis" else np.nan))
        print(f"   d_{axis}: {sum(r['axis'] == axis for r in cp_rows)} CP values", flush=True)
    M = pd.DataFrame(match_rows); C = pd.DataFrame(cp_rows)
    M.to_csv(f"{OUT}/function_matched_peaks.csv", index=False); C.to_csv(f"{OUT}/chromatin_potential_pairs.csv", index=False)
    print(f"-> {OUT}")
    return M, C


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
