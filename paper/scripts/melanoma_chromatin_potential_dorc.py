"""Chromatin potential per domain per rho, SHARE-seq style: DORC first, then potential.

Single cis peaks gave no systematic chromatin potential (gene-level net CP ~0 in every domain;
melanoma_chromatin_potential.py) and unstable per-pair values. SHARE-seq (Ma et al., Cell 2020)
aggregates first:

  DORC(g)  peaks within +-50 kb of the TSS whose spot-level correlation with g exceeds the 95th
           percentile of g's correlation with 200 random peaks of similar total count; genes with
           >= MIN_PK such peaks are DORC genes; DORC score = log CP10k of the summed peaks.
  CP_k     fitted gene and DORC curves (kNN-15 smoothed, z-scored, GASTON Gaussian piecewise fit,
           120-point grid) -> CP_k = (f_DORC - f_gene) / f_gene' at the domain midpoint, as a
           fraction of the domain's rho-width. Only domains where the gene's fitted slope is at least
           MIN_SLOPE (z per rho-width) -- near-flat genes make CP explode.
  control  the same number of random peaks (matched by total count), summed, same CP; 50 sets.
  net CP   CP(DORC) - median CP(random sets), per gene; per domain: median over genes, bootstrap CI.
           Positive = the DORC already sits at the level the gene reaches further along +rho.

Usage: python scripts/melanoma_chromatin_potential_dorc.py
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from domain_annot import CFG, R, all_tss
from piecewise import fit_gauss
from saturn_cis_function_pairs import smooth, zz, curves, PANEL, KDOM, T_LRT, NGRID

DS, WIN50, MIN_PK, NRAND_COR, NSET, MIN_SLOPE = "melanoma_tissue", 50_000, 3, 200, 50, 0.5


def ranks(M):
    return np.argsort(np.argsort(M, axis=0), axis=0).astype(np.float32)


def main():
    OUT = f"{R}/{DS}_chromatin_potential"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[DS]
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True)); xy = np.c_[ax.x.values, ax.y.values]
    rna = cfg["rna"](); atac = cfg["atac"]()
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}; ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    Xr = sp.csr_matrix(rna.X)[[ri[b] for b in bc]].astype(np.float32)
    Xa = sp.csc_matrix(sp.csr_matrix(atac.X)[[ai[b] for b in bc]]).astype(np.float32)
    gn = np.array([str(g) for g in rna.var_names]); pn = np.array([str(p) for p in atac.var_names])
    lib_r = np.maximum(np.asarray(Xr.sum(1)).ravel(), 1); lib_a = np.maximum(np.asarray(Xa.sum(1)).ravel(), 1)
    tp = np.asarray(Xa.sum(0)).ravel()
    pc = np.array([p.split(":")[0] for p in pn])
    pm = np.array([(lambda a, b: (float(a) + float(b)) / 2)(*p.split(":")[1].split("-")) for p in pn])
    ch, po, nm = all_tss(cfg["gtf"]); tss = {}
    for c_, p_, n_ in zip(ch, po, nm): tss.setdefault(n_, (c_, p_))
    gidx = {g: i for i, g in enumerate(gn)}
    rng = np.random.default_rng(0)
    order_tp = np.argsort(tp); tp_sorted = tp[order_tp]

    def matched_random(k_tot, n, exclude):
        """n random peaks whose total counts bracket k_tot"""
        pos = np.searchsorted(tp_sorted, k_tot)
        win = order_tp[max(pos - 2000, 0):pos + 2000]
        win = win[~np.isin(win, exclude)]
        return rng.choice(win, n, replace=len(win) < n)

    dorc, genes, sizes = [], [], []
    for g in PANEL[DS]:
        if g not in gidx or g not in tss: continue
        c_, t_ = tss[g]
        cand = np.where((pc == c_) & (np.abs(pm - t_) <= WIN50) & (tp >= 20))[0]
        if len(cand) < MIN_PK: continue
        gv = np.log1p(np.asarray(Xr[:, gidx[g]].todense()).ravel() / lib_r * 1e4)
        rv = ranks(gv[:, None])[:, 0]; rv = (rv - rv.mean()) / rv.std()
        def rcor(idx):
            M = np.log1p(np.asarray(Xa[:, idx].todense()) / lib_a[:, None] * 1e4)
            Rm = ranks(M); Rm = (Rm - Rm.mean(0)) / np.maximum(Rm.std(0), 1e-9)
            return (Rm * rv[:, None]).mean(0)
        rc = rcor(cand)
        bg = rcor(np.concatenate([matched_random(np.median(tp[cand]), NRAND_COR // 2, cand),
                                  rng.choice(np.where(tp >= 20)[0], NRAND_COR // 2)]))
        sel = cand[rc > np.percentile(bg, 95)]
        if len(sel) < MIN_PK: continue
        dorc.append(np.log1p(np.asarray(Xa[:, sel].sum(1)).ravel() / lib_a * 1e4)); genes.append(g); sizes.append(sel)
    print(f"{len(genes)} DORC genes (>= {MIN_PK} correlated peaks within +-50 kb): {', '.join(genes)}", flush=True)
    Gm = np.column_stack([np.log1p(np.asarray(Xr[:, gidx[g]].todense()).ravel() / lib_r * 1e4) for g in genes])
    Dm = np.column_stack(dorc)
    Rs = []                                                        # random sets: NSET per gene
    for s in sizes:
        for _ in range(NSET):
            idx = np.concatenate([matched_random(tp[p], 1, s) for p in s])
            Rs.append(np.log1p(np.asarray(Xa[:, idx].sum(1)).ravel() / lib_a * 1e4))
    Rm = np.column_stack(Rs)
    G = zz(smooth(Gm, xy)); Dz = zz(smooth(Dm, xy)); Rz = zz(smooth(Rm, xy))

    rows, pair = [], []
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values; lab = gaston_domains(d, z[f"Y_{axis}"], L_=kdom(DS, axis))
        grid = np.linspace(np.percentile(d, 1), np.percentile(d, 99), NGRID)
        kgrid = lab[np.argmin(np.abs(d[None, :] - grid[:, None]), axis=1)]
        sg, ig = fit_gauss(G, lab, d, t=T_LRT); sd_, id_ = fit_gauss(Dz, lab, d, t=T_LRT); sr, ir_ = fit_gauss(Rz, lab, d, t=T_LRT)
        CG = curves(sg, ig, d, lab, grid); CD = curves(sd_, id_, d, lab, grid); CR = curves(sr, ir_, d, lab, grid)
        CRq = np.stack([np.percentile(CR[:, a * NSET:(a + 1) * NSET], q, axis=1) for a in range(len(genes))
                        for q in (5, 50, 95)], 1).reshape(len(grid), len(genes), 3)
        np.savez_compressed(f"{OUT}/dorc_curves_{axis}.npz", grid=grid, kgrid=kgrid, genes=np.array(genes),
                            gene=CG, dorc=CD, random_q=CRq,
                            dorc_peaks=np.array([";".join(pn[s_]) for s_ in sizes], dtype=object))
        for k in range(lab.max() + 1):
            gk = np.where(kgrid == k)[0]
            if len(gk) < 3: continue
            w = grid[gk[-1]] - grid[gk[0]]; mid = gk[len(gk) // 2]
            for a, g in enumerate(genes):
                slope_w = (CG[gk[-1], a] - CG[gk[0], a])                 # z change across the domain
                if abs(slope_w) < MIN_SLOPE: continue
                cp = lambda C_, j: (C_[mid, j] - CG[mid, a]) / (slope_w / w) / w
                cpd = cp(CD, a); cpr = np.array([cp(CR, a * NSET + s) for s in range(NSET)])
                pair.append(dict(axis=axis, domain=k, gene=g, n_peaks=len(sizes[a]), cp_dorc=cpd,
                                 cp_random_median=np.median(cpr), net=cpd - np.median(cpr),
                                 pct_vs_random=float((cpr < cpd).mean())))
    P = pd.DataFrame(pair); P.to_csv(f"{OUT}/dorc_cp_pairs.csv", index=False)
    rng2 = np.random.default_rng(1)
    for (axis, k), s in P.groupby(["axis", "domain"]):
        net = s.net.values
        if len(net) < 5: continue
        bs = [np.median(rng2.choice(net, len(net))) for _ in range(4000)]
        rows.append(dict(axis=axis, domain=k, n_genes=len(net), net_cp=np.median(net), ci_lo=np.percentile(bs, 2.5),
                         ci_hi=np.percentile(bs, 97.5), p=wilcoxon(net).pvalue, frac_ahead=float((net > 0).mean())))
    T = pd.DataFrame(rows); T["p_bonf"] = np.minimum(T.p * len(T), 1)
    T.to_csv(f"{OUT}/dorc_cp_domain_summary.csv", index=False)
    print(T.round(3).to_string(index=False))
    lead = P[(P.pct_vs_random >= 0.95) | (P.pct_vs_random <= 0.05)]
    print(f"\ngene-domain cases outside their own random-set 90% band: {len(lead)}/{len(P)} "
          f"(chance ~10%) -> {100 * len(lead) / max(len(P), 1):.1f}%")
    print(lead.sort_values("net", key=abs, ascending=False).head(12)[["axis", "domain", "gene", "n_peaks", "cp_dorc", "cp_random_median", "net", "pct_vs_random"]].round(3).to_string(index=False))
    return P, T


if __name__ == "__main__":
    main()
