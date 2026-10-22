"""Per-gene isodepth functions using SATURN'S OWN PATHWAY STRUCTURE.

Saturn learns two isodepths (d_RNA, d_ATAC) and four pathway functions:
    RNA  = f_own(d_RNA)  + f_cross(d_ATAC)
    ATAC = g_own(d_ATAC) + g_cross(d_RNA)
so at gene level we fit the same additive decomposition, with each pathway a
continuous piecewise-linear function (one breakpoint), on isodepth bins:

  gene g expression:  y_g ~ f_g(d_RNA) + h_g(d_ATAC)      (own + ATAC->RNA cross)
  g's enhancers:      e_g ~ k_g(d_ATAC)                    (d_ATAC ONLY, as the peaks
                                                            are the inflammatory
                                                            candidate enhancers)
Enhancers = peaks within +-100 kb of the TSS, promoters (+-2 kb) excluded.
Segmentation: genes are segmented on d_RNA, enhancers on d_ATAC (1-D k-means, ordered).

BINNING follows GASTON (gaston.binning_and_plotting.bin_data / segmented_fit.pw_linear_fit),
which is what GASTON uses for its per-gene figures -- NOT metacells:
  * equal-width bins along isodepth (GASTON default 70; fewer here, the sections have
    1.7-3.1k spots),
  * raw counts SUMMED per bin with the library sizes summed into a per-bin exposure,
  * per-gene fit = POISSON regression with a log-exposure offset, so the curve is
    log rate vs isodepth,
  * genes filtered by total UMI across spots (GASTON default 500).
"""
import sys, os, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import _read_h5ad_safe, LPC_TP, INFL_GENES, _tss, H
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})
AXSET = os.environ.get("LPC_AXES", "cut")   # "cut" = whole-genome Saturn fit (GLM-PCA 3k HVG +
                                            # genome-wide LSI); "infl" = inflammation-targeted
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/lpc_gene_isodepth_{AXSET}"; os.makedirs(OUT, exist_ok=True)
TPS = ["LPC5", "LPC10", "LPC21"]
WIN, PROM, NB, KREG = 100_000, 2_000, 30, 4
# GASTON uses total-UMI > 500 on Visium. These DBiT sections are far shallower: the
# median inflammatory gene has 146-301 UMI in total, so 500 would keep 8-19 of 52 genes.
UMI_MIN, ENH_MIN, OFFSET = 50, 50, 1e6
GRP = {g: k for k, gs in INFL_GENES.items() for g in gs}


def basis(d, bp):
    return np.c_[d, np.maximum(0.0, d - bp)]


def gaston_bins(d, nb):
    """equal-width bins along isodepth, as in gaston.bin_data"""
    lo, hi = d.min() - 1e-6, d.max() + 1e-6
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
    keep = np.array([ (idx == b).sum() > 0 for b in range(nb) ])
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, centres[keep], np.where(keep)[0]


def pois_fit(cnt, expo, d1, b1, d2=None, b2=None, n_iter=25):
    """Poisson regression with log-exposure offset (IRLS), GASTON's per-gene model."""
    X = np.c_[np.ones(len(cnt)), basis(d1, b1)] if d2 is None else \
        np.c_[np.ones(len(cnt)), basis(d1, b1), basis(d2, b2)]
    off = np.log(np.maximum(expo, 1))
    beta = np.zeros(X.shape[1]); beta[0] = np.log(max(cnt.sum(), 1) / max(expo.sum(), 1))
    for _ in range(n_iter):
        eta = X @ beta + off
        mu = np.exp(np.clip(eta, -30, 30))
        W = np.maximum(mu, 1e-8)
        z = eta - off + (cnt - mu) / W
        beta_new = np.linalg.lstsq(X * W[:, None], z * W, rcond=None)[0]
        if not np.all(np.isfinite(beta_new)):
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new; break
        beta = beta_new
    mu = np.exp(np.clip(X @ beta + off, -30, 30))
    # deviance-based pseudo-R2 vs intercept-only
    mu0 = np.maximum(expo * cnt.sum() / max(expo.sum(), 1), 1e-9)
    dev = lambda m: 2 * np.sum(np.where(cnt > 0, cnt * np.log(np.maximum(cnt, 1e-9) / m), 0) - (cnt - m))
    d1_, d0_ = dev(mu), dev(mu0)
    phi = float(np.sum((cnt - mu) ** 2 / np.maximum(mu, 1e-9)) / max(len(cnt) - X.shape[1], 1))
    return dict(beta=beta, mu=mu, rate=mu / np.maximum(expo, 1), R2=float(1 - d1_ / max(d0_, 1e-9)), phi=phi,
                part1=basis(d1, b1) @ beta[1:3],
                part2=(basis(d2, b2) @ beta[3:5]) if d2 is not None else np.zeros(len(cnt)))


def fit_add(y, d1, d2=None, n_grid=20):
    """y ~ pw(d1) [+ pw(d2)]; grid search breakpoints. Returns dict with R2 and parts."""
    g1 = np.quantile(d1, np.linspace(0.15, 0.85, n_grid))
    g2 = np.quantile(d2, np.linspace(0.15, 0.85, n_grid)) if d2 is not None else [None]
    best = None
    for b1 in g1:
        for b2 in g2:
            X = np.c_[np.ones(len(y)), basis(d1, b1)] if b2 is None else \
                np.c_[np.ones(len(y)), basis(d1, b1), basis(d2, b2)]
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ c
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            if best is None or r2 > best["R2"]:
                p1 = basis(d1, b1) @ c[1:3]
                p2 = basis(d2, b2) @ c[3:5] if b2 is not None else np.zeros_like(y)
                best = dict(R2=r2, bp1=b1, bp2=b2, part1=p1, part2=p2, pred=pred)
    v1, v2 = best["part1"].var(), best["part2"].var()
    best["share_own"] = float(v1 / (v1 + v2 + 1e-12))
    return best


def main():
    genes_all = [g for grp in INFL_GENES.values() for g in grp]
    tss = _tss(genes_all)
    summary = []
    for tp in TPS:
        ax = pd.read_csv(f"{R}/lpc_{tp.lower()}_{AXSET}/axes.csv", index_col=0)
        dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
        xy = np.c_[ax.x.values, ax.y.values]
        cfg = LPC_TP[tp]
        r = _read_h5ad_safe(f"{H}/{cfg['rna']}"); a = _read_h5ad_safe(f"{H}/{cfg['atac']}")
        idx = pd.Index([str(b) for b in ax.index]); r = r[idx].copy(); a = a[idx].copy()
        X = sp.csr_matrix(r.X); vn = list(map(str, r.var_names)); lib = np.asarray(X.sum(1)).ravel().clip(1)
        A_ = sp.csr_matrix(a.X); alib = np.asarray(A_.sum(1)).ravel().clip(1)
        pk = pd.Series(a.var_names).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
        pk.columns = ["chr", "s", "e"]; mid = (pk["s"].astype(float) + pk["e"].astype(float)) / 2

        seg = {}
        for nm, dd in [("RNA", dR), ("ATAC", dA)]:
            km = KMeans(KREG, n_init=10, random_state=0).fit(dd.reshape(-1, 1))
            rm = {o: n for n, o in enumerate(np.argsort(km.cluster_centers_.ravel()))}
            seg[nm] = np.array([rm[l] for l in km.labels_])
        # GASTON-style equal-width isodepth bins; counts summed, exposure summed
        iR, cR, kR = gaston_bins(dR, NB)
        iA, cA, kA = gaston_bins(dA, NB)
        binsum = lambda v, idx, keep: np.array([v[idx == b].sum() for b in keep])
        expR = binsum(lib, iR, kR); expA = binsum(alib, iA, kA)
        dA_on_R = np.array([dA[iR == b].mean() for b in kR])

        def best_pois(cnt, expo, d1, d2=None, n_grid=12):
            g1 = np.quantile(d1, np.linspace(0.15, 0.85, n_grid))
            g2 = np.quantile(d2, np.linspace(0.15, 0.85, n_grid)) if d2 is not None else [None]
            best = None
            for b1 in g1:
                for b2 in g2:
                    f = pois_fit(cnt, expo, d1, b1, d2, b2)
                    if best is None or f["R2"] > best["R2"]:
                        best = dict(f, bp1=b1, bp2=b2)
            return best

        recs, Br, Be, names = [], [], [], []
        for g in genes_all:
            if g not in vn or g not in tss:
                continue
            cg = np.asarray(X[:, vn.index(g)].todense()).ravel()
            if cg.sum() < UMI_MIN:                                  # GASTON's UMI filter (relaxed)
                continue
            ch, t = tss[g]
            m = (pk["chr"] == ch).values & (np.abs(mid.values - t) <= WIN) & (np.abs(mid.values - t) > PROM)
            if m.sum() < 3:
                continue
            ce = np.asarray(A_[:, np.where(m)[0]].sum(1)).ravel()
            if ce.sum() < ENH_MIN:
                continue
            yb = binsum(cg, iR, kR)                                  # genes binned on d_RNA
            eb = binsum(ce, iA, kA)                                  # enhancers binned on d_ATAC
            fr = best_pois(yb, expR, cR, dA_on_R)                    # own + ATAC->RNA cross
            fo = best_pois(yb, expR, cR)                             # own only
            dR_on_A = np.array([dR[iA == b].mean() for b in kA])
            fe = best_pois(eb, expA, cA)                             # enhancers: own axis only
            fe2 = best_pois(eb, expA, cA, dR_on_A)                   # enhancers: own + d_RNA cross
            Br.append(fr["rate"]); Be.append(fe["rate"]); names.append(g)
            v1, v2 = fr["part1"].var(), fr["part2"].var()
            recs.append(dict(gene=g, group=GRP[g], n_enh=int(m.sum()), umi_rna=int(cg.sum()), umi_enh=int(ce.sum()),
                             R2_rna_own=round(fo["R2"], 3), R2_rna_own_plus_cross=round(fr["R2"], 3),
                             gain_from_dATAC=round(fr["R2"] - fo["R2"], 3),
                             share_own=round(float(v1 / (v1 + v2 + 1e-12)), 2),
                             bp_rna=round(float(fr["bp1"]), 2), bp_cross=round(float(fr["bp2"]), 2),
                             R2_enh_dATAC=round(fe["R2"], 3), bp_enh=round(float(fe["bp1"]), 2),
                             R2_enh_own_plus_cross=round(fe2["R2"], 3),
                             gain_enh_from_dRNA=round(fe2["R2"] - fe["R2"], 3),
                             phi_rna=round(fr["phi"], 1), phi_enh=round(fe["phi"], 1),
                             **{f"RNA_R{k}": round(float(np.log1p(cg[seg['RNA'] == k].sum() / max(lib[seg['RNA'] == k].sum(), 1) * 1e4)), 3) for k in range(KREG)},
                             **{f"ENH_R{k}": round(float(np.log1p(ce[seg['ATAC'] == k].sum() / max(alib[seg['ATAC'] == k].sum(), 1) * 1e4)), 3) for k in range(KREG)}))
        df = pd.DataFrame(recs); Br = np.array(Br); Be = np.array(Be)
        zs = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
        order = np.argsort(np.argmax(zs(Br), 1))

        fig, AX = plt.subplots(2, 5, figsize=(16, 7))
        for row_i, (M, dbb, segname, nm) in enumerate([(Br, cR, "RNA", "RNA rate vs d_RNA (Poisson, GASTON bins)"),
                                                       (Be, cA, "ATAC", "enhancer rate vs d_ATAC")]):
            a_ = AX[row_i, 0]
            im = a_.imshow(zs(M)[order], aspect="auto", cmap="coolwarm", vmin=-2, vmax=2,
                           extent=[dbb[0], dbb[-1], len(order), 0])
            for k in range(KREG - 1):
                a_.axvline(np.max((dR if segname == "RNA" else dA)[seg[segname] == k]), color="k", lw=0.8)
            a_.set_yticks(np.arange(len(order)) + 0.5); a_.set_yticklabels(np.array(names)[order], fontsize=4)
            a_.set_xlabel(f"d_{segname}"); a_.set_title(f"{tp}: {nm}"); fig.colorbar(im, ax=a_, fraction=0.046)
            a_ = AX[row_i, 1]
            cols = [c for c in df.columns if c.startswith("RNA_R" if row_i == 0 else "ENH_R")]
            V = df[cols].values[order]
            im = a_.imshow(V, aspect="auto", cmap="coolwarm", vmin=-np.abs(V).max(), vmax=np.abs(V).max())
            a_.set_xticks(range(KREG)); a_.set_xticklabels([f"R{k}" for k in range(KREG)])
            a_.set_yticks(range(len(order))); a_.set_yticklabels(np.array(names)[order], fontsize=4)
            a_.set_title(f"{tp}: per region of d_{segname}"); fig.colorbar(im, ax=a_, fraction=0.046)
        top = df.sort_values("R2_rna_own_plus_cross", ascending=False).head(3)
        for c, (_, row) in enumerate(top.iterrows()):
            gi = names.index(row.gene)
            cg = np.asarray(X[:, vn.index(row.gene)].todense()).ravel()
            yb = binsum(cg, iR, kR)
            fr = best_pois(yb, expR, cR, dA_on_R)
            a_ = AX[0, c + 2]
            L = lambda rate: np.log(rate * OFFSET + 1)                      # GASTON's y scale
            a_.plot(cR, L(yb / np.maximum(expR, 1)), "o", ms=3, color="#999999", label="binned rate")
            a_.plot(cR, L(fr["rate"]), color="#1f77b4", lw=1.5, label="own + cross")
            base = np.log(OFFSET) + fr["beta"][0]
            a_.plot(cR, base + fr["part1"], color="#9ecae1", lw=1.2, ls="--", label="f(d_RNA)")
            a_.plot(cR, base + fr["part2"], color="#fdae6b", lw=1.2, ls="--", label="h(d_ATAC) cross")
            a_.axvline(fr["bp1"], color="#9ecae1", ls=":", lw=0.6)
            a_.legend(fontsize=5, frameon=False)
            a_.set_xlabel("d_RNA"); a_.set_ylabel("log(rate x 1e6 + 1)")
            a_.set_title(f"{row.gene} RNA: R2 {row.R2_rna_own_plus_cross:.2f}\n(own {row.R2_rna_own:.2f}, "
                         f"+d_ATAC {row.gain_from_dATAC:+.2f}, own share {row.share_own:.2f})")
            ch, t = tss[row.gene]
            m = (pk["chr"] == ch).values & (np.abs(mid.values - t) <= WIN) & (np.abs(mid.values - t) > PROM)
            ce = np.asarray(A_[:, np.where(m)[0]].sum(1)).ravel()
            eb = binsum(ce, iA, kA)
            dR_on_A = np.array([dR[iA == b].mean() for b in kA])
            fe = best_pois(eb, expA, cA)
            fe2 = best_pois(eb, expA, cA, dR_on_A)
            a_ = AX[1, c + 2]
            a_.plot(cA, L(eb / np.maximum(expA, 1)), "o", ms=3, color="#999999", label="binned rate")
            a_.plot(cA, L(fe2["rate"]), color="#d62728", lw=1.5, label="own + cross")
            baseA = np.log(OFFSET) + fe2["beta"][0]
            a_.plot(cA, baseA + fe2["part1"], color="#fcae91", lw=1.2, ls="--", label="g(d_ATAC)")
            a_.plot(cA, baseA + fe2["part2"], color="#9ecae1", lw=1.2, ls="--", label="g_cross(d_RNA)")
            a_.axvline(fe2["bp1"], color="#fcae91", ls=":", lw=0.6)
            a_.legend(fontsize=5, frameon=False)
            a_.set_xlabel("d_ATAC"); a_.set_ylabel("log(rate x 1e6 + 1)")
            a_.set_title(f"{row.gene} enhancers ({int(row.n_enh)} peaks): own R2 {row.R2_enh_dATAC:.2f}, "
                         f"+d_RNA {row.gain_enh_from_dRNA:+.2f}")
        fig.suptitle(f"{tp}: Saturn pathway decomposition per gene -- RNA = f(d_RNA) + h(d_ATAC); "
                     f"enhancers = k(d_ATAC)", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{OUT}/gene_isodepth_{tp}.{ext}", dpi=250 if ext == "png" else None)
        plt.close(fig)
        fig2, a2 = plt.subplots(1, 2, figsize=(7.2, 3.4))
        for i, nm in enumerate(["RNA", "ATAC"]):
            a2[i].scatter(xy[:, 0], xy[:, 1], c=seg[nm], cmap="tab10", s=4, linewidths=0, vmin=0, vmax=9)
            a2[i].set_aspect("equal"); a2[i].set_xticks([]); a2[i].set_yticks([])
            a2[i].set_title(f"{tp}: regions of d_{nm}")
        fig2.tight_layout(); fig2.savefig(f"{OUT}/regions_{tp}.png", dpi=250); plt.close(fig2)
        df.to_csv(f"{OUT}/gene_isodepth_{tp}.csv", index=False)
        summary.append(dict(timepoint=tp, genes=len(df),
                            med_R2_rna_own=round(df.R2_rna_own.median(), 2),
                            med_R2_rna_both=round(df.R2_rna_own_plus_cross.median(), 2),
                            med_gain_dATAC=round(df.gain_from_dATAC.median(), 2),
                            med_share_own=round(df.share_own.median(), 2),
                            med_R2_enh_dATAC=round(df.R2_enh_dATAC.median(), 2),
                            med_gain_enh_dRNA=round(df.gain_enh_from_dRNA.median(), 2),
                            med_phi_rna=round(df.phi_rna.median(), 1), med_phi_enh=round(df.phi_enh.median(), 1)))
    print(pd.DataFrame(summary).to_string(index=False)); print("written", OUT)



if __name__ == "__main__":
    main()