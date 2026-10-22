"""GASTON Fig-4l-style composite panel, per gene.

Per gene, 2 rows x 3 columns:
  row 1 (RNA):  [domains in tissue + isodepth contours] [raw expression map]
                [FITTED expression map: GASTON's function evaluated at each spot's
                 isodepth, exp(intercept_domain + slope_domain * d)]
                [expression as a function of d_RNA, per-domain fits, boundaries]
  row 2 (ATAC): same four for the gene's best enhancer peak, on d_ATAC.

The FUNCTION itself is GASTON's own estimator: gaston.segmented_fit.pw_linear_fit runs a
per-domain Poisson regression at SPOT level with a likelihood-ratio test (t=0.1) that
picks a sloped or a constant fit per domain, and returns slopes, intercepts and
discontinuities. Points are the binned rates (GASTON's plotting convention).

DOMAINS are GASTON's own: gaston.dp_related.dp_bucketized + find_segments_from_dp run on
our Saturn isodepth and the modality's target matrix (RNA GLM-PCs / ATAC LSI). The DP
places boundaries where the targets change slope along the isodepth -- NOT k-means on d,
which is what the earlier version used.
Boundaries are drawn in ORIGINAL SPACE as isodepth contours at the boundary values.
Empty (zero-count) bins are dropped from the function plots.
"""
import sys, os, numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from saturn import domains as SD
from prep_real2 import _read_h5ad_safe, LPC_TP, _tss, H
from lpc_gene_isodepth import gaston_bins, pois_fit
from gaston.dp_related import dp_bucketized, find_segments_from_dp
from gaston.segmented_fit import pw_linear_fit
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})
AXSET = os.environ.get("LPC_AXES", "cut")
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/lpc_gaston_fig4l_{AXSET}"; os.makedirs(OUT, exist_ok=True)
NB_RNA, NB_ATAC, KDOM, NBUCKET, OFFSET = 30, 20, 4, 50, 1e6
WIN, PROM, PROXIMAL = 100_000, 2_000, 20_000
# Candidate regulatory peaks: everything within +-WIN of the TSS, INCLUDING promoters.
# Classes: promoter |d|<=PROM (2 kb); proximal enhancer PROM<|d|<=PROXIMAL (20 kb);
#          long-distance enhancer |d|>PROXIMAL.


def peak_class(dist):
    ad = abs(dist)
    return "promoter" if ad <= PROM else ("proximal enhancer" if ad <= PROXIMAL else "long-distance enhancer")
# ATAC is segmented more coarsely than RNA: sparse peaks hold only ~12-39 counts per
# domain at K=4, so GASTON's per-domain LRT has no power and returns flat fits. The power
# study (lpc_peak_filter_design.py) shows a domain needs ~50 counts (power 0.98); at K=2
# the number of peaks powered in ALL domains goes 0->11 (LPC5), 0->29 (LPC10), 10->42 (LPC21).
T_LRT = 0.1      # GASTON's default LRT gate: a domain keeps its slope only if p < t
KDOM_ATAC = 4
MIN_DOM_FRAC = 0.01     # domains smaller than this are DP tail artifacts -> merged
NBUCKET_ATAC = 20       # ATAC is bucketed more coarsely than RNA (GASTON default 50):
                        # sparse peaks need more counts per bucket for the DP to be stable (50 -> 20)
MIN_BIN_SPOTS = 5       # isodepth bins with fewer spots are dropped (outlier rates)
L = lambda rate: np.log(rate * OFFSET + 1)
COLS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def gaston_domains(d, Y, L_=KDOM, nb=NBUCKET):
    """GASTON's DP segmentation of the isodepth using the modality's targets (saturn.domains.gaston_domains)"""
    return SD.gaston_domains(d, Y, n_domains=L_, n_buckets=nb, min_frac=MIN_DOM_FRAC, verbose=True)


def boundaries(d, lab):
    """isodepth values where the domain changes (saturn.domains.boundaries)"""
    return SD.boundaries(d, lab)


def pois_linear(cnt, expo, x):
    X = np.c_[np.ones(len(x)), x]
    b = np.array([np.log(max(cnt.sum(), 1) / max(expo.sum(), 1)), 0.0])
    for _ in range(30):
        mu = np.exp(np.clip(X @ b + np.log(np.maximum(expo, 1)), -30, 30))
        W = np.maximum(mu, 1e-8); z = X @ b + (cnt - mu) / W
        nb_ = np.linalg.lstsq(X * W[:, None], z * W, rcond=None)[0]
        if not np.all(np.isfinite(nb_)):
            break
        if np.max(np.abs(nb_ - b)) < 1e-7:
            b = nb_; break
        b = nb_
    return b


def panel_space(a_, xy, val, title, cmap, contour_d=None, bnds=(), discrete=False):
    if discrete:
        for k in np.unique(val.astype(int)):
            m = val.astype(int) == k
            a_.scatter(xy[m, 0], xy[m, 1], s=4, color=COLS[k % len(COLS)], linewidths=0, label=f"domain {k}")
        a_.legend(fontsize=5, frameon=False, loc="lower right")
    else:
        sc = a_.scatter(xy[:, 0], xy[:, 1], c=val, cmap=cmap, s=4, linewidths=0,
                        vmin=np.nanpercentile(val, 1), vmax=np.nanpercentile(val, 99))
        plt.colorbar(sc, ax=a_, fraction=0.046)
    if contour_d is not None and len(bnds):
        a_.tricontour(xy[:, 0], xy[:, 1], contour_d, levels=sorted(bnds), colors="k", linewidths=1.0, alpha=0.7)
    a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([]); a_.set_title(title)


def panel_function(a_, cnt, expo, d, lab, nb, xlab, ylab, title, gfit=None, grow=None):
    """points = binned rates; lines = GASTON's pw_linear_fit (slope/intercept per domain)"""
    idx, centres, keep = gaston_bins(d, nb)
    bc = np.array([cnt[idx == b].sum() for b in keep])
    be = np.array([expo[idx == b].sum() for b in keep])
    bl = np.array([np.bincount(lab[idx == b], minlength=lab.max() + 1).argmax() for b in keep])
    nsp = np.array([(idx == b).sum() for b in keep])
    ok = (bc > 0) & (nsp >= MIN_BIN_SPOTS)          # drop empty AND under-populated bins
    bc, be, bl, cen = bc[ok], be[ok], bl[ok], centres[ok]
    y = L(bc / np.maximum(be, 1))
    for dom in np.unique(bl):
        m = bl == dom
        a_.scatter(cen[m], y[m], s=16, color=COLS[dom % len(COLS)], zorder=3)
        if gfit is not None:                      # GASTON slope/intercept for this domain
            slope_mat, intercept_mat = gfit[0], gfit[1]
            xx = np.linspace(d[lab == dom].min(), d[lab == dom].max(), 50)
            a_.plot(xx, np.log(OFFSET) + intercept_mat[grow, dom] + slope_mat[grow, dom] * xx,
                    color=COLS[dom % len(COLS)], lw=1.8, zorder=2)
        elif m.sum() >= 3:
            b = pois_linear(bc[m], be[m], cen[m])
            xx = np.linspace(cen[m].min(), cen[m].max(), 50)
            a_.plot(xx, np.log(OFFSET) + b[0] + b[1] * xx, color=COLS[dom % len(COLS)], lw=1.8, zorder=2)
    for i in range(len(bl) - 1):
        if bl[i] != bl[i + 1]:
            a_.axvline(0.5 * (cen[i] + cen[i + 1]), color="k", ls="--", lw=1.2, alpha=0.3)
    a_.set_xlabel(xlab); a_.set_ylabel(ylab); a_.set_title(f"{title}\n{ok.sum()}/{len(ok)} bins kept")
    a_.spines[["top", "right"]].set_visible(False)
    out = pd.DataFrame(dict(isodepth_bin=cen, domain=bl, n_spots=nsp[ok], counts=bc, exposure=be,
                            rate=bc / np.maximum(be, 1), log_obs=y))
    if gfit is not None:
        sl, ic = gfit[0], gfit[1]
        out["slope"] = sl[grow, bl]; out["intercept"] = ic[grow, bl]
        out["log_fitted"] = np.log(OFFSET) + ic[grow, bl] + sl[grow, bl] * cen
        out["fitted_rate"] = np.exp(ic[grow, bl] + sl[grow, bl] * cen)
    return out


def main():
    allvals, class_rows = [], []
    def best_pois_simple(cnt, expo, centres, n_grid=10):
        best = None
        for bp in np.quantile(centres, np.linspace(0.2, 0.8, n_grid)):
            f = pois_fit(cnt, expo, centres, bp)
            if best is None or f["R2"] > best["R2"]:
                best = f
        return best


    for tp in ["LPC5", "LPC10", "LPC21"]:
        ax = pd.read_csv(f"{R}/lpc_{tp.lower()}_{AXSET}/axes.csv", index_col=0)
        z = dict(np.load(f"{R}/targets_lpc_{tp.lower()}_{AXSET}.npz", allow_pickle=True))
        dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
        xy = np.c_[ax.x.values, ax.y.values]
        labR = gaston_domains(dR, z["Y_RNA"])
        labA = gaston_domains(dA, z["Y_ATAC"], L_=KDOM_ATAC, nb=NBUCKET_ATAC)
        bR, bA = boundaries(dR, labR), boundaries(dA, labA)
        print(f"{tp}: GASTON DP domains -- RNA sizes {np.bincount(labR).tolist()} boundaries {np.round(bR,2).tolist()}; "
              f"ATAC sizes {np.bincount(labA).tolist()} boundaries {np.round(bA,2).tolist()}", flush=True)
        r = _read_h5ad_safe(f"{H}/{LPC_TP[tp]['rna']}"); a = _read_h5ad_safe(f"{H}/{LPC_TP[tp]['atac']}")
        idx = pd.Index([str(b) for b in ax.index]); r = r[idx].copy(); a = a[idx].copy()
        X = sp.csr_matrix(r.X); vn = list(map(str, r.var_names)); lib = np.asarray(X.sum(1)).ravel().clip(1)
        A_ = sp.csr_matrix(a.X); alib = np.asarray(A_.sum(1)).ravel().clip(1); pkn = list(a.var_names)
        pk = pd.Series(a.var_names).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
        pk.columns = ["chr", "s", "e"]
        mid = (pk["s"].astype(float) + pk["e"].astype(float)) / 2
        pw = pd.read_csv(f"{R}/lpc_enhancer_peakwise_{AXSET}/peakwise_{tp}.csv")
        sig_all = pw[pw.q < 0.05] if (pw.q < 0.05).any() else pw.nsmallest(6, "p")
        gene_idx = [vn.index(g) for g in dict.fromkeys(sig_all.gene) if g in vn]
        iA0 = gaston_bins(dA, NB_ATAC)
        expA0 = np.array([alib[iA0[0] == b].sum() for b in iA0[2]])
        tss0 = _tss([vn[j] for j in gene_idx])
        best_peak = {}
        for j0 in gene_idx:
            g0 = vn[j0]
            if g0 not in tss0:
                continue
            ch0, t0 = tss0[g0]
            cj = np.where((pk["chr"] == ch0).values & (np.abs(mid.values - t0) <= WIN))[0]
            bb = None
            for j_ in cj:
                c_ = np.asarray(A_[:, j_].todense()).ravel()
                if c_.sum() < 1:
                    continue
                yb_ = np.array([c_[iA0[0] == b].sum() for b in iA0[2]])
                f_ = best_pois_simple(yb_, expA0, iA0[1])
                if bb is None or f_["R2"] > bb[0]:
                    bb = (f_["R2"], j_, int(c_.sum()))
            if bb is not None:
                dist0 = float(mid.values[bb[1]] - t0)
                best_peak[g0] = bb + (dist0, peak_class(dist0))
                # also keep the best peak of each class, for the report
                for cls in ("promoter", "proximal enhancer", "long-distance enhancer"):
                    sub = [j_ for j_ in cj if peak_class(mid.values[j_] - t0) == cls]
                    bc = None
                    for j_ in sub:
                        c_ = np.asarray(A_[:, j_].todense()).ravel()
                        if c_.sum() < 1:
                            continue
                        yb_ = np.array([c_[iA0[0] == b].sum() for b in iA0[2]])
                        f_ = best_pois_simple(yb_, expA0, iA0[1])
                        if bc is None or f_["R2"] > bc[0]:
                            bc = (f_["R2"], j_, int(c_.sum()))
                    if bc is not None:
                        class_rows.append(dict(timepoint=tp, gene=g0, cls=cls, peak=pkn[bc[1]],
                                               kb_from_TSS=round((mid.values[bc[1]] - t0) / 1000, 1),
                                               counts=bc[2], R2=round(bc[0], 3), n_candidates=len(sub)))
        peak_idx = [v[1] for v in best_peak.values()]
        print(f"   GASTON pw_linear_fit: {len(gene_idx)} genes, {len(peak_idx)} peaks", flush=True)
        def gaston_fit(counts_sel, total, labels, iso):
            """GASTON derives each spot's exposure by summing ALL features of the matrix it
            is given, so a matrix of only the selected features leaves most spots with zero
            exposure (NaN). Append a 'rest of the library' column so exposures are the true
            library sizes, and keep only the selected features."""
            from saturn.piecewise import fit_poisson
            return fit_poisson(counts_sel, total, labels, iso, t=T_LRT)

        fitG = gaston_fit(np.asarray(X[:, gene_idx].todense()), lib, labR, dR)
        fitP = gaston_fit(np.asarray(A_[:, peak_idx].todense()), alib, labA, dA)
        gpos = {vn[j]: i for i, j in enumerate(gene_idx)}
        ppos = {pkn[j]: i for i, j in enumerate(peak_idx)}
        sig = pw[pw.q < 0.05] if (pw.q < 0.05).any() else pw.nsmallest(6, "p")
        iR_, _, kR_ = gaston_bins(dR, NB_RNA)
        for g in dict.fromkeys(sig.gene):
            if g not in vn:
                continue
            cg = np.asarray(X[:, vn.index(g)].todense()).ravel()
            if sum(cg[iR_ == b].sum() > 0 for b in kR_) < 10:
                continue
            if g not in best_peak:
                continue
            r2b, jb, cntb, distb, clsb = best_peak[g]
            row = pd.Series(dict(peak=pkn[jb], total_counts=cntb, R2=round(r2b, 3), q=np.nan,
                                 kb=round(distb / 1000, 1), cls=clsb))
            cp = np.asarray(A_[:, pkn.index(row.peak)].todense()).ravel()
            def fitted_map(gfit, grow, d, lab):
                """GASTON's fitted function evaluated per spot -> rate per spot"""
                sl, ic = gfit[0], gfit[1]
                return np.exp(ic[grow, lab] + sl[grow, lab] * d)

            fitR = fitted_map(fitG, gpos[g], dR, labR)
            fitP_ = fitted_map(fitP, ppos[row.peak], dA, labA)
            fig, AX = plt.subplots(2, 5, figsize=(18.5, 6.6))
            panel_space(AX[0, 0], xy, dR, f"{tp}: rho_RNA (Saturn isodepth)", "coolwarm", contour_d=dR, bnds=bR)
            panel_space(AX[1, 0], xy, dA, f"{tp}: rho_ATAC (Saturn isodepth)", "coolwarm", contour_d=dA, bnds=bA)
            panel_space(AX[0, 1], xy, labR, f"{tp}: RNA domains (GASTON DP)\ncontours = domain boundaries",
                        None, contour_d=dR, bnds=bR, discrete=True)
            panel_space(AX[0, 2], xy, L(cg / lib), f"{g} RAW expression", "magma", contour_d=dR, bnds=bR)
            panel_space(AX[0, 3], xy, L(fitR), f"{g} FITTED expression (GASTON function)", "magma",
                        contour_d=dR, bnds=bR)
            vG = panel_function(AX[0, 4], cg, lib, dR, labR, NB_RNA, "isodepth d_RNA", "log(expr x 1e6 + 1)",
                                f"{g} expression function ({int(cg.sum())} UMI), GASTON fit",
                                gfit=fitG, grow=gpos[g])
            vG.insert(0, "feature", g); vG.insert(1, "modality", "RNA"); vG.insert(2, "timepoint", tp)
            panel_space(AX[1, 1], xy, labA, f"{tp}: ATAC domains (GASTON DP)\ncontours = domain boundaries",
                        None, contour_d=dA, bnds=bA, discrete=True)
            panel_space(AX[1, 2], xy, L(cp / alib), f"{row.peak}\n{row.cls} ({row.kb:+.1f} kb) RAW", "magma",
                        contour_d=dA, bnds=bA)
            panel_space(AX[1, 3], xy, L(fitP_), f"{row.peak}\nFITTED accessibility (GASTON function)", "magma",
                        contour_d=dA, bnds=bA)
            dom_counts = [int(cp[labA == k].sum()) for k in range(labA.max() + 1)]
            vP = panel_function(AX[1, 4], cp, alib, dA, labA, NB_ATAC, "isodepth d_ATAC", "log(access x 1e6 + 1)",
                                f"{g} enhancer {row.peak}\n{int(row.total_counts)} counts, R2 {row.R2:.2f}, "
                                f"best-R2 peak of {g}: {row.cls} at {row.kb:+.1f} kb, R2 {row.R2:.2f}\n"
                                f"counts/domain {dom_counts}",
                                gfit=fitP, grow=ppos[row.peak])
            vP.insert(0, "feature", row.peak); vP.insert(1, "modality", "ATAC"); vP.insert(2, "timepoint", tp)
            pd.concat([vG, vP]).to_csv(f"{OUT}/fitted_values_{tp}_{g}.csv", index=False)
            allvals.append(pd.concat([vG, vP]))
            fig.suptitle(f"{tp} {g}: expression and enhancer accessibility as functions of isodepth, "
                         f"with GASTON domain boundaries in tissue", fontsize=9)
            fig.tight_layout()
            for e in ("png", "svg"):
                fig.savefig(f"{OUT}/{tp}_{g}.{e}", dpi=300 if e == "png" else None)
            plt.close(fig)
            print(f"   wrote {tp}_{g}", flush=True)
    if class_rows:
        cr = pd.DataFrame(class_rows)
        cr.to_csv(f"{OUT}/best_peak_by_class.csv", index=False)
        print("\nbest peak per class (median R2 across genes):")
        print(cr.groupby(["timepoint", "cls"]).agg(genes=("gene", "nunique"), median_R2=("R2", "median"),
                                                   median_counts=("counts", "median"),
                                                   median_candidates=("n_candidates", "median")).to_string())
        win = cr.loc[cr.groupby(["timepoint", "gene"]).R2.idxmax()]
        print("\nwhich class wins per gene:")
        print(win.groupby(["timepoint", "cls"]).size().to_string())
    if allvals:
        big = pd.concat(allvals)
        big.to_csv(f"{OUT}/fitted_values_all.csv", index=False)
        sl = (big.dropna(subset=["slope"]).groupby(["timepoint", "modality", "feature", "domain"])
              [["slope", "intercept"]].first().reset_index())
        sl.to_csv(f"{OUT}/domain_slopes.csv", index=False)
        print(sl.to_string(index=False))
    print("written", OUT)


if __name__ == "__main__":
    main()
