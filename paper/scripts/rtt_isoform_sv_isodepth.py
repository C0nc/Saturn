"""Isodepth analysis of the whole-slice SPLISOSM-filtered isoform fit (PCA on shrunk CLR ratios), section 5.

Fit: rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3 (100 um bins, RNA GLM-PCA + isoform PCA on 1,071 SV
genes; no restarts, lam_cross 0.3). Every segmented cell is mapped to its 100 um bin (same floor-grid code as the
fit), so it carries rho_RNA, rho_Isoform, both domain labels and its own cell type.

1 DOMAINS   GASTON DP segmentation (elbow K, domain_k.kdom: RNA 4, Isoform 3) of rho_RNA and rho_Isoform on their own targets; each domain described
            by its top cell types.
2 SCREEN    every non-major isoform j of the SV genes, bin level, binomial GLM with n_j successes of N_gene trials:
              M_iso   logit u = b0 + b_iso * rho_Iso + c * log depth
              M_rna   logit u = b0 + b_rna * rho_RNA + c * log depth
              M_both  logit u = b0 + b_iso|rna * rho_Iso + b_rna * rho_RNA + c * log depth
            rho z-scored, so slopes are per SD of rho. Spatial calibration: rho_Iso rotated 200 times (rho re-read at
            nearest bin) and the model refitted; z = b / SD(rotation slopes), p = 2 Phi(-|z|), BH.
            'isodepth-specific' = b_iso|rna significant (rho_Iso explains usage beyond rho_RNA and depth).
3 SPLIT     for the top isoforms by deviance explained along rho_Iso (>= 200 reads; nothing passes the rotation FDR): usage change between the rho_Iso domains of lowest and
            highest usage, split exactly into composition + within-cell-type (Kitagawa, merfish_isoform_deep.decompose,
            cell bootstrap CI).
4 PER TYPE  (kept from Fig 4g) usage as a function of rho inside each cell class: per-class rho sextiles with bootstrap
            CI; per-class binomial slope per SD of rho with the 200-rotation null. Run for the top 6 isoforms by deviance
            explained along rho_Iso AND
            the old examples (Tcf7l2 111651 on rho_RNA, Nkx2-2 109970 on rho_Isoform), now on the new fit.
Outputs results/real_saturn/rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3/isodepth/.
Usage: python scripts/rtt_isoform_sv_isodepth.py
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtt_isoform_variants as V
import merfish_isoform_deep as MD
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from sproxseq_validate2 import rotated_fields

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"
DS = "rtt_merfish_s5_iso100_sv_pca_shrunk"; FIT = f"{R}/{DS}_norestart_lc0.3"; OUT = f"{FIT}/isodepth"
SV = f"{R}/rtt_isoform_variants/whole_sv/sv_results.csv"
BIN, NROT, NQ, TOP_SPLIT, TOP_TYPE, FDR = 100.0, 200, 6, 40, 6, 0.05
OLD_EX = [("Tcf7l2", "RNA", "ENSMUST00000111651"), ("Nkx2-2", "Isoform", "ENSMUST00000109970")]
CLASSES = ["neuron", "oligodendrocyte", "astrocyte", "OPC"]
CCOL = {"neuron": "#3B6FB6", "oligodendrocyte": "#C2603F", "astrocyte": "#5E9E57", "OPC": "#9B6AB0", "all cells": "#1a1a1a"}


def cell_class(t):
    if t in ("Oligodendrocytes", "NFOLs"): return "oligodendrocyte"
    if t == "OPCs": return "OPC"
    if t == "Astrocytes": return "astrocyte"
    if t == "Microglia": return "microglia"
    if t in ("Endothelial", "Pericytes", "SMCs", "VLMCs"): return "vascular"
    if t in ("Choroid Plexus", "Ependymal"): return "choroid/ependymal"
    if t == "Other": return "other"
    return "neuron"


def irls(A, N, X, iters=25, ridge=1e-6):
    """batched binomial IRLS. A, N: (bins, I); X: (bins, p) shared or (I, bins, p). Returns coef (I, p), deviance (I,)"""
    Xb = np.broadcast_to(X, (A.shape[1],) + X.shape) if X.ndim == 2 else X
    p0 = np.clip(A.sum(0) / np.maximum(N.sum(0), 1), 1e-4, 1 - 1e-4)
    B = np.zeros((A.shape[1], Xb.shape[2])); B[:, 0] = np.log(p0 / (1 - p0))
    for _ in range(iters):
        eta = np.einsum("ibp,ip->bi", Xb, B); p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = N * p * (1 - p); z = eta + (A - N * p) / np.maximum(W, 1e-9)
        XtWX = np.einsum("ibp,bi,ibq->ipq", Xb, W, Xb) + ridge * np.eye(Xb.shape[2])
        XtWz = np.einsum("ibp,bi->ip", Xb, W * z)
        Bn = np.linalg.solve(XtWX, XtWz[..., None])[..., 0]
        if np.max(np.abs(Bn - B)) < 1e-7: B = Bn; break
        B = Bn
    eta = np.einsum("ibp,ip->bi", Xb, B); p = np.clip(1 / (1 + np.exp(-np.clip(eta, -30, 30))), 1e-12, 1 - 1e-12)
    dev = -2 * (A * np.log(p) + (N - A) * np.log(1 - p)).sum(0)
    return B, dev


def zs(v): return (v - v.mean()) / v.std()


def load():
    a, iso, ig, ok_i = V.load(5)
    z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True))
    ax = pd.read_csv(f"{FIT}/axes.csv", index_col=0); ax.index = ax.index.astype(str)
    assert list(ax.index) == [str(b) for b in z["barcodes"]]
    xy = a.obs[["global_x", "global_y"]].values.astype(float)
    key = np.floor(xy / BIN).astype(np.int64); code = (key[:, 0] * 1_000_000 + key[:, 1]).astype(str)
    cbin = pd.Index(ax.index).get_indexer(pd.Index(code))
    X = sp.csr_matrix(a.X); Xi = X[:, np.where(iso)[0][ok_i]].tocsc()
    ct = a.obs["cell_label"].astype(str).values
    C = dict(bin=code, cbin=cbin, xy=xy, ct=ct, cls=np.array([cell_class(t) for t in ct]), Xi=Xi,
             iso=np.array([str(v).split(".")[0] for v in a.var_names[iso][ok_i]]), iso_gene=np.array(ig[ok_i], str),
             Xg=X[:, np.where(~iso)[0]].tocsr(), genes=np.array(a.var_names[~iso], str))
    G = sp.csr_matrix((np.ones((cbin >= 0).sum()), (cbin[cbin >= 0], np.where(cbin >= 0)[0])), shape=(len(ax), len(ct)))
    B = dict(xy=ax[["x", "y"]].values, d={"RNA": ax.d_RNA.values, "Isoform": ax.d_Isoform.values}, z=z, G=G,
             Xi=sp.csc_matrix(G @ Xi), ct_frac=pd.DataFrame((G @ pd.get_dummies(ct).values.astype(float)) /
                                                             np.maximum(np.asarray(G.sum(1)), 1), columns=sorted(set(ct))))
    B["depth"] = np.asarray(B["Xi"].sum(1)).ravel()
    B["Xg"] = sp.csc_matrix(G @ X[:, np.where(~iso)[0]]); B["genes"] = np.array(a.var_names[~iso], str)
    B["lib"] = np.asarray(B["Xg"].sum(1)).ravel().clip(1)
    return C, B


def domains(B):
    lab, desc = {}, {}
    for m in ("RNA", "Isoform"):
        l = gaston_domains(B["d"][m], B["z"][f"Y_{m}"], L_=kdom(DS, m))         # elbow K: RNA 4, Isoform 3
        order = np.argsort([np.median(B["d"][m][l == k]) for k in range(l.max() + 1)])   # 0 = lowest rho
        l = np.argsort(order)[l]; lab[m] = l
        desc[m] = {int(k): dict(n_bins=int((l == k).sum()), rho_median=round(float(np.median(B["d"][m][l == k])), 2),
                                top_types=", ".join(f"{t} {v:.0%}" for t, v in B["ct_frac"][l == k].mean().nlargest(3).items()))
                   for k in range(l.max() + 1)}
    return lab, desc


def boundary(lab):
    """the switch boundary: the two highest-rho Isoform domains (amygdala / olfactory cortex | cortex);
    domains 2|3 at K = 4, 1|2 at the elbow K = 3"""
    return int(lab.max()) - 1, int(lab.max())


def build(B, C):
    sv = set(pd.read_csv(SV, index_col=0).query("ir_padj < @FDR").index)
    tot = np.asarray(B["Xi"].sum(0)).ravel(); rows = []
    for g, idx in pd.Series(np.arange(len(C["iso_gene"]))).groupby(C["iso_gene"]):
        if g not in sv: continue
        idx = idx.values; idx = idx[tot[idx] / tot[idx].sum() >= 0.05]
        if len(idx) < 2: continue
        idx = idx[np.argsort(-tot[idx])]
        for j in idx[1:]: rows.append((g, C["iso"][j], j, idx))
    A = np.column_stack([np.asarray(B["Xi"][:, j].todense()).ravel() for _, _, j, _ in rows]).astype(float)
    N = np.column_stack([np.asarray(B["Xi"][:, ix].sum(1)).ravel() for _, _, _, ix in rows]).astype(float)
    return A, N, rows


def screen(B, C):
    A, N, rows = build(B, C)
    if os.path.exists(f"{OUT}/usage_screen.csv"):                      # rotations are the slow part: reuse
        T = pd.read_csv(f"{OUT}/usage_screen.csv"); return T, A, N, rows
    ri, rr, ld = zs(B["d"]["Isoform"]), zs(B["d"]["RNA"]), zs(np.log(B["depth"].clip(1)))
    one = np.ones(len(ri))
    b_i, d_i = irls(A, N, np.c_[one, ri, ld]); b_r, d_r = irls(A, N, np.c_[one, rr, ld])
    b_b, d_b = irls(A, N, np.c_[one, ri, rr, ld]); _, d_0 = irls(A, N, np.c_[one, ld])
    rot = rotated_fields(B["xy"], B["d"]["Isoform"], NROT, np.random.default_rng(0))
    nul_i, nul_b = [], []
    for k, r in enumerate(rot):
        r = zs(r); nul_i.append(irls(A, N, np.c_[one, r, ld])[0][:, 1]); nul_b.append(irls(A, N, np.c_[one, r, rr, ld])[0][:, 1])
        if k % 50 == 0: print(f"  rotation {k}/{NROT}", flush=True)
    nul_i, nul_b = np.array(nul_i), np.array(nul_b)
    T = pd.DataFrame(dict(gene=[r[0] for r in rows], isoform=[r[1] for r in rows], reads=A.sum(0), gene_reads=N.sum(0),
                          usage=A.sum(0) / N.sum(0), slope_iso=b_i[:, 1], slope_rna=b_r[:, 1], slope_iso_given_rna=b_b[:, 1],
                          dev_expl_iso=1 - d_i / d_0, dev_expl_rna=1 - d_r / d_0, dev_expl_both=1 - d_b / d_0))
    T["z_iso"] = T.slope_iso / nul_i.std(0); T["z_iso_given_rna"] = T.slope_iso_given_rna / nul_b.std(0)
    T["p_rot_iso"] = ((np.abs(nul_i) >= np.abs(b_i[:, 1])).sum(0) + 1) / (NROT + 1)
    for c in ("iso", "iso_given_rna"):
        T[f"p_{c}"] = 2 * norm.sf(np.abs(T[f"z_{c}"])); T[f"padj_{c}"] = multipletests(T[f"p_{c}"], method="fdr_bh")[1]
    return T.sort_values("p_iso_given_rna"), A, N, rows


def top_by_effect(T, n):
    """isoforms ranked by deviance explained along rho_Iso (effect size); used when nothing passes the rotation FDR"""
    return T[T.reads >= 200].sort_values("dev_expl_iso", ascending=False).head(n)


def split(T, B, C, lab):
    rng = np.random.default_rng(0); MD.NBOOT = 300; out = []
    dom = np.full(len(C["ct"]), -1); ok = C["cbin"] >= 0; dom[ok] = lab["Isoform"][C["cbin"][ok]]
    top = top_by_effect(T, TOP_SPLIT)
    for _, r in top.iterrows():
        a, tot = MD.usage_counts(C, r.gene, r.isoform)
        u = [a[dom == k].sum() / max(tot[dom == k].sum(), 1) if tot[dom == k].sum() >= 50 else np.nan for k in range(lab["Isoform"].max() + 1)]
        if np.sum(np.isfinite(u)) < 2: continue
        lo, hi = int(np.nanargmin(u)), int(np.nanargmax(u))
        res, tt = MD.decompose(a, tot, C["ct"], dom, lo, hi, rng)
        wf = res["within"] / res["delta"] if abs(res["delta"]) > 1e-9 else np.nan
        sig_types = [t["type"] for t in tt if t["ci_lo"] > 0 or t["ci_hi"] < 0]
        out.append(dict(gene=r.gene, isoform=r.isoform, slope_iso=r.slope_iso, dom_from=lo, dom_to=hi, delta=res["delta"],
                        composition=res["comp"], within=res["within"], within_frac=wf, within_ci_lo=res["within_ci"][0],
                        within_ci_hi=res["within_ci"][1], within_sig=bool(res["within_ci"][0] > 0 or res["within_ci"][1] < 0),
                        types_with_sig_change=";".join(sig_types)))
    return pd.DataFrame(out)


def _slope(a, n, x):
    return irls(a[:, None].astype(float), n[:, None].astype(float), np.c_[np.ones_like(x), x])[0][0, 1]


def per_type(C, B, gene, iso, axis, rot, rng, ax=None):
    a, tot = MD.usage_counts(C, gene, iso); ok = (C["cbin"] >= 0) & (tot > 0)
    dbin = B["d"][axis]; sd = dbin.std(); st = {}
    for cls in CLASSES + ["all cells"]:
        idx = np.where(ok & ((C["cls"] == cls) if cls != "all cells" else True))[0]
        if tot[idx].sum() < 100: continue
        x = dbin[C["cbin"][idx]]; A_, N_ = a[idx].astype(float), tot[idx].astype(float)
        b1 = _slope(A_, N_, (x - x.mean()) / sd)
        Rn = np.stack([r[C["cbin"][idx]] for r in rot], 1); Rn = (Rn - Rn.mean(0)) / sd
        nl = irls(np.repeat(A_[:, None], len(rot), 1), np.repeat(N_[:, None], len(rot), 1),
                  np.stack([np.c_[np.ones(len(x)), Rn[:, k]] for k in range(len(rot))]))[0][:, 1]
        st[cls] = dict(n_cells=int(len(idx)), reads=int(N_.sum()), usage=round(float(A_.sum() / N_.sum()), 3),
                       slope_per_sd=round(float(b1), 3), p_rot=round(float(((np.abs(nl) >= abs(b1)).sum() + 1) / (len(rot) + 1)), 4))
        if ax is not None:
            q = np.quantile(x, np.linspace(0, 1, NQ + 1)); bi = np.clip(np.digitize(x, q[1:-1]), 0, NQ - 1); xb, yb, lo, hi = [], [], [], []
            for b in range(NQ):
                mb = np.where(bi == b)[0]
                if N_[mb].sum() < 10: continue
                bs = [A_[s].sum() / max(N_[s].sum(), 1) for s in (rng.choice(mb, len(mb)) for _ in range(200))]
                xb.append(x[mb].mean()); yb.append(A_[mb].sum() / N_[mb].sum()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            ax.fill_between(xb, lo, hi, color=CCOL[cls], alpha=0.13, lw=0)
            ax.plot(xb, yb, color=CCOL[cls], lw=1.8 if cls == "all cells" else 1.2, marker="o", ms=3,
                    label=f"{cls}: {b1:+.2f}/SD (p {st[cls]['p_rot']:.3f})")
    if ax is not None:
        ax.set_ylim(0, 1); ax.set_xlabel(f"rho_{axis}"); ax.set_ylabel("isoform usage"); ax.legend(fontsize=7, loc="best")
        ax.set_title(f"{gene} {iso[-6:]} along rho_{axis}", fontsize=10)
    return st


def main():
    os.makedirs(OUT, exist_ok=True)
    C, B = load(); print(f"{len(C['ct'])} cells, {(C['cbin'] >= 0).mean():.1%} in fitted bins; {len(B['xy'])} bins", flush=True)
    lab, desc = domains(B); print(json.dumps(desc, indent=1), flush=True)
    T, A, N, rows = screen(B, C); T.to_csv(f"{OUT}/usage_screen.csv", index=False)
    print("top by deviance explained along rho_Iso:\n" + top_by_effect(T, 15)[["gene", "isoform", "reads", "usage", "slope_iso",
          "slope_rna", "dev_expl_iso", "dev_expl_rna", "z_iso", "p_rot_iso", "p_iso_given_rna"]].round(4).to_string(index=False), flush=True)
    n_iso, n_spec = int((T.padj_iso < FDR).sum()), int((T.padj_iso_given_rna < FDR).sum())
    print(f"screen: {len(T)} isoforms; usage changes along rho_Iso (rotation-calibrated FDR<{FDR}): {n_iso}; "
          f"beyond rho_RNA + depth: {n_spec}", flush=True)
    print(T.head(15)[["gene", "isoform", "reads", "usage", "slope_iso", "slope_rna", "slope_iso_given_rna", "dev_expl_iso",
                      "dev_expl_rna", "padj_iso", "padj_iso_given_rna"]].round(4).to_string(index=False), flush=True)
    S = split(T, B, C, lab); S.to_csv(f"{OUT}/composition_vs_within.csv", index=False)
    print(f"split ({len(S)} top isoforms): median within-type share {S.within_frac.median():.2f}; within-type significant "
          f"{int(S.within_sig.sum())}/{len(S)}", flush=True)
    print(S.round(3).head(15).to_string(index=False), flush=True)
    # per-type curves: top isodepth-specific hits + old examples
    rng = np.random.default_rng(0)
    rots = {m: rotated_fields(B["xy"], B["d"][m], NROT, np.random.default_rng(1)) for m in ("RNA", "Isoform")}
    spec = top_by_effect(T, TOP_TYPE)
    ex = [(r.gene, "Isoform", r.isoform) for _, r in spec.iterrows()] + OLD_EX
    fig, Ax = plt.subplots(2, (len(ex) + 1) // 2, figsize=(5.2 * ((len(ex) + 1) // 2), 9)); PT = {}
    for axp, (g, axis, iso) in zip(Ax.ravel(), ex):
        PT[f"{g}|{iso}|{axis}"] = per_type(C, B, g, iso, axis, rots[axis], rng, axp)
    for axp in Ax.ravel()[len(ex):]: axp.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/per_type_along_rho.png", dpi=80)
    json.dump(PT, open(f"{OUT}/per_type_along_rho.json", "w"), indent=1)
    # overview figure: rho + domains, screen volcano, decomposition, top usage maps
    fig, Ax = plt.subplots(2, 4, figsize=(22, 10)); cm = plt.get_cmap("tab10")
    for j, m in enumerate(("RNA", "Isoform")):
        d = B["d"][m]; lo, hi = np.percentile(d, [1, 99])
        Ax[0, 2 * j].scatter(*B["xy"].T, c=d, s=4, lw=0, cmap="coolwarm", vmin=lo, vmax=hi); Ax[0, 2 * j].set_title(f"rho_{m}")
        for k in range(lab[m].max() + 1):
            mk = lab[m] == k; Ax[0, 2 * j + 1].scatter(*B["xy"][mk].T, s=4, lw=0, color=cm(k), label=f"{k}: {desc[m][k]['top_types'][:45]}")
        Ax[0, 2 * j + 1].legend(fontsize=6.5, markerscale=3, loc="lower left"); Ax[0, 2 * j + 1].set_title(f"{m} domains (0 = lowest rho)")
    for a_ in Ax[0]: a_.set_aspect("equal"); a_.axis("off")
    a_ = Ax[1, 0]; sig = T.padj_iso_given_rna < FDR
    a_.scatter(T.slope_iso, -np.log10(T.p_iso.clip(1e-300)), s=4, color="#b0b7bd", lw=0, label="all")
    a_.scatter(T.slope_iso[T.padj_iso < FDR], -np.log10(T.p_iso[T.padj_iso < FDR].clip(1e-300)), s=5, color="#3B6FB6", lw=0, label=f"along rho_Iso ({n_iso})")
    a_.scatter(T.slope_iso[sig], -np.log10(T.p_iso[sig].clip(1e-300)), s=9, color="#B64342", lw=0, label=f"beyond rho_RNA ({n_spec})")
    a_.set_xlabel("usage slope per SD of rho_Iso (logit)"); a_.set_ylabel("-log10 p (rotation-calibrated)"); a_.legend(fontsize=8)
    a_ = Ax[1, 1]; a_.scatter(T.dev_expl_rna, T.dev_expl_iso, s=4, lw=0, color="#7f8a94"); a_.plot([0, 0.6], [0, 0.6], "k--", lw=0.6)
    a_.set_xlabel("deviance explained, rho_RNA + depth"); a_.set_ylabel("deviance explained, rho_Iso + depth")
    a_.set_title(f"rho_Iso better for {(T.dev_expl_iso > T.dev_expl_rna).mean():.0%} of isoforms")
    a_ = Ax[1, 2]
    if len(S):
        v = S.within_frac.clip(-0.25, 1); x = np.random.default_rng(0).uniform(-0.25, 0.25, len(v))
        a_.scatter(x[~S.within_sig], v[~S.within_sig], s=12, color="#b0b7bd"); a_.scatter(x[S.within_sig], v[S.within_sig], s=12, color="#B64342")
        a_.axhline(S.within_frac.median(), color="k"); a_.set_xticks([]); a_.set_ylabel("within-cell-type share of usage change")
        a_.set_title(f"top {len(S)} rho_Iso switches: median within {S.within_frac.median():.2f}, sig {int(S.within_sig.sum())}")
    a_ = Ax[1, 3]; r0 = top_by_effect(T, 1).iloc[0]; j = [k for k, r in enumerate(rows) if r[1] == r0.isoform][0]
    u = np.where(N[:, j] >= 3, A[:, j] / np.maximum(N[:, j], 1), np.nan); okk = np.isfinite(u)
    a_.scatter(*B["xy"][~okk].T, s=3, color="#eeeeee", lw=0)
    a_.scatter(*B["xy"][okk].T, c=u[okk], s=4, lw=0, cmap="coolwarm", vmin=np.nanpercentile(u, 2), vmax=np.nanpercentile(u, 98))
    a_.set_aspect("equal"); a_.axis("off"); a_.set_title(f"top hit {r0.gene} {r0.isoform[-6:]} usage (raw bins)")
    plt.tight_layout(); plt.savefig(f"{OUT}/overview.png", dpi=75)
    json.dump(dict(domains=desc, screen=dict(n_isoforms=len(T), along_rho_iso=n_iso, beyond_rho_rna=n_spec,
                                               iso_better_than_rna=round(float((T.dev_expl_iso > T.dev_expl_rna).mean()), 3)),
                   split=dict(n=len(S), median_within=float(S.within_frac.median()) if len(S) else None,
                              n_within_sig=int(S.within_sig.sum()) if len(S) else 0), per_type=PT),
              open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(PT, indent=1)); print("->", OUT)


if __name__ == "__main__":
    main()
