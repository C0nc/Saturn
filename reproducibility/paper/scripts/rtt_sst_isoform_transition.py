"""Isoform-switch state transition inside ONE cell type: Sst interneurons across rho_Isoform (section 5, SV-PCA fit).

Sst interneurons are one transcriptomic type present on both sides of the amygdala/olfactory | cortex boundary
(domains 2|3 at K = 4, 1|2 at the elbow K = 3; I.boundary) and carry the most within-type switches there (Kcnab3 0.31 -> 0.61, Ubr1 0.38 -> 0.10).
Adult tissue: the 'transition' is a spatial continuum of states within the type, not a measured time course.

Steps (each in the env that has the package):
  prep      (spatialmultivelo) Sst cells in fitted bins -> sst_cells.h5ad: gene counts, rho_Iso / rho_RNA / domain of the
            cell's bin, xy; isoform counts per cell and pooled over its 10 nearest NON-Sst neurons (spillover control)
  scvi      (tutorial_lpc)     scVI on the Sst cells only (1,000 HVG seurat_v3 on counts, 10 latent, seed 42) -> X_scvi
  cellrank  (stories)          RNA STATE SCORE t = P(cortex state | scVI latent): logistic regression amygdala/olfactory
            (domain 2) vs cortex (domain 3) Sst cells, 5-fold cross-validated for those cells (out-of-fold), model on all
            2/3 cells for the rest. (An unsupervised ρ-directed kernel gave compressed fates, r 0.38 with rho; RNA-only DPT
            r 0.15; yet the RNA state separates 2 vs 3 at CV AUC 0.93.) CellRank 2: kernel = 0.8 PseudotimeKernel(t) +
            0.2 ConnectivityKernel on the scVI kNN; GPCCA 3 macrostates (amygdala-like, intermediate, cortex-like).
            Transitional cells: 0.25 <= t <= 0.75.
  dynamics  (spatialmultivelo) per non-major isoform of the SV genes with >= 80 reads in Sst cells:
            binomial GLM usage ~ t at cell level; null = t permuted across cells (500x) -> p_perm, BH.
            Within-domain test: usage ~ t + domain, t permuted WITHIN domain (500x) -> does usage follow the RNA state
            beyond the cell's location?  Switch point (Isosceles-style pseudobulk): cells in 10 quantile windows of t, 4-parameter sigmoid
            u(t) = lo + (hi - lo) / (1 + exp(-k (t - t0))) fitted by binomial ML -> t0 (where usage changes fastest).
            Expression timing: the gene's CP10k in the same windows, 4PL in log CP10k -> t0_expr; is usage switching
            before or after expression?  Spillover: usage in the Sst cells vs in their non-Sst neuron neighbours
            (window-level r) and Sst / neighbour expression ratio; flagged when ratio < 0.3.
Outputs results/real_saturn/rtt_sst_transition/.
Usage: python scripts/rtt_sst_isoform_transition.py {prep|scvi|cellrank|dynamics}
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

R = "<workspace>/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/rtt_sst_transition"
TYPE, NNB, MIN_ISO_READS, NPERM, NWIN, SEED = "Inhibitory_Sst", 10, 80, 500, 10, 42


def prep():
    import anndata as ad
    from scipy.spatial import cKDTree
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rtt_isoform_sv_isodepth as I
    os.makedirs(OUT, exist_ok=True)
    C, B = I.load(); lab, _ = I.domains(B)
    sel = np.where((C["ct"] == TYPE) & (C["cbin"] >= 0))[0]
    nbr_pool = np.where((C["cls"] == "neuron") & (C["ct"] != TYPE))[0]
    nb = nbr_pool[cKDTree(C["xy"][nbr_pool]).query(C["xy"][sel], k=NNB)[1]]
    Gn = sp.csr_matrix((np.ones(nb.size), (np.repeat(np.arange(len(sel)), NNB), nb.ravel())), shape=(len(sel), len(C["ct"])))
    b = C["cbin"][sel]
    a = ad.AnnData(X=C["Xg"][sel].astype(np.float32), var=pd.DataFrame(index=C["genes"]))
    a.obs_names = [f"sst_{i}" for i in sel]
    a.obs["rho_iso"], a.obs["rho_rna"] = B["d"]["Isoform"][b], B["d"]["RNA"][b]
    a.obs["domain_iso"], a.obs["domain_rna"] = lab["Isoform"][b], lab["RNA"][b]
    a.obs["x"], a.obs["y"] = C["xy"][sel, 0], C["xy"][sel, 1]; a.obsm["spatial"] = C["xy"][sel]
    a.layers["counts"] = a.X.copy(); a.var_names_make_unique(); a.write_h5ad(f"{OUT}/sst_cells.h5ad")
    sp.save_npz(f"{OUT}/sst_iso.npz", sp.csr_matrix(C["Xi"][sel])); sp.save_npz(f"{OUT}/sst_iso_nbr.npz", sp.csr_matrix(Gn @ C["Xi"]))
    sp.save_npz(f"{OUT}/sst_gene_nbr.npz", sp.csr_matrix(Gn @ C["Xg"]))
    pd.DataFrame(dict(iso=C["iso"], gene=C["iso_gene"])).to_csv(f"{OUT}/iso_meta.csv", index=False)
    print(f"{len(sel)} {TYPE} cells; per isoform-domain: {pd.Series(a.obs.domain_iso).value_counts().sort_index().to_dict()}")


def run_scvi():
    import anndata as ad, scanpy as sc, scvi
    a = ad.read_h5ad(f"{OUT}/sst_cells.h5ad"); sc.pp.filter_genes(a, min_cells=10)
    sc.pp.highly_variable_genes(a, n_top_genes=1000, flavor="seurat_v3", layer="counts")
    h = a[:, a.var.highly_variable].copy(); scvi.settings.seed = SEED
    scvi.model.SCVI.setup_anndata(h, layer="counts"); m = scvi.model.SCVI(h, n_latent=10)
    m.train(max_epochs=400, early_stopping=True)
    np.save(f"{OUT}/X_scvi.npy", m.get_latent_representation()); print("scVI latent saved", h.shape)


def run_cellrank():
    import anndata as ad, scanpy as sc, cellrank as cr
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    a = ad.read_h5ad(f"{OUT}/sst_cells.h5ad"); Z = np.load(f"{OUT}/X_scvi.npy"); a.obsm["X_scvi"] = Z
    sc.pp.neighbors(a, use_rep="X_scvi", n_neighbors=15, random_state=0); sc.tl.umap(a, random_state=0)
    r = a.obs.rho_iso.values; dom = a.obs.domain_iso.values; ka, kc = int(dom.max()) - 1, int(dom.max()); m = np.isin(dom, [ka, kc]); y = (dom[m] == kc).astype(int)   # 1|2 at elbow K = 3
    clf = LogisticRegression(C=1.0, max_iter=2000)
    t = np.zeros(a.n_obs)
    t[m] = cross_val_predict(clf, Z[m], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    t[~m] = clf.fit(Z[m], y).predict_proba(Z[~m])[:, 1]
    a.obs["t_state"] = t
    sc.tl.diffmap(a); a.uns["iroot"] = int(np.argmin(r)); sc.tl.dpt(a)
    pk = cr.kernels.PseudotimeKernel(a, time_key="t_state").compute_transition_matrix(threshold_scheme="soft")
    ck = cr.kernels.ConnectivityKernel(a).compute_transition_matrix()
    g = cr.estimators.GPCCA(0.8 * pk + 0.2 * ck); g.compute_schur(n_components=10)
    g.compute_macrostates(n_states=3, cluster_key=None)
    ms = g.macrostates.astype(str); mt = pd.Series(t, index=a.obs_names).groupby(ms.values).mean().drop("nan", errors="ignore").sort_values()
    name = {mt.index[0]: "amygdala-like", mt.index[1]: "intermediate", mt.index[2]: "cortex-like"}
    a.obs["macrostate"] = [name.get(v, "unassigned") for v in ms.values]
    # every cell's most likely macrostate (GPCCA memberships), not only the 30 core cells
    mem = np.asarray(g.macrostates_memberships); cols = [str(c) for c in g.macrostates_memberships.names]
    a.obs["macro_member"] = [name[cols[i]] for i in mem.argmax(1)]
    a.obs["transitional"] = (t >= 0.25) & (t <= 0.75)
    rep = dict(n_cells=a.n_obs, cv_auc_2v3=round(float(roc_auc_score(y, t[m])), 3), t_vs_rho=round(float(spearmanr(t, r)[0]), 3),
               dpt_vs_rho=round(float(spearmanr(a.obs.dpt_pseudotime, r)[0]), 3), n_transitional=int(a.obs.transitional.sum()),
               t_by_domain=a.obs.groupby("domain_iso")["t_state"].mean().round(3).to_dict(),
               macro_member_by_domain=pd.crosstab(a.obs.domain_iso, a.obs.macro_member).to_dict(),
               macro_mean_t={name[k]: round(float(v), 3) for k, v in mt.items()})
    a.obs[["rho_iso", "rho_rna", "domain_iso", "x", "y", "dpt_pseudotime", "t_state", "macrostate", "macro_member", "transitional"]].assign(
        t_fate=t, umap1=a.obsm["X_umap"][:, 0], umap2=a.obsm["X_umap"][:, 1]).to_csv(f"{OUT}/sst_states.csv")
    json.dump(rep, open(f"{OUT}/cellrank_summary.json", "w"), indent=1, default=str); print(json.dumps(rep, indent=1, default=str))


def sigmoid4(A, N, t, w):
    """4-parameter sigmoid by binomial ML on window-pooled counts; returns lo, hi, k, t0"""
    from scipy.optimize import minimize
    q = np.quantile(t, np.linspace(0, 1, w + 1)); bi = np.clip(np.digitize(t, q[1:-1]), 0, w - 1)
    tw = np.array([t[bi == b].mean() for b in range(w)]); aw = np.array([A[bi == b].sum() for b in range(w)]); nw = np.array([N[bi == b].sum() for b in range(w)])
    uw = aw / np.maximum(nw, 1)
    def nll(p):
        lo, hi = 1 / (1 + np.exp(-p[0])), 1 / (1 + np.exp(-p[1])); u = lo + (hi - lo) / (1 + np.exp(-np.exp(p[2]) * (tw - p[3])))
        u = np.clip(u, 1e-6, 1 - 1e-6); return -(aw * np.log(u) + (nw - aw) * np.log(1 - u)).sum()
    best = None
    for t0 in (0.25, 0.5, 0.75):
        r = minimize(nll, [np.log((uw[0] + .02) / (1.02 - uw[0])), np.log((uw[-1] + .02) / (1.02 - uw[-1])), np.log(8), t0], method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-5, fatol=1e-7))
        if best is None or r.fun < best.fun: best = r
    p = best.x; return 1 / (1 + np.exp(-p[0])), 1 / (1 + np.exp(-p[1])), float(np.exp(p[2])), float(np.clip(p[3], -0.2, 1.2)), tw, uw, nw


def expr_t0(y, lib, t, w):
    from scipy.optimize import curve_fit
    q = np.quantile(t, np.linspace(0, 1, w + 1)); bi = np.clip(np.digitize(t, q[1:-1]), 0, w - 1)
    tw = np.array([t[bi == b].mean() for b in range(w)]); ew = np.log1p(np.array([y[bi == b].sum() / lib[bi == b].sum() for b in range(w)]) * 1e4)
    f = lambda x, lo, hi, k, t0: lo + (hi - lo) / (1 + np.exp(-k * (x - t0)))
    try:
        p, _ = curve_fit(f, tw, ew, p0=[ew[0], ew[-1], 8, 0.5], bounds=([-10, -10, 0.1, -0.2], [20, 20, 200, 1.2]), maxfev=20000)
        return float(p[3]), float(p[1] - p[0]), tw, ew
    except Exception:
        return np.nan, float(ew[-1] - ew[0]), tw, ew


def dynamics():
    import anndata as ad
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from statsmodels.stats.multitest import multipletests
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rtt_isoform_sv_isodepth import irls
    a = ad.read_h5ad(f"{OUT}/sst_cells.h5ad"); S = pd.read_csv(f"{OUT}/sst_states.csv", index_col=0).loc[a.obs_names]
    Xi = sp.load_npz(f"{OUT}/sst_iso.npz").tocsc(); Xn = sp.load_npz(f"{OUT}/sst_iso_nbr.npz").tocsc(); Gn = sp.load_npz(f"{OUT}/sst_gene_nbr.npz").tocsc()
    meta = pd.read_csv(f"{OUT}/iso_meta.csv"); sv = set(pd.read_csv(f"{R}/rtt_isoform_variants/whole_sv/sv_results.csv", index_col=0).query("ir_padj < 0.05").index)
    t = S.t_state.values; Xg = sp.csc_matrix(a.layers["counts"]); lib = np.asarray(Xg.sum(1)).ravel().clip(1)
    gidx = pd.Index(a.var_names); libn = np.asarray(Gn.sum(1)).ravel().clip(1)
    tot = np.asarray(Xi.sum(0)).ravel(); rows = []
    for g, idx in meta.groupby("gene").groups.items():
        if g not in sv: continue
        idx = np.asarray(idx); idx = idx[np.argsort(-tot[idx])]
        if tot[idx].sum() < 150: continue
        for j in idx[1:]:
            if tot[j] >= MIN_ISO_READS and tot[j] / tot[idx].sum() >= 0.05: rows.append((g, meta.iso[j], j, idx))
    A = np.column_stack([Xi[:, j].toarray().ravel() for _, _, j, _ in rows]).astype(float)
    N = np.column_stack([np.asarray(Xi[:, ix].sum(1)).ravel() for *_, ix in rows]).astype(float)
    one = np.ones(len(t)); tz = (t - t.mean()) / t.std()
    b, dev1 = irls(A, N, np.c_[one, tz]); _, dev0 = irls(A, N, one[:, None])
    rng = np.random.default_rng(SEED); nul = []
    for k in range(NPERM):
        tp = rng.permutation(tz); nul.append(irls(A, N, np.c_[one, tp])[0][:, 1])
    nul = np.array(nul); p = ((np.abs(nul) >= np.abs(b[:, 1])).sum(0) + 1) / (NPERM + 1)
    T = pd.DataFrame(dict(gene=[r[0] for r in rows], isoform=[r[1] for r in rows], reads=A.sum(0), gene_reads=N.sum(0),
                          usage=A.sum(0) / N.sum(0), slope_per_sd=b[:, 1], dev_expl=1 - dev1 / dev0, p_perm=p))
    T["q"] = multipletests(T.p_perm, method="fdr_bh")[1]
    dom = S.domain_iso.values; Dm = np.c_[[(dom == k).astype(float) for k in sorted(set(dom))[1:]]].T
    bw, _ = irls(A, N, np.c_[one, tz, Dm]); nulw = []
    for k in range(NPERM):
        tp = tz.copy()
        for d_ in set(dom): ii = np.where(dom == d_)[0]; tp[ii] = rng.permutation(tz[ii])
        nulw.append(irls(A, N, np.c_[one, tp, Dm])[0][:, 1])
    nulw = np.array(nulw); T["slope_within_dom"] = bw[:, 1]
    T["p_within_dom"] = ((np.abs(nulw) >= np.abs(bw[:, 1])).sum(0) + 1) / (NPERM + 1)
    T["q_within_dom"] = multipletests(T.p_within_dom, method="fdr_bh")[1]
    out = []
    for i, r in T.iterrows():
        lo, hi, k_, t0, tw, uw, nw = sigmoid4(A[:, i], N[:, i], t, NWIN)
        gj = gidx.get_indexer([r.gene])[0]
        if gj >= 0:
            y = Xg[:, gj].toarray().ravel(); e0, de, _, _ = expr_t0(y, lib, t, NWIN)
            ratio = (y.sum() / lib.sum()) / max(Gn[:, gj].toarray().ravel().sum() / libn.sum(), 1e-12)
        else:
            e0, de, ratio = np.nan, np.nan, np.nan
        an, nn = Xn[:, rows[i][2]].toarray().ravel(), np.asarray(Xn[:, rows[i][3]].sum(1)).ravel()
        _, _, _, _, _, unw, _ = sigmoid4(an, nn, t, NWIN)
        out.append(dict(u_lo=lo, u_hi=hi, steep=k_, t0=t0, du=hi - lo, t0_expr=e0, expr_dlog=de, expr_ratio_sst_vs_nbr=ratio,
                        nbr_window_r=float(np.corrcoef(uw, unw)[0, 1]) if np.std(unw) > 0 else np.nan,
                        uw=";".join(f"{v:.3f}" for v in uw)))
    T = pd.concat([T, pd.DataFrame(out)], axis=1)
    T["spillover_flag"] = T.expr_ratio_sst_vs_nbr < 0.3
    T["switch"] = (T.q < 0.1) & (T.du.abs() >= 0.15) & ~T.spillover_flag
    T["state_coupled"] = T.switch & (T.q_within_dom < 0.1)
    T = T.sort_values("q"); T.to_csv(f"{OUT}/isoform_dynamics.csv", index=False)
    sw = T[T.switch].sort_values("t0")
    print(f"{len(T)} isoforms tested in {len(t)} Sst cells; q<0.1: {(T.q < 0.1).sum()}; switches (|du|>=0.15, no spillover flag): {len(sw)}")
    print(sw[["gene", "isoform", "reads", "slope_per_sd", "du", "t0", "t0_expr", "expr_dlog", "expr_ratio_sst_vs_nbr", "nbr_window_r", "q",
              "slope_within_dom", "q_within_dom"]].round(3).to_string(index=False))
    both = sw[np.isfinite(sw.t0_expr) & (sw.expr_dlog.abs() >= 0.3)]
    rep = dict(n_tested=len(T), n_q01=int((T.q < 0.1).sum()), n_switch=len(sw), n_spillover_flag=int(T.spillover_flag.sum()),
               n_state_coupled_within_domain=int(T.state_coupled.sum()),
               switch_t0_quartiles=np.round(np.percentile(sw.t0, [25, 50, 75]), 3).tolist() if len(sw) else None,
               usage_before_expression=int((both.t0 < both.t0_expr).sum()), usage_after_expression=int((both.t0 >= both.t0_expr).sum()))
    json.dump(rep, open(f"{OUT}/dynamics_summary.json", "w"), indent=1); print(json.dumps(rep, indent=1))
    # ---- diagnostic figure
    fig = plt.figure(figsize=(22, 13)); gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0]); sc_ = ax.scatter(S.x, S.y, c=S.t_state, cmap="viridis", s=8, lw=0); ax.set_aspect("equal"); ax.axis("off")
    plt.colorbar(sc_, ax=ax, shrink=0.6, label="RNA state score t (P cortex-like)"); ax.set_title(f"{len(S)} Sst interneurons in space")
    ax = fig.add_subplot(gs[0, 1]); cm = plt.get_cmap("tab10")
    for d_ in sorted(S.domain_iso.unique()):
        m = S.domain_iso == d_; ax.scatter(S.umap1[m], S.umap2[m], s=8, lw=0, color=cm(d_), label=f"rho_Iso domain {d_}")
    tr = S.transitional; ax.scatter(S.umap1[tr], S.umap2[tr], s=22, facecolor="none", edgecolor="k", lw=0.5, label="transitional (0.25-0.75)")
    ax.legend(fontsize=8); ax.set_title("scVI UMAP (Sst only)"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    if len(sw):
        H = np.array([[float(v) for v in s_.split(";")] for s_ in sw.uw]); Hz = (H - H.mean(1, keepdims=True)) / np.maximum(H.std(1, keepdims=True), 1e-9)
        im = ax.imshow(Hz, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_yticks(range(len(sw))); ax.set_yticklabels([f"{g} {i[-6:]}" for g, i in zip(sw.gene, sw.isoform)], fontsize=7)
        ax.set_xlabel("t window (amygdala state → cortex state)"); plt.colorbar(im, ax=ax, shrink=0.6, label="usage z"); ax.set_title("switching isoforms ordered by switch point t0")
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(both.t0_expr, both.t0, s=25, color="#3B6FB6")
    for _, r in both.iterrows(): ax.annotate(r.gene, (r.t0_expr, r.t0), fontsize=7)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6); ax.set_xlabel("expression switch point"); ax.set_ylabel("usage switch point")
    ax.set_title(f"usage before expression: {rep['usage_before_expression']}, after: {rep['usage_after_expression']}")
    for c, (_, r) in enumerate(sw.reindex(sw.du.abs().sort_values(ascending=False).index).head(4).iterrows()):
        ax = fig.add_subplot(gs[1, c]); i = T.index.get_loc(_) if False else None
        uw = np.array([float(v) for v in r.uw.split(";")]); tw = np.quantile(t, (np.arange(NWIN) + 0.5) / NWIN)
        ax.plot(tw, uw, "o", color="#3B6FB6"); xs = np.linspace(0, 1, 100)
        ax.plot(xs, r.u_lo + (r.u_hi - r.u_lo) / (1 + np.exp(-r.steep * (xs - r.t0))), color="#3B6FB6")
        ax.axvline(r.t0, color="#B64342", lw=0.8, ls="--"); ax.set_ylim(0, 1); ax.set_xlabel("RNA state score t"); ax.set_ylabel("usage in Sst cells")
        ax.set_title(f"{r.gene} {r.isoform[-6:]}: t0 {r.t0:.2f}, q {r.q:.3f}, within-domain q {r.q_within_dom:.3f}\nexpr Sst/nbr {r.expr_ratio_sst_vs_nbr:.2f}")
    plt.tight_layout(); plt.savefig(f"{OUT}/sst_transition.png", dpi=75); print("->", OUT)


if __name__ == "__main__":
    dict(prep=prep, scvi=run_scvi, cellrank=run_cellrank, dynamics=dynamics)[sys.argv[1]]()
