"""Tables behind the new Fig 4 (fit rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3, rho_Isoform domains at the elbow K = 3).

Elbow K (2026-10-14): rho_Isoform has 3 domains (0 subcortical & white matter, 1 amygdala / olfactory cortex, 2 cortex).
The switch boundary is the two highest-rho domains (I.boundary); file and column names keep the K = 4 suffixes
'_2' / '_3' = amygdala/olfactory side / cortex side.

1 domain_slopes.csv   per non-major isoform of the 1,071 SV genes x rho_Isoform domain, fitted INSIDE the domain:
      usage       logit u = a + b * rho      (binomial: isoform reads of gene reads)
      expression  log mu  = log lib + a + b * rho  (Poisson: the gene's RNA counts, library = all gene counts)
    Gate = quasi-likelihood LRT (LR divided by the Pearson dispersion of the sloped fit, chi2 df 1), BH within domain.
    Effect sizes over the domain's rho range (2nd-98th pct): usage change du (probability scale) and expression
    log2 fold change. Calls:  usage sloped  = q < 0.05 and |du| >= 0.15
                              expr flat     = q >= 0.05 or |log2FC| < 0.5
    'isoform-only' = usage sloped and expression flat in the same domain.
2 switch_23.csv   the domain-2 / domain-3 boundary of rho_Isoform (adjacent intervals of rho): per isoform, fitted usage at
    the boundary from each side (jump), within-domain gradients (du/drho), pooled usage in 2 and 3; switch = dominant
    isoform flips (usage on opposite sides of 0.5) or |pooled change| >= 0.2 with both domains >= 100 gene reads.
    Top switches: exact Kitagawa split 2 -> 3 into composition + within-cell-type (cell bootstrap CI), and usage in
    neurons only on each side.
Usage: python scripts/rtt_isoform_sv_fig4_prep.py
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtt_isoform_sv_isodepth as I
import merfish_isoform_deep as MD

OUT = I.OUT; Q, DU, LFC, SW, NTOP = 0.05, 0.15, 0.5, 0.2, 30


def fit_glm(A, N, x, fam, off=None, iters=30):
    """batched 1-predictor GLM by IRLS. fam 'binom': A successes of N trials; 'pois': A counts with log offset `off`.
    Returns coefficients (I, 2), log-lik of the sloped and flat fits, Pearson dispersion of the sloped fit (>= 1)"""
    I_ = A.shape[1]; o = np.zeros(len(x)) if off is None else off
    def run(Xm):
        B = np.zeros((I_, Xm.shape[1]))
        if fam == "binom":
            p0 = np.clip(A.sum(0) / np.maximum(N.sum(0), 1), 1e-5, 1 - 1e-5); B[:, 0] = np.log(p0 / (1 - p0))
        else:
            B[:, 0] = np.log(np.maximum(A.sum(0), 0.5) / np.exp(o).sum())
        for _ in range(iters):
            eta = Xm @ B.T + o[:, None]
            if fam == "binom":
                mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); W = N * mu * (1 - mu)
                z = eta - o[:, None] + (A - N * mu) / np.maximum(W, 1e-9)
            else:
                mu = np.exp(np.clip(eta, -30, 30)); W = mu
                z = eta - o[:, None] + (A - mu) / np.maximum(mu, 1e-9)
            XtWX = np.einsum("bp,bi,bq->ipq", Xm, W, Xm) + 1e-8 * np.eye(Xm.shape[1])
            B = np.linalg.solve(XtWX, np.einsum("bp,bi->ip", Xm, W * z)[..., None])[..., 0]
        eta = Xm @ B.T + o[:, None]
        if fam == "binom":
            mu = np.clip(1 / (1 + np.exp(-np.clip(eta, -30, 30))), 1e-9, 1 - 1e-9)
            ll = (A * np.log(mu) + (N - A) * np.log(1 - mu)).sum(0)
            pr = np.where(N > 0, (A - N * mu) ** 2 / np.maximum(N * mu * (1 - mu), 1e-9), 0)
        else:
            mu = np.exp(np.clip(eta, -30, 30)); ll = (A * np.log(np.maximum(mu, 1e-300)) - mu).sum(0)
            pr = (A - mu) ** 2 / np.maximum(mu, 1e-9)
        return B, ll, pr
    X = np.c_[np.ones_like(x), x]
    B1, ll1, pr = run(X); _, ll0, _ = run(X[:, :1])
    nobs = (N > 0).sum(0) if fam == "binom" else np.full(I_, len(x))
    phi = np.maximum(pr.sum(0) / np.maximum(nobs - 2, 1), 1.0)
    return B1, ll1, ll0, phi


def main():
    C, B = I.load(); labs, desc = I.domains(B); lab = labs["Isoform"]; A, N, rows = I.build(B, C)
    d = B["d"]["Isoform"]; gi = pd.Index(B["genes"])
    gcol = gi.get_indexer([r[0] for r in rows]); has = gcol >= 0
    E = np.zeros_like(A); E[:, has] = B["Xg"][:, gcol[has]].toarray()
    out = []
    for k in range(lab.max() + 1):
        m = lab == k; x = d[m]; lo, hi = np.percentile(x, [2, 98]); xc = x - x.mean()
        bu, l1u, l0u, phu = fit_glm(A[m], N[m], xc, "binom")
        be, l1e, l0e, phe = fit_glm(E[m], None, xc, "pois", off=np.log(B["lib"][m]))
        pu = chi2.sf(np.maximum(2 * (l1u - l0u), 0) / phu, 1); pe = chi2.sf(np.maximum(2 * (l1e - l0e), 0) / phe, 1)
        sig = lambda v: 1 / (1 + np.exp(-v))
        du = sig(bu[:, 0] + bu[:, 1] * (hi - x.mean())) - sig(bu[:, 0] + bu[:, 1] * (lo - x.mean()))
        T = pd.DataFrame(dict(gene=[r[0] for r in rows], isoform=[r[1] for r in rows], domain=k, gene_reads=N[m].sum(0),
                              usage=A[m].sum(0) / np.maximum(N[m].sum(0), 1), usage_slope=bu[:, 1], usage_du=du, usage_p=pu,
                              usage_phi=phu, expr_counts=E[m].sum(0), expr_slope=be[:, 1],
                              expr_log2fc=be[:, 1] * (hi - lo) / np.log(2), expr_p=pe, expr_phi=phe, gene_on_panel=has))
        ok = T.gene_reads >= 50
        T["usage_q"] = np.nan; T.loc[ok, "usage_q"] = multipletests(T.usage_p[ok], method="fdr_bh")[1]
        okE = ok & T.gene_on_panel & (T.expr_counts >= 20)
        T["expr_q"] = np.nan; T.loc[okE, "expr_q"] = multipletests(T.expr_p[okE], method="fdr_bh")[1]
        T["usage_sloped"] = (T.usage_q < Q) & (T.usage_du.abs() >= DU)
        T["expr_flat"] = okE & ((T.expr_q >= Q) | (T.expr_log2fc.abs() < LFC))
        T["isoform_only"] = T.usage_sloped & T.expr_flat
        out.append(T)
    T = pd.concat(out); T.to_csv(f"{OUT}/domain_slopes.csv", index=False)
    s = T[T.gene_reads >= 50].groupby("domain").agg(tested=("isoform", "size"), usage_sloped=("usage_sloped", "sum"),
                                                    expr_flat=("expr_flat", "sum"), isoform_only=("isoform_only", "sum"))
    print(s.to_string(), flush=True)
    print(T[T.isoform_only].sort_values("usage_du", key=abs, ascending=False).head(20)[
        ["gene", "isoform", "domain", "gene_reads", "usage", "usage_du", "usage_q", "expr_log2fc", "expr_q"]].round(4).to_string(index=False))
    # --- 2 / 3 boundary
    k2, k3 = I.boundary(lab)                                      # amygdala/olfactory | cortex (1|2 at elbow K = 3)
    m2, m3 = lab == k2, lab == k3; bnd = (d[m2].max() + d[m3].min()) / 2
    sig = lambda v: 1 / (1 + np.exp(-v)); W = []
    T2, T3 = T[T.domain == k2].reset_index(drop=True), T[T.domain == k3].reset_index(drop=True)
    b2, *_ = fit_glm(A[m2], N[m2], d[m2] - d[m2].mean(), "binom"); b3, *_ = fit_glm(A[m3], N[m3], d[m3] - d[m3].mean(), "binom")
    u2b, u3b = sig(b2[:, 0] + b2[:, 1] * (bnd - d[m2].mean())), sig(b3[:, 0] + b3[:, 1] * (bnd - d[m3].mean()))
    S = pd.DataFrame(dict(gene=T2.gene, isoform=T2.isoform, reads_2=N[m2].sum(0), reads_3=N[m3].sum(0), usage_2=T2.usage, usage_3=T3.usage,
                          change=T3.usage - T2.usage, u_boundary_2=u2b, u_boundary_3=u3b, jump=u3b - u2b,
                          grad_2=b2[:, 1] * u2b * (1 - u2b), grad_3=b3[:, 1] * u3b * (1 - u3b), q_2=T2.usage_q, q_3=T3.usage_q,
                          expr_log2fc_23=np.log2((E[m3].sum(0) / B["lib"][m3].sum() + 1e-9) / (E[m2].sum(0) / B["lib"][m2].sum() + 1e-9)),
                          gene_on_panel=has))
    enough = (S.reads_2 >= 100) & (S.reads_3 >= 100)
    S["flip"] = enough & (np.sign(S.usage_2 - 0.5) != np.sign(S.usage_3 - 0.5))
    S["switch"] = enough & (S.flip | (S.change.abs() >= SW))
    S = S.sort_values("change", key=abs, ascending=False)
    # composition vs within for the top switches + neuron-only usage
    rng = np.random.default_rng(0); MD.NBOOT = 300
    dom = np.full(len(C["ct"]), -1); ok = C["cbin"] >= 0; dom[ok] = lab[C["cbin"][ok]]
    extra = []
    for _, r in S[S.switch].head(NTOP).iterrows():
        a, tot = MD.usage_counts(C, r.gene, r.isoform)
        res, tt = MD.decompose(a, tot, C["ct"], dom, k2, k3, rng)
        nm = C["cls"] == "neuron"
        un = [a[nm & (dom == k)].sum() / max(tot[nm & (dom == k)].sum(), 1) for k in (k2, k3)]
        rn = [int(tot[nm & (dom == k)].sum()) for k in (k2, k3)]
        extra.append(dict(isoform=r.isoform, composition=res["comp"], within=res["within"], within_ci_lo=res["within_ci"][0],
                          within_ci_hi=res["within_ci"][1], within_frac=res["within"] / res["delta"] if abs(res["delta"]) > 1e-9 else np.nan,
                          neuron_usage_2=un[0], neuron_usage_3=un[1], neuron_reads_2=rn[0], neuron_reads_3=rn[1],
                          types_sig=";".join(t["type"] for t in tt if t["ci_lo"] > 0 or t["ci_hi"] < 0)))
    S = S.merge(pd.DataFrame(extra), on="isoform", how="left"); S.to_csv(f"{OUT}/switch_23.csv", index=False)
    print(f"\n{k2}|{k3} boundary at rho_Iso {bnd:.2f}; isoforms with >= 100 reads both sides: {int(enough.sum())}; switches: "
          f"{int(S.switch.sum())} (dominant flips {int(S.flip.sum())})", flush=True)
    print(S[S.switch].head(NTOP)[["gene", "isoform", "usage_2", "usage_3", "change", "jump", "grad_2", "grad_3", "expr_log2fc_23",
                                  "within_frac", "within_ci_lo", "within_ci_hi", "neuron_usage_2", "neuron_usage_3"]].round(3).to_string(index=False))
    json.dump(dict(domains=desc, boundary_23=float(bnd), boundary_domains=[k2, k3], per_domain=s.reset_index().to_dict("records"),
                   n_switch=int(S.switch.sum()), n_flip=int(S.flip.sum())), open(f"{OUT}/fig4_prep_summary.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
