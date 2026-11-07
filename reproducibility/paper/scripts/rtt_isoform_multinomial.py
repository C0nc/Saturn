"""Whole-slice isoform target: SPLISOSM spatial-variability filter + MULTINOMIAL GLM-PCA on full isoform vectors.

RT&T-MERFISH section 5, 100 um bins with >= MIN_CELLS cells (whole slice, no region restriction).
Isoforms per gene: those with >= MIN_ISO_SHARE of the gene's reads; gene kept if >= 2 remain and >= MIN_READS reads.
1. SPLISOSM (SplisosmNP, asymptotic null, BH): hsic-ir (isoform ratios vs space) = filter; hsic-gc for reference.
   SV usage genes: hsic-ir padj < FDR.
2. Multinomial GLM-PCA on the SV genes' full isoform count vectors (baseline-category logit, reference = the
   gene's major isoform):
       n[s, g, :] ~ Multinomial( N[s, g], softmax_j eta[s, g, j] ),  eta[., g, ref] = 0
       eta[s, g, j] = a_gj + b_gj * log depth_s + U_s . V_gj          L = 10, ridge on U, V (as the binomial)
   Fit by full-batch L-BFGS; factors SVD-rotated, then scale_target. Deviance explained vs a + b*log depth.
3. Comparison: binomial GLM-PCA (major share only) on the same SV genes.
Diagnostics per factor: 6-NN r, plane R2, Spearman with depth, best anatomy (cell-type fraction). No Saturn fit.
Outputs results/real_saturn/rtt_isoform_variants/whole_sv/ + targets_rtt_merfish_s5_iso100_sv_multinom.npz and
targets_rtt_merfish_s5_iso100_sv_binom.npz.
Usage: python scripts/rtt_isoform_multinomial.py
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "<workspace>/Saturn/src")
from splisosm import SplisosmNP
from prep_real2 import save, rna_glmpc_real
from saturn.preprocess import scale_target
import rtt_isoform_variants as V
import rtt_isoform_binomial_glmpca as B

R = "<workspace>/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/rtt_isoform_variants/whole_sv"
MIN_CELLS, MIN_READS, MIN_ISO_SHARE, FDR, L, RIDGE = 5, 100, 0.05, 0.05, 10, 1.0


def r2(Y, P): return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def multinomial_glmpca(C, cov, L=L, ridge=RIDGE, seed=0, iters=500):
    """C: (bins, genes, Jmax) counts, NaN where the gene has fewer isoforms; isoform 0 = reference."""
    torch.manual_seed(seed)
    mask = torch.as_tensor(~np.isnan(C[0])); c = torch.as_tensor(np.nan_to_num(C), dtype=torch.float64)   # mask: (genes, J)
    S, Gn, J = C.shape; free = mask.clone(); free[..., 0] = False          # non-reference isoforms
    p0 = (c.sum(0) + 0.5) * mask; a0 = torch.log(p0.clamp_min(1e-12) / p0[:, :1]).where(mask, torch.zeros((), dtype=torch.float64))
    a = a0.clone().requires_grad_(True); b = torch.zeros(Gn, J, dtype=torch.float64, requires_grad=True)
    U = (0.01 * torch.randn(S, L, dtype=torch.float64)).requires_grad_(True)
    Vv = (0.01 * torch.randn(Gn, J, L, dtype=torch.float64)).requires_grad_(True)
    x = torch.as_tensor((cov - cov.mean()) / cov.std(), dtype=torch.float64)[:, None, None]
    fm = free.to(torch.float64); neg = torch.tensor(-1e30, dtype=torch.float64)
    def eta(lowrank=True):
        e = a[None] + x * b[None] + (torch.einsum("sl,gjl->sgj", U, Vv) if lowrank else 0)
        return torch.where(mask[None].expand_as(e), e * fm[None], neg)    # ref isoform -> 0, missing -> -inf
    def nll(e):
        return -(c * torch.log_softmax(e, -1)).sum()
    opt = torch.optim.LBFGS([a, b, U, Vv], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = nll(eta()) + ridge * 0.5 * (U.pow(2).sum() + (Vv * fm[..., None]).pow(2).sum()); loss.backward(); return loss
    opt.step(closure)
    # null model: a + b * depth only (refit)
    a_n = a0.clone().requires_grad_(True); b_n = torch.zeros(Gn, J, dtype=torch.float64, requires_grad=True)
    opt0 = torch.optim.LBFGS([a_n, b_n], lr=1, max_iter=200, line_search_fn="strong_wolfe")
    def cl0():
        opt0.zero_grad(); e = a_n[None] + x * b_n[None]; e = torch.where(mask[None].expand_as(e), e * fm[None], neg)
        l = nll(e); l.backward(); return l
    opt0.step(cl0)
    with torch.no_grad():
        sat = (c * torch.log(torch.where(c > 0, c / c.sum(-1, keepdim=True).clamp_min(1), torch.ones_like(c)))).sum()
        e0 = a_n[None] + x * b_n[None]; e0 = torch.where(mask[None].expand_as(e0), e0 * fm[None], neg)
        dev, dev0 = 2 * (nll(eta()) + sat), 2 * (nll(e0) + sat)
        W = torch.einsum("sl,gjl->sgj", U, Vv * fm[..., None]).reshape(S, -1).numpy()
    Us, sv, _ = np.linalg.svd(W - W.mean(0), full_matrices=False)
    return Us[:, :L] * sv[:L], float(1 - dev / dev0), sv[:L] ** 2 / (sv ** 2).sum()


def splisosm_test(blocks, names, xy):
    m = SplisosmNP(); m.setup_data([torch.as_tensor(b_, dtype=torch.float32) for b_ in blocks], torch.as_tensor(xy, dtype=torch.float32), gene_names=names)
    res = {}
    for meth in ["hsic-ir", "hsic-gc"]:
        m.test_spatial_variability(method=meth, print_progress=False); res[meth] = m.get_formatted_test_results("sv").set_index("gene")
    T = pd.DataFrame(dict(n_iso=[b_.shape[1] for b_ in blocks], reads=[b_.sum() for b_ in blocks], ir_stat=res["hsic-ir"].statistic,
                          ir_padj=res["hsic-ir"].pvalue_adj, gc_padj=res["hsic-gc"].pvalue_adj), index=names).sort_values("ir_padj")
    T.to_csv(f"{OUT}/sv_results.csv"); return T


def main():
    os.makedirs(OUT, exist_ok=True)
    a, iso, ig, ok_i = V.load(5)
    Xg, Xi, bxy, codes, G, n, inv, kb = V.binned(a, iso, ok_i, 100.0)
    keep = n >= MIN_CELLS; xy = bxy[keep]; Xc = sp.csc_matrix(Xi[keep]).astype(float); genes = np.asarray(ig[ok_i], str)
    tot = np.asarray(Xc.sum(0)).ravel(); dep = np.asarray(Xi.sum(1)).ravel()[keep]
    blocks, names = [], []
    for g, idx in pd.Series(np.arange(len(genes))).groupby(genes):
        idx = idx.values
        if len(idx) < 2 or tot[idx].sum() < MIN_READS: continue
        idx = idx[tot[idx] / tot[idx].sum() >= MIN_ISO_SHARE]
        if len(idx) < 2: continue
        idx = idx[np.argsort(-tot[idx])]                                  # major isoform first (= reference)
        blocks.append(Xc[:, idx].toarray()); names.append(g)
    print(f"{keep.sum()} bins; {len(names)} genes tested", flush=True)
    if os.path.exists(f"{OUT}/sv_results.csv"):
        T = pd.read_csv(f"{OUT}/sv_results.csv", index_col=0)
    else:
        T = splisosm_test(blocks, names, xy)
    sv = set(T.index[T.ir_padj < FDR]); si = [i for i, g in enumerate(names) if g in sv]
    print(f"SV usage (HSIC-IR FDR<{FDR}): {len(sv)} of {len(T)}; SV expression (HSIC-GC): {(T.gc_padj < FDR).sum()}", flush=True)
    print(T.head(15).round(4).to_string(), flush=True)
    J = max(blocks[i].shape[1] for i in si); C = np.full((len(xy), len(si), J), np.nan)
    for k, i in enumerate(si): C[:, k, :blocks[i].shape[1]] = blocks[i]
    print(f"multinomial GLM-PCA: {len(si)} genes, {sum(blocks[i].shape[1] for i in si)} isoforms (max {J}/gene)", flush=True)
    Fm, de_m, ve_m = multinomial_glmpca(C, np.log(dep))
    Sb = np.column_stack([blocks[i][:, 0] for i in si]); Nb = np.column_stack([blocks[i].sum(1) for i in si])
    Fb, de_b, _ = B.binomial_glmpca(Sb, Nb, cov=np.log(dep))
    ann_all = V.annotations(a, iso, Xg, G, n, inv, kb)
    ann = {k: v[keep] for k, v in ann_all.items() if k.startswith("ann_frac_") or k.startswith("mk_")}
    nn6 = cKDTree(xy).query(xy, k=7)[1][:, 1:]; old = np.load(f"{R}/targets_rtt_merfish_s5_iso100_binom_depthcov_hv1000.npz", allow_pickle=True)["Y_Isoform"]
    Ym, Yb = scale_target(Fm, "global").astype(np.float32), scale_target(Fb, "global").astype(np.float32)
    rep = dict(bins=int(keep.sum()), genes_tested=len(T), sv_genes=len(si), sv_isoforms=int(sum(blocks[i].shape[1] for i in si)),
               deviance_explained=dict(multinomial=round(de_m, 3), binomial_major=round(de_b, 3)),
               ceiling_6NN=dict(multinomial_sv=round(r2(Ym, Ym[nn6].mean(1)), 3), binomial_sv=round(r2(Yb, Yb[nn6].mean(1)), 3),
                                previous_binomial_hv1000_unfiltered=round(r2(old, old[nn6].mean(1)), 3)),
               multinomial_factor_variance=np.round(ve_m[:5], 3).tolist())
    D = {}
    for lab, Y in [("multinomial_sv", Fm), ("binomial_sv", Fb)]:
        Tb = B.diagnostics(Y, xy, dep, ann); Tb.to_csv(f"{OUT}/diagnostics_{lab}.csv", index=False); D[lab] = Tb
        print(f"\n{lab}\n" + Tb.to_string(index=False), flush=True)
    json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1); print(json.dumps(rep, indent=1), flush=True)
    Yr, ir = rna_glmpc_real(Xg[keep], K=10)
    for lab, Y in [("multinom", Ym), ("binom", Yb)]:
        ex = dict(Y_RNA=Yr, Y_Isoform=Y, depth_RNA=ir["depth"], depth_Isoform=dep)
        ex.update({k: (v[keep] if hasattr(v, "__len__") and len(v) == len(codes) else v) for k, v in ann_all.items()})
        save(f"rtt_merfish_s5_iso100_sv_{lab}", xy, [str(u) for u in codes[keep]], ["RNA", "Isoform"], ex)
    np.save(f"{OUT}/multinomial_factors.npy", Fm)


if __name__ == "__main__":
    main()
