"""PCA reductions of the FULL isoform-ratio vector (whole slice, 1,071 SPLISOSM SV genes) vs multinomial GLM-PCA.

Per gene g with isoforms j (same isoform set as rtt_isoform_multinomial.py: >= 5% share, major first), per bin s:
  shrunk   p_sgj = (n_sgj + K * pbar_gj) / (N_sg + K)      K = 2 pseudo-reads toward the section-wide ratio pbar
           (N = 0 -> pbar, i.e. mean filling). CLR within gene, block weight 1/sqrt(J_g - 1), centre, PCA 10.
  meanfill p_sgj = n_sgj / N_sg; bins with N_sg = 0 -> pbar (SPLISOSM nan_filling='mean'); CLR on (p + 1e-3),
           same weighting, PCA 10.
Compared with the multinomial GLM-PCA factors on: 6-NN r, plane R2, depth Spearman, best anatomy, 6-NN ceiling,
and canonical correlation-like agreement (R2 of each PCA's top 4 PCs regressed on the 10 GLM-PCA factors).
Outputs results/real_saturn/rtt_isoform_variants/whole_sv/ratio_pca_*.csv, ratio_pca_maps.png. No Saturn fit.
Usage: python scripts/rtt_isoform_ratio_pca.py
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from saturn.preprocess import scale_target
import rtt_isoform_variants as V
import rtt_isoform_binomial_glmpca as B

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/rtt_isoform_variants/whole_sv"
K, L = 2.0, 10


def r2(Y, P): return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def pca(Z, L=L):
    Z = Z - Z.mean(0); U, s, _ = np.linalg.svd(Z, full_matrices=False); return U[:, :L] * s[:L], s[:L] ** 2 / (s ** 2).sum()


def main():
    a, iso, ig, ok_i = V.load(5)
    Xg, Xi, bxy, codes, G, n, inv, kb = V.binned(a, iso, ok_i, 100.0)
    keep = n >= 5; xy = bxy[keep]; Xc = sp.csc_matrix(Xi[keep]).astype(float); genes = np.asarray(ig[ok_i], str)
    tot = np.asarray(Xc.sum(0)).ravel(); dep = np.asarray(Xi.sum(1)).ravel()[keep]
    sv = set(pd.read_csv(f"{OUT}/sv_results.csv", index_col=0).query("ir_padj < 0.05").index)
    Zs, Zm = [], []
    for g, idx in pd.Series(np.arange(len(genes))).groupby(genes):
        if g not in sv: continue
        idx = idx.values; idx = idx[tot[idx] / tot[idx].sum() >= 0.05]; idx = idx[np.argsort(-tot[idx])]
        C = Xc[:, idx].toarray(); N = C.sum(1, keepdims=True); pbar = C.sum(0) / C.sum(); w = 1 / np.sqrt(len(idx) - 1)
        P = (C + K * pbar) / (N + K); Zs.append(w * (np.log(P) - np.log(P).mean(1, keepdims=True)))
        P2 = np.where(N > 0, C / np.maximum(N, 1), pbar) + 1e-3; Zm.append(w * (np.log(P2) - np.log(P2).mean(1, keepdims=True)))
    Zs, Zm = np.hstack(Zs), np.hstack(Zm)
    Fm = np.load(f"{OUT}/multinomial_factors.npy")
    ann_all = V.annotations(a, iso, Xg, G, n, inv, kb)
    ann = {k: v[keep] for k, v in ann_all.items() if k.startswith("ann_frac_") or k.startswith("mk_")}
    nn6 = cKDTree(xy).query(xy, k=7)[1][:, 1:]; X1 = np.c_[np.ones(len(Fm)), Fm]
    out, maps = {}, {"multinomial GLM-PCA": Fm}
    for lab, Z in [("PCA, shrunk ratios (K=2) + CLR", Zs), ("PCA, raw ratios, mean-filled + CLR", Zm)]:
        F, ve = pca(Z); maps[lab] = F; Y = scale_target(F, "global")
        tag = "shrunk" if "shrunk" in lab else "meanfill"
        z = dict(np.load(f"{R}/targets_rtt_merfish_s5_iso100_sv_multinom.npz", allow_pickle=True))
        assert np.allclose(z["coords"], xy); z["Y_Isoform"] = Y.astype(np.float32)
        np.savez_compressed(f"{R}/targets_rtt_merfish_s5_iso100_sv_pca_{tag}.npz", **z)
        T = B.diagnostics(F, xy, dep, ann); T.to_csv(f"{OUT}/ratio_pca_{'shrunk' if 'shrunk' in lab else 'meanfill'}.csv", index=False)
        agree = [round(r2(F[:, [k]], X1 @ np.linalg.lstsq(X1, F[:, k], rcond=None)[0][:, None]), 3) for k in range(4)]
        out[lab] = dict(variance_share_PC1_5=np.round(ve[:5], 3).tolist(), ceiling_6NN=round(r2(Y, Y[nn6].mean(1)), 3),
                        PC1_4_R2_on_glmpca_factors=agree)
        print(f"\n{lab}  (input {Z.shape[1]} CLR coordinates)\n" + T.to_string(index=False), flush=True)
    Ym = scale_target(Fm, "global"); out["multinomial GLM-PCA"] = dict(ceiling_6NN=round(r2(Ym, Ym[nn6].mean(1)), 3))
    print(json.dumps(out, indent=1)); json.dump(out, open(f"{OUT}/ratio_pca_summary.json", "w"), indent=1)
    fig, A = plt.subplots(3, 4, figsize=(20, 15))
    for i, (lab, F) in enumerate(maps.items()):
        for k in range(4):
            v = F[:, k]; lo, hi = np.percentile(v, [2, 98]); ax = A[i, k]
            ax.scatter(xy[:, 0], xy[:, 1], c=v, s=4, lw=0, cmap="coolwarm", vmin=lo, vmax=hi); ax.set_aspect("equal"); ax.axis("off")
            X = np.c_[np.ones(len(xy)), xy]; b = np.linalg.lstsq(X, v, rcond=None)[0]
            ax.set_title(f"factor {k + 1} · spatial r {np.corrcoef(v, v[nn6].mean(1))[0, 1]:.2f} · plane {1 - ((v - X @ b) ** 2).sum() / ((v - v.mean()) ** 2).sum():.2f}", fontsize=11)
        A[i, 0].text(0, 1.12, lab, transform=A[i, 0].transAxes, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/ratio_pca_maps.png", dpi=70); print("->", OUT)


if __name__ == "__main__":
    main()
