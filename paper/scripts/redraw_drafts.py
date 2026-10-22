"""Redraw real-data drafts from saved axes (no refitting).

Changes vs the first drafts:
  * axis maps coloured by RANK (percentile), so one extreme region cannot wash out
    the colour scale of the rest of the section;
  * per-axis kurtosis shown next to seed stability. Kurtosis > ~6 marks a
    DOMAIN-LIKE axis (range concentrated in one compartment) rather than a
    tissue-wide gradient -- interpret it as a boundary, not a gradient;
  * annotations / markers re-read from the updated targets (E11 WSNN clusters,
    Xenium protein markers, Xenium RNA markers that exist in its 396-gene panel).
Writes results/real_saturn/<name>/review_<name>.png/.svg and updates summary.json.
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import kurtosis, rankdata
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/sim")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from metrics_v2 import spearman
from run_real2 import eta2, KEEP, MAPANN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})
R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"
TITLE = {"lpc10": "LPC10 whole brain (RNA + ATAC)", "melanoma": "Melanoma in-tissue (RNA + ATAC)",
         "e11_hic": "E11 embryo (RNA + Hi-C A/B)", "e11_meth": "E11 embryo (RNA + DNA methylation)",
         "xenium_rcc": "Xenium RCC, 100 um bins (RNA + protein)", "rcc_tcr_bin": "RCC Slide-GoTags, 200 um bins (RNA + TCR)",
         "redcat": "REDCAT lymphoma Fig3, 50 px bins (Raman SRS + autofluorescence)",
         "lpc10_sel": "LPC10 curated cortex + lesion-adjacent (RNA + ATAC)",
         "melanoma_sel": "Melanoma curated tumour domain (RNA + ATAC)",
         "e11_meth_sel": "E11 neural tube (RNA + DNA methylation)",
         "xenium_he_left": "Xenium RCC LEFT nest, 50 um bins (RNA + H&E; protein = validation)",
         "xenium_he_right": "Xenium RCC RIGHT nest, 50 um bins (RNA + H&E; protein = validation)",
         "splisoseq_v2": "Spl-ISO-Seq2 Visium HD V2 mouse brain, 64 um bins (gene + isoform usage)",
         "rtt_merfish_s5": "RT&T-AMP-MERFISH sample 5 mouse brain, 50 um bins (gene + isoform usage)",
         "rcc_tcr_clone": "RCC Slide-GoTags, 200 um bins (RNA + TCR clonal composition)",
         "lpc_lpc5_cut": "LPC5 brushed cortex subset (RNA + ATAC)",
         "lpc_lpc10_cut": "LPC10 brushed cortex subset (RNA + ATAC)",
         "lpc_lpc21_cut": "LPC21 brushed cortex subset (RNA + ATAC)",
         "lpc10_fov": "LPC10 FOV: injection site + distal WM, cortex excluded (RNA + ATAC)"}


# per-dataset map annotation / annotation filters for the review drafts
MAPANN_REVIEW = {"melanoma": "ann_cnv_burden_v2", "melanoma_sel": "ann_cnv_burden_v2",
                 "lpc10_sel": "ann_cc_cluster_cortex", "e11_meth_sel": "ann_cc_K10",
                 "xenium_he_left": "ann_frac_RCC", "xenium_he_right": "ann_frac_RCC",
                 "splisoseq_v2": "ann_region", "rtt_merfish_s5": "ann_ccf_region",
                 "rcc_tcr_clone": "ann_frac_top_clone", "lpc_lpc5_cut": "ann_domain",
                 "lpc_lpc10_cut": "ann_domain", "lpc_lpc21_cut": "ann_domain"}
ANN_DROP = {"e11_hic": ("_concat", "_rna_", "_hic_", "cell_type_group", "cell_type_pred")}


def signed(a, v):
    ok = np.isfinite(v)
    return round(float(np.sign(np.corrcoef(a[ok], v[ok])[0, 1]) * spearman(a[ok], v[ok])), 3)


def main(name):
    z = dict(np.load(f"{R}/targets_{name}.npz", allow_pickle=True))
    keep = KEEP.get(name, lambda z: None)(z)
    N0 = len(z["barcodes"])
    if keep is not None:
        z = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == N0 else v) for k, v in z.items()}
    S = json.load(open(f"{R}/{name}/summary.json"))
    ax_df = pd.read_csv(f"{R}/{name}/axes.csv", index_col=0)
    mods = S["modalities"]; M = len(mods)
    D = ax_df[[f"d_{m}" for m in mods]].values; x, y = ax_df.x.values, ax_df.y.values
    assert len(D) == len(z["barcodes"]), f"{name}: axes/targets length mismatch"

    S["kurtosis"] = {m: float(kurtosis(D[:, k], fisher=False)) for k, m in enumerate(mods)}
    ann = {}
    for key in [k for k in z if k.startswith("ann_") and not any(t in k for t in ANN_DROP.get(name, ()))]:
        v = z[key]
        if v.dtype.kind in "fiu":
            v = v.astype(float)
            if np.isfinite(v).sum() > 10 and np.nanstd(v) > 0:
                ann[key[4:]] = {m: signed(D[:, k], v) for k, m in enumerate(mods)}
        elif 1 < len(set(v)) < 50:
            ann[key[4:]] = {m: round(eta2(D[:, k], v), 3) for k, m in enumerate(mods)}
    S["annotation_assoc"] = ann
    mk = [k for k in z if k.startswith("mk_") and np.std(z[k]) > 0]
    S["marker_corr"] = {k[3:]: {m: signed(D[:, j], z[k].astype(float)) for j, m in enumerate(mods)} for k in mk}
    json.dump(S, open(f"{R}/{name}/summary.json", "w"), indent=1, default=float)

    ncol = M + 4
    fig, ax = plt.subplots(2, ncol, figsize=(2.05 * ncol, 4.4))
    s = max(0.4, 2500 / len(x))
    for k in range(M):
        pr = rankdata(D[:, k]) / len(D)
        sc = ax[0, k].scatter(x, y, c=pr, cmap="coolwarm", s=s, linewidths=0, vmin=0, vmax=1)
        flag = "  [domain-like]" if S["kurtosis"][mods[k]] > 6 else ""
        ax[0, k].set_title(f"d_{mods[k]} (rank){flag}"); fig.colorbar(sc, ax=ax[0, k], fraction=0.046, label="percentile")
    mapk = MAPANN_REVIEW.get(name, MAPANN.get(name))
    if mapk and mapk in z:
        v = z[mapk]
        if v.dtype.kind in "fiu":
            v = v.astype(float); ok = np.isfinite(v)
            ax[0, M].scatter(x[~ok], y[~ok], c="#d0d0d0", s=s, linewidths=0)       # no annotation
            sc = ax[0, M].scatter(x[ok], y[ok], c=v[ok], cmap="viridis", s=s, linewidths=0); fig.colorbar(sc, ax=ax[0, M], fraction=0.046)
        else:
            cats = pd.Categorical(v)
            cmap = plt.get_cmap("tab20" if len(cats.categories) > 10 else "tab10")
            ax[0, M].scatter(x, y, c=cats.codes, cmap=cmap, s=s, linewidths=0, vmin=0, vmax=max(cmap.N - 1, 1))
        ax[0, M].set_title(mapk[4:])
    C = np.array(S.get("coupling_best", S["coupling_mean"])); Csd = np.array(S["coupling_sd"])
    ax[0, M + 1].imshow(C, cmap="viridis", vmin=0, vmax=1)
    for i in range(M):
        for j in range(M):
            ax[0, M + 1].text(j, i, f"{C[i, j]:.2f}\n±{Csd[i, j]:.2f}", ha="center", va="center",
                              color="w" if C[i, j] < 0.5 else "k", fontsize=5.5)
    ax[0, M + 1].set_xticks(range(M)); ax[0, M + 1].set_xticklabels([f"d_{m}" for m in mods], rotation=45)
    ax[0, M + 1].set_yticks(range(M)); ax[0, M + 1].set_yticklabels(mods); ax[0, M + 1].set_title("coupling, reported model\n(row=modality, col=axis; ± sd over seeds)")
    W = S.get("R2_whole")
    if W:
        parts = [("full", "full model", "#1f77b4"), ("own_axis_only", "own axis only", "#9ecae1"),
                 ("other_axes_only", "other axes only", "#fdae6b")]
        for i, (kk, lab, col) in enumerate(parts):
            ax[0, M + 2].bar(np.arange(M) + i * 0.27, [W[m][kk] for m in mods], 0.27, label=lab, color=col)
        ax[0, M + 2].set_xticks(np.arange(M) + 0.27); ax[0, M + 2].set_xticklabels(mods); ax[0, M + 2].axhline(0, color="k", lw=0.5)
        ax[0, M + 2].set_title(f"variance explained, ALL {len(x)} spots\n(best of {len(S['fit_all_by_seed'])} seeds)")
        ax[0, M + 2].legend(fontsize=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    # per-seed whole-section loss; the reported model is the lowest (GASTON best-of-seeds rule)
    fl = S.get("fit_all_by_seed")
    if fl:
        # fit_all_by_seed is stored best-first; recover the original seed order
        sl = S.get("seeds", list(range(len(fl))))
        best = sl.index(S["best_seed"]); order = list(range(len(fl))); order.remove(best); order.insert(0, best)
        by_seed = np.empty(len(fl)); by_seed[order] = fl
        cs = ["#d62728" if i == best else "#bdbdbd" for i in range(len(fl))]
        ax[0, M + 3].bar(range(len(fl)), by_seed, color=cs)
        ax[0, M + 3].set_ylim(min(by_seed) * 0.95, max(by_seed) * 1.02)
        ax[0, M + 3].set_xticks(range(len(fl))); ax[0, M + 3].set_xlabel("seed")
        ax[0, M + 3].set_title(f"whole-section loss per seed\nreported = seed {S['best_seed']} (red)")
        ax[0, M + 3].set_xticklabels([str(x) for x in S.get("seeds", order)])
    for k in range(M):
        a_ = ax[1, k]
        if mk:
            q = pd.qcut(pd.Series(D[:, k]).rank(method="first"), 10, labels=False).values
            Mt = np.array([[np.mean(z[g][q == b]) for b in range(10)] for g in mk])
            Mt = (Mt - Mt.mean(1, keepdims=True)) / (Mt.std(1, keepdims=True) + 1e-9)
            a_.imshow(Mt, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
            a_.set_yticks(range(len(mk))); a_.set_yticklabels([g[3:].replace("prot_", "P:") for g in mk], fontsize=5)
            a_.set_xlabel(f"d_{mods[k]} decile"); a_.set_title(f"markers along d_{mods[k]} (z)")
    if ann:
        keys = sorted(ann, key=lambda kk: -max(abs(vv) for vv in ann[kk].values()))[:12]
        A = np.array([[ann[kk][m] for m in mods] for kk in keys])
        ax[1, M].imshow(A, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax[1, M].set_yticks(range(len(keys))); ax[1, M].set_yticklabels(keys, fontsize=5)
        ax[1, M].set_xticks(range(M)); ax[1, M].set_xticklabels(mods)
        for i in range(len(keys)):
            for j in range(M):
                ax[1, M].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5)
        ax[1, M].set_title("axis vs annotation\n(numeric: Spearman; categorical: eta2)")
    st, ku = S["seed_stability"], S["kurtosis"]
    if not np.isfinite(list(st.values())).any():
        st = {m: 0.0 for m in mods}
    ax[1, M + 1].bar(np.arange(M) - 0.2, [st[m] for m in mods], 0.4, color="grey", label="seed stability")
    ax2 = ax[1, M + 1].twinx(); ax2.bar(np.arange(M) + 0.2, [ku[m] for m in mods], 0.4, color="#9467bd", label="kurtosis")
    ax2.axhline(6, color="#9467bd", ls="--", lw=0.6); ax2.set_ylabel("kurtosis", color="#9467bd")
    ax[1, M + 1].set_ylim(0, 1); ax[1, M + 1].set_xticks(range(M)); ax[1, M + 1].set_xticklabels(mods)
    ax[1, M + 1].set_title("stability (grey; n/a for 1 seed) | kurtosis (purple)")
    dc = S["axis_depth_corr"]
    ax[1, M + 2].bar(range(len(dc)), list(dc.values()), color="grey"); ax[1, M + 2].set_ylim(0, 1)
    ax[1, M + 2].set_xticks(range(len(dc))); ax[1, M + 2].set_xticklabels(list(dc)); ax[1, M + 2].set_title("|corr| axis vs depth\n(modalities with a depth only)")
    ax[1, M + 3].axis("off")
    for a_ in ax[0, :M + 1]:
        a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{TITLE.get(name, name)}   n={len(x)} (all spots)   reported seed {S.get('best_seed', 0)}   -- REVIEW DRAFT", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/{name}/review_{name}.png", dpi=200); fig.savefig(f"{R}/{name}/review_{name}.svg")
    plt.close(fig); print(f"{name}: review draft written; kurtosis {({m: round(v, 1) for m, v in ku.items()})}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
