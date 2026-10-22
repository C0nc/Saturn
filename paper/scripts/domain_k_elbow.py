"""Choose the number of GASTON domains K per figure axis from the DP error curve (elbow).

GASTON's DP cuts an isodepth into exactly K contiguous intervals with a linear fit of the
modality's targets inside each one; the error always falls with K, so K has to be chosen.
GASTON picks it at the elbow of this curve. Every figure used to hard-code K = 4 (KDOM,
inherited from the LPC figure); this script replaces that with a per-axis elbow.

Elbow = kneedle (Satopaa 2011): normalise K = 1..KMAX and the unexplained fraction to [0, 1]
and take the K with the largest gap below the chord from K = 1 to K = KMAX.
Buckets: NBUCKET (50) as gaston_domains uses by default in every figure script.

-> results/real_saturn/domain_k_elbow/{k_elbow.csv, k_elbow.json, k_elbow_curves.{png,svg}}
   k_elbow.json is what the figure scripts read: {"<figure>|<modality>": K}

Usage: python scripts/domain_k_elbow.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpc_gaston_fig4l import NBUCKET, SD
from domain_annot import R

KMAX = 10
OUT = f"{R}/domain_k_elbow"
AXES_DS = [("fig2", "lpc10_fov", ("RNA", "ATAC")),
           ("fig3", "melanoma_tissue", ("RNA", "ATAC")),
           ("fig6", "mel_hic_fh_dr50_51", ("RNA", "HiC"))]


def curve(d, Y, nb=NBUCKET):
    """unexplained fraction, K = 1..KMAX (saturn.domains.dp_error_curve)"""
    return SD.dp_error_curve(d, Y, k_max=KMAX, n_buckets=nb)


def kneedle(u):
    """elbow K of the curve (saturn.domains.kneedle)"""
    return SD.kneedle(u)


def inputs():
    for fig, ds, mods in AXES_DS:
        ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0)
        z = np.load(f"{R}/targets_{ds}.npz", allow_pickle=True)
        for m in mods:
            yield fig, ds, m, ax[f"d_{m}"].values, z[f"Y_{m}"]
    import rtt_isoform_sv_isodepth as I                              # Fig 4 (MERFISH isoform, bins)
    _, B = I.load()
    for m in ("RNA", "Isoform"):
        yield "fig4", I.DS, m, B["d"][m], B["z"][f"Y_{m}"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for fig, ds, m, d, Y in inputs():
        u = curve(d, Y); k = kneedle(u)
        rows.append(dict(figure=fig, dataset=ds, modality=m, n=len(d), K_elbow=k,
                         **{f"u{j + 1}": round(float(v), 4) for j, v in enumerate(u)}))
        print(f"{fig} {ds} {m}: elbow K = {k}; unexplained {np.round(u[:6], 3).tolist()}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/k_elbow.csv", index=False)
    json.dump({f"{r.figure}|{r.modality}": int(r.K_elbow) for r in T.itertuples()}, open(f"{OUT}/k_elbow.json", "w"), indent=1)

    sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from paper_style import apply, save_dual, MODALITY
    apply(base_pt=6.5)
    figs = list(dict.fromkeys(T.figure))
    fig, axs = plt.subplots(1, len(figs), figsize=(1.75 * len(figs), 1.7), sharey=False)
    for a, f in zip(axs, figs):
        for r in T[T.figure == f].itertuples():
            u = np.array([getattr(r, f"u{j}") for j in range(1, KMAX + 1)]); k = np.arange(1, KMAX + 1)
            c = MODALITY.get(r.modality, "#555555")
            a.plot(k, 100 * (1 - u), color=c, lw=1, marker="o", ms=2, label=f"{r.modality} (K={r.K_elbow})")
            a.plot(r.K_elbow, 100 * (1 - u[r.K_elbow - 1]), "o", ms=5, mfc="none", mec=c, mew=1)
        a.axvline(4, color="#9aa2aa", lw=0.6, ls=":")
        a.set_title(f"{f}: {T[T.figure == f].dataset.iloc[0]}", fontsize=6, pad=2)
        a.set_xlabel("domains K", labelpad=1); a.set_xticks([1, 2, 4, 6, 8, 10])
        a.legend(fontsize=5.2, loc="lower right", handlelength=1.2, borderaxespad=0.1)
    axs[0].set_ylabel("variance explained by\nK linear pieces (%)", labelpad=1)
    fig.tight_layout()
    save_dual(fig, f"{OUT}/k_elbow_curves", extra=dict(rule="kneedle on unexplained fraction, K=1..10",
                                                       K=json.load(open(f"{OUT}/k_elbow.json"))))
    print(T[["figure", "dataset", "modality", "K_elbow"]].to_string(index=False))


if __name__ == "__main__":
    main()
