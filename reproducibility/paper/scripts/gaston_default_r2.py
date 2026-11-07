"""Default GASTON fitted separately on each modality of each figure dataset, same targets as Saturn -> whole-section R2.

GASTON as in its tutorial: per-PC z-scored target, z-scored coordinates, S/A hidden [20, 20], 10,000 epochs, Adam
lr 1e-3, 30 restarts, the restart with the lowest final training loss is kept. R2 is computed on the Saturn target Y
(the prediction mapped back from GASTON's z-scale), on all spots, exactly as Saturn's R2_whole.full.
-> results/real_saturn/gaston_default_r2/{restarts.csv, summary.csv}
Usage: python scripts/gaston_default_r2.py [--restarts 30] [--workers 16]
"""
import os, sys, io, json, contextlib
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"; OUT = f"{R}/gaston_default_r2"
NR = int(sys.argv[sys.argv.index("--restarts") + 1]) if "--restarts" in sys.argv else 30
NW = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
# figure -> (targets, Saturn fit whose R2_whole the figure reports)
FITS = {"Fig 2 LPC10 (RNA+ATAC)": ("lpc10_fov", "lpc10_fov/summary.json"),
        "Fig 3 melanoma (RNA+ATAC)": ("melanoma_tissue", "melanoma_tissue/summary.json"),
        "Fig 4 MERFISH (RNA+isoform)": ("rtt_merfish_s5_iso100_sv_pca_shrunk", "rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3/diag.json"),
        "Fig 5 Sprox-seq B1 (RNA+protein)": ("sproxseq_B1_p46_dr", "sproxseq_B1_p46_dr/summary.json"),
        "Fig 6 melanoma Hi-C (RNA+Hi-C)": ("mel_hic_abins_dr50_51_rnadr", "mel_hic_abins_dr50_51_rnadr/summary.json")}


def r2w(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-12))


def one(job):
    ds, m, seed = job
    import torch
    torch.set_num_threads(2)
    from gaston.neural_net import train
    z = np.load(f"{R}/targets_{ds}.npz", allow_pickle=True)
    Y = np.asarray(z[f"Y_{m}"], float); X = np.asarray(z["coords"], float)
    mu, sd = Y.mean(0), Y.std(0) + 1e-8; S = (X - X.mean(0)) / X.std(0)
    with contextlib.redirect_stdout(io.StringIO()):
        mod, loss = train(torch.tensor(S, dtype=torch.float32), torch.tensor((Y - mu) / sd, dtype=torch.float32),
                          S_hidden_list=[20, 20], A_hidden_list=[20, 20], epochs=10000, checkpoint=500,
                          optim="adam", lr=1e-3, seed=seed, device="cpu")
    with torch.no_grad():
        P = mod(torch.tensor(S, dtype=torch.float32)).numpy() * sd + mu
    return dict(dataset=ds, modality=m, seed=seed, loss=float(np.ravel(loss)[-1]), R2=r2w(Y, P))


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(ds, str(m), s) for ds, _ in FITS.values()
            for m in np.load(f"{R}/targets_{ds}.npz", allow_pickle=True)["modalities"] for s in range(NR)]
    rows = []
    with ProcessPoolExecutor(NW) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            rows.append(r)
            if (i + 1) % 20 == 0: print(f"{i + 1}/{len(jobs)}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/restarts.csv", index=False)
    out = []
    for fig, (ds, sj) in FITS.items():
        sat = json.load(open(f"{R}/{sj}"))["R2_whole"]
        for m, g in T[T.dataset == ds].groupby("modality", sort=False):
            b = g.loc[g.loss.idxmin()]
            out.append(dict(figure=fig, dataset=ds, modality=m, Saturn=round(sat[m]["full"], 3),
                            Saturn_own_axis_only=round(sat[m]["own_axis_only"], 3),
                            GASTON_best_loss=round(b.R2, 3), GASTON_median=round(g.R2.median(), 3),
                            GASTON_max=round(g.R2.max(), 3), n_restarts=len(g)))
    S = pd.DataFrame(out); S.to_csv(f"{OUT}/summary.csv", index=False); print(S.to_string(index=False))


if __name__ == "__main__":
    main()
