"""Saturn on the real-data suite (any M, masked modalities) + draft figure.

Evaluation (no ground truth):
  seed stability | held-out R2 vs GASTON (default, global-scaled) | axis-depth corr |
  coupling matrix (mean +- sd over seeds) | axis vs annotations (numeric: Spearman;
  categorical: eta^2 = share of axis variance explained by the categories) |
  axis vs marker genes (signed Spearman).
Outputs results/real_saturn/<name>/{summary.json, axes.csv, draft_<name>.png/.svg}.
"""
from __future__ import annotations
import os, sys, io, json, argparse, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

ROOT = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm"
sys.path.insert(0, os.path.join(ROOT, "sim")); sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/src")
from saturn import train_saturn                                         # noqa
from metrics_v2 import spearman                                         # noqa
from gaston.neural_net import train as gaston_train                     # noqa
EPOCHS, HID = 10000, [20, 20]
DEV = os.environ.get("SAT_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("SAT_THREADS", "4")))

# spot filters. Default is now the WHOLE section (every spot); --subset restores
# the old in-tissue melanoma filter for comparison.
REFIT_ALL = 3000
KEEP_SUBSET = {"melanoma": lambda z: np.isfinite(np.asarray(z["ann_cnv_burden_v2"], dtype=float))
        if "ann_cnv_burden_v2" in z else None}
# the annotation to show on the map, per dataset
MAPANN = {"lpc10": "ann_wb_lesion_v2", "melanoma": "ann_mel_marker", "e11_hic": "ann_cluster_cellcharter",
          "e11_meth": "ann_wsnn_res.1", "rcc_tcr": "ann_TCRpos", "xenium_rcc": "ann_frac_RCC",
          "rcc_tcr_bin": "ann_n_TCRpos_beads"}
KEEP = {}


def r2w(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-12))


def fit_gaston(S, A, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        mod, _ = gaston_train(torch.tensor(S, dtype=torch.float32), torch.tensor(A, dtype=torch.float32),
                              S_hidden_list=HID, A_hidden_list=HID, epochs=EPOCHS,
                              optim="adam", lr=1e-3, seed=seed, device=DEV)
    return mod.cpu().eval()


def zfit(X):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def eta2(axis, cats):
    df = pd.DataFrame({"a": axis, "c": cats}); df = df[df.c.notna() & (df.c.astype(str) != "nan")]
    tot = ((df.a - df.a.mean()) ** 2).sum()
    between = df.groupby("c").a.apply(lambda s: len(s) * (s.mean() - df.a.mean()) ** 2).sum()
    return float(between / (tot + 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=42, help="first seed; runs seed0 .. seed0+seeds-1")
    ap.add_argument("--subset", action="store_true", help="old in-tissue filter (melanoma)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam_tail", type=float, default=0.0,
                    help="kurtosis tail penalty on the axes (saturn.core._tail_penalty); 0 = off (default, all published fits)")
    ap.add_argument("--bench", action="store_true", help="also fit GASTON + held-out R2 (off: real-data runs report Saturn only)")
    a = ap.parse_args()
    KEEP.update(KEEP_SUBSET if a.subset else {})
    z = dict(np.load(f"{ROOT}/results/real_saturn/targets_{a.dataset}.npz", allow_pickle=True))
    keep = KEEP.get(a.dataset, lambda z: None)(z)
    N0 = len(z["barcodes"])
    if keep is not None:
        for k, v in list(z.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == N0 and k not in ("modalities",):
                z[k] = v[keep]
    mods = [str(m) for m in z["modalities"]]
    Y = [z[f"Y_{m}"] for m in mods]
    masks = [z.get(f"mask_{m}") for m in mods]
    xyn, raw = z["coords_z"], z["coords"]
    N, M = len(xyn), len(mods)
    out = f"{ROOT}/results/real_saturn/{a.dataset}{a.tag}"; os.makedirs(out, exist_ok=True)
    S = dict(dataset=a.dataset, modalities=mods, n_spots=N,
             n_observed={m: int(np.sum(mk)) if mk is not None else N for m, mk in zip(mods, masks)})
    print(f"{a.dataset}: {N} spots, {mods}, device {DEV}", flush=True)
    xy = torch.as_tensor(xyn, device=DEV)

    # ---- Saturn, several seeds ------------------------------------------------
    # every seed ends with a refit on ALL spots; the reported model is the seed
    # with the lowest whole-section loss (GASTON's best-of-seeds rule)
    axes, C, fits, models = [], [], [], []
    seeds = [a.seed0 + i for i in range(a.seeds)]
    for s in seeds:
        m, h = train_saturn(Y, xyn, seed=s, masks=masks, device=DEV, refit_all_epochs=REFIT_ALL,
                            lam_tail=a.lam_tail)
        axes.append(m.isodepth(xy).detach().cpu().numpy()); C.append(m.coupling(xy))
        fits.append(h["fit_all"]); models.append(m)
        print(f"  seed {s}: whole-section fit {h['fit_all']:.4f}; restarts accepted "
              f"{[(r, hd) for r, hd, b, af, ok in h['restarts'] if ok]}", flush=True)
    best = int(np.argmin(fits))
    axes.insert(0, axes.pop(best)); C.insert(0, C.pop(best)); fits.insert(0, fits.pop(best))
    torch.save(models[best].state_dict(), f"{out}/saturn_best.pt")
    S["best_seed"], S["fit_all_by_seed"], S["seeds"] = seeds[best], fits, seeds
    # sign-align seeds to the best seed, then report stability
    for i in range(1, a.seeds):
        for k in range(M):
            if np.corrcoef(axes[i][:, k], axes[0][:, k])[0, 1] < 0:
                axes[i][:, k] *= -1
    S["seed_stability"] = {mods[k]: (float(np.mean([spearman(axes[i][:, k], axes[j][:, k])
                                                    for i in range(a.seeds) for j in range(i + 1, a.seeds)]))
                                     if a.seeds > 1 else float("nan")) for k in range(M)}
    S["coupling_best"] = np.asarray(C[0]).round(4).tolist()
    S["coupling_mean"] = np.mean(C, 0).round(4).tolist(); S["coupling_sd"] = np.std(C, 0).round(4).tolist()
    # whole-section variance explained per modality (observed spots), best model
    with torch.no_grad():
        mb = models[best]; mb.cross_on = True
        _, full_all, own_all, cross_all = mb(xy)
    S["R2_whole"] = {}
    for k in range(M):
        ob = np.ones(N, bool) if masks[k] is None else np.asarray(masks[k], bool)
        f_, o_, c_ = (t[k].cpu().numpy()[ob] for t in (full_all, own_all, cross_all))
        # own-axis only: cross pathways frozen at their mean; cross-only: vice versa
        S["R2_whole"][mods[k]] = dict(full=r2w(Y[k][ob], f_),
                                      own_axis_only=r2w(Y[k][ob], f_ - c_ + c_.mean(0)),
                                      other_axes_only=r2w(Y[k][ob], f_ - o_ + o_.mean(0)))
    D = axes[0]
    S["axis_axis_corr"] = [[round(spearman(D[:, i], D[:, j]), 3) for j in range(M)] for i in range(M)]

    if a.bench:
        # ---- GASTON per modality (fit on observed spots, evaluate everywhere) -----
        G_def, G_glo = [], []
        for k in range(M):
            ob = np.ones(N, bool) if masks[k] is None else np.asarray(masks[k], bool)
            Zk, _, _ = zfit(Y[k][ob]); Sz, smu, ssd = zfit(raw[ob])
            g = fit_gaston(Sz, Zk, 0)
            G_def.append(g.spatial_embedding(torch.tensor((raw - smu) / ssd, dtype=torch.float32)).detach().numpy().ravel())
            g2 = fit_gaston(xyn[ob], Y[k][ob], 0)
            G_glo.append(g2.spatial_embedding(torch.tensor(xyn, dtype=torch.float32)).detach().numpy().ravel())
        S["saturn_vs_gaston"] = {mods[k]: dict(default=spearman(D[:, k], G_def[k]), global_scaled=spearman(D[:, k], G_glo[k]))
                                 for k in range(M)}

        # ---- held-out R2 (observed test spots only) --------------------------------
        rng = np.random.default_rng(0); perm = rng.permutation(N); te, tr = perm[:N // 5], perm[N // 5:]
        ms, _ = train_saturn([y[tr] for y in Y], xyn[tr], seed=0, refit_all_epochs=REFIT_ALL,
                             masks=[None if mk is None else np.asarray(mk)[tr] for mk in masks], device=DEV)
        ms.cross_on = True
        with torch.no_grad():
            _, full, _, _ = ms(torch.as_tensor(xyn[te], device=DEV))
        held = {"Saturn": {}, "GASTON (default)": {}, "GASTON (global-scaled)": {}}
        for k in range(M):
            obt = np.ones(len(te), bool) if masks[k] is None else np.asarray(masks[k], bool)[te]
            obr = np.ones(len(tr), bool) if masks[k] is None else np.asarray(masks[k], bool)[tr]
            if obt.sum() < 5 or obr.sum() < 10:
                continue
            held["Saturn"][mods[k]] = r2w(Y[k][te][obt], full[k].cpu().numpy()[obt])
            Zk, mu, sd = zfit(Y[k][tr][obr]); Sz, smu, ssd = zfit(raw[tr][obr])
            g = fit_gaston(Sz, Zk, 0)
            pz = g(torch.tensor((raw[te][obt] - smu) / ssd, dtype=torch.float32)).detach().numpy()
            held["GASTON (default)"][mods[k]] = r2w(Y[k][te][obt], pz * sd + mu)
            g2 = fit_gaston(xyn[tr][obr], Y[k][tr][obr], 0)
            held["GASTON (global-scaled)"][mods[k]] = r2w(
                Y[k][te][obt], g2(torch.tensor(xyn[te][obt], dtype=torch.float32)).detach().numpy())
        S["heldout_R2"] = held

    # ---- depth, annotations, markers ------------------------------------------
    # Spearman on LOG depth: Pearson on raw depth understates the confound badly
    # (melanoma_whole read 0.13 by Pearson and 0.43 by this).
    from scipy.stats import spearmanr as _sp
    S["axis_depth_corr"] = {mods[k]: float(abs(_sp(D[:, k], np.log1p(np.asarray(z[f"depth_{mods[k]}"], float))).statistic))
                            for k in range(M) if f"depth_{mods[k]}" in z}
    ann = {}
    for key in [k for k in z if k.startswith("ann_")]:
        v = z[key]
        if v.dtype.kind in "fiu":
            v = v.astype(float); ok = np.isfinite(v)
            if ok.sum() > 10 and np.nanstd(v) > 0:
                ann[key[4:]] = {mods[k]: round(float(np.sign(np.corrcoef(D[ok, k], v[ok])[0, 1]) *
                                                     spearman(D[ok, k], v[ok])), 3) for k in range(M)}
        else:
            if 1 < len(set(v)) < 50:
                ann[key[4:]] = {mods[k]: round(eta2(D[:, k], v), 3) for k in range(M)}
    S["annotation_assoc"] = ann
    S["marker_corr"] = {key[3:]: {mods[k]: round(float(np.sign(np.corrcoef(D[:, k], z[key])[0, 1]) *
                                                       spearman(D[:, k], z[key])), 3) for k in range(M)}
                        for key in [k for k in z if k.startswith("mk_")] if np.std(z[key]) > 0}

    pd.DataFrame(np.c_[raw, D], columns=["x", "y"] + [f"d_{m}" for m in mods], index=z["barcodes"]).to_csv(f"{out}/axes.csv")
    json.dump(S, open(f"{out}/summary.json", "w"), indent=1, default=float)
    import redraw_drafts                                   # review draft from the saved axes
    redraw_drafts.main(a.dataset)
    print(json.dumps({k: S[k] for k in ["best_seed", "R2_whole", "seed_stability", "coupling_best", "axis_depth_corr"]},
                     indent=1, default=float), flush=True)


def draft_figure(name, z, raw, D, mods, S, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7,
                         "svg.fonttype": "none"})
    M = len(mods); mapk = MAPANN.get(name)
    ncol = M + 3
    fig, ax = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.2))
    x, y = raw[:, 0], raw[:, 1]
    s = max(0.5, 3000 / len(x))
    for k in range(M):                                                    # axis maps
        sc = ax[0, k].scatter(x, y, c=D[:, k], cmap="coolwarm", s=s, linewidths=0)
        ax[0, k].set_title(f"Saturn d_{mods[k]}"); fig.colorbar(sc, ax=ax[0, k], fraction=0.046)
    if mapk and mapk in z:                                                # annotation map
        v = z[mapk]
        if v.dtype.kind in "fiu":
            sc = ax[0, M].scatter(x, y, c=v.astype(float), cmap="viridis", s=s, linewidths=0); fig.colorbar(sc, ax=ax[0, M], fraction=0.046)
        else:
            cats = pd.Categorical(v); ax[0, M].scatter(x, y, c=cats.codes, cmap="tab10", s=s, linewidths=0)
        ax[0, M].set_title(mapk[4:])
    C = np.array(S["coupling_mean"])                                       # coupling
    im = ax[0, M + 1].imshow(C, cmap="viridis", vmin=0, vmax=1)
    ax[0, M + 1].set_xticks(range(M)); ax[0, M + 1].set_xticklabels([f"d_{m}" for m in mods], rotation=45)
    ax[0, M + 1].set_yticks(range(M)); ax[0, M + 1].set_yticklabels(mods)
    for i in range(M):
        for j in range(M):
            ax[0, M + 1].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="w" if C[i, j] < 0.5 else "k", fontsize=6)
    ax[0, M + 1].set_title("coupling (row=modality, col=axis)")
    H = S["heldout_R2"]; meths = list(H); w = 0.8 / len(meths)              # held-out R2
    for i, mth in enumerate(meths):
        ax[0, M + 2].bar(np.arange(M) + i * w, [H[mth].get(m, np.nan) for m in mods], w, label=mth)
    ax[0, M + 2].set_xticks(np.arange(M) + 0.4 - w / 2); ax[0, M + 2].set_xticklabels(mods)
    ax[0, M + 2].set_title("held-out R2"); ax[0, M + 2].legend(fontsize=5, frameon=False)
    mk = [k for k in z if k.startswith("mk_") and np.std(z[k]) > 0]       # marker gradients
    for k in range(M):
        a_ = ax[1, k]
        if mk:
            q = pd.qcut(pd.Series(D[:, k]).rank(method="first"), 10, labels=False)
            Mt = np.array([[np.mean(z[g][q == b]) for b in range(10)] for g in mk])
            Mt = (Mt - Mt.mean(1, keepdims=True)) / (Mt.std(1, keepdims=True) + 1e-9)
            a_.imshow(Mt, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
            a_.set_yticks(range(len(mk))); a_.set_yticklabels([g[3:] for g in mk], fontsize=5)
            a_.set_xlabel(f"d_{mods[k]} decile"); a_.set_title(f"markers along d_{mods[k]}")
    ann = S["annotation_assoc"]                                           # annotation assoc
    if ann:
        keys = list(ann)[:10]; A = np.array([[ann[kk][m] for m in mods] for kk in keys])
        ax[1, M].imshow(A, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax[1, M].set_yticks(range(len(keys))); ax[1, M].set_yticklabels(keys, fontsize=5)
        ax[1, M].set_xticks(range(M)); ax[1, M].set_xticklabels(mods)
        for i in range(len(keys)):
            for j in range(M):
                ax[1, M].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5)
        ax[1, M].set_title("axis vs annotation\n(numeric: Spearman; categorical: eta2)")
    st = S["seed_stability"]
    ax[1, M + 1].bar(range(M), [st[m] for m in mods], color="grey"); ax[1, M + 1].set_ylim(0, 1)
    ax[1, M + 1].set_xticks(range(M)); ax[1, M + 1].set_xticklabels(mods); ax[1, M + 1].set_title("seed stability")
    dc = S["axis_depth_corr"]
    ax[1, M + 2].bar(range(len(dc)), list(dc.values()), color="grey"); ax[1, M + 2].set_ylim(0, 1)
    ax[1, M + 2].set_xticks(range(len(dc))); ax[1, M + 2].set_xticklabels(list(dc)); ax[1, M + 2].set_title("|corr| axis vs depth")
    for a_ in ax[0, :M + 1]:
        a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{name} -- DRAFT (n={len(x)} spots)", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/draft_{name}.png", dpi=200); fig.savefig(f"{out}/draft_{name}.svg")


if __name__ == "__main__":
    main()
