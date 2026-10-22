"""High-level fitting: several seeds, best-of-seeds selection, and the standard read-outs.

`fit` is the procedure behind every published Saturn fit (scripts `run_real2.py`):

  1. train_saturn once per seed; every seed ends with a refit on ALL spots
  2. keep the seed with the lowest whole-section loss (GASTON's best-of-seeds rule)
  3. sign-align the other seeds to it and report seed stability (mean pairwise Spearman)
  4. report, per modality, the variance explained by the full model, by its own axis
     alone (cross pathways frozen at their mean), and by the other axes alone
  5. the coupling matrix, axis-axis correlation, and |Spearman| of each axis with log depth

Isodepth sign is arbitrary: orient axes against a marker (`orient`) before comparing fits.
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .core import Saturn, train_saturn

REFIT_ALL = 3000


def r2(Y, P) -> float:
    """pooled R2 of an (N, K) target: 1 - SSE / total sum of squares"""
    Y, P = np.asarray(Y, float), np.asarray(P, float)
    return float(1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-12))


@torch.no_grad()
def decompose(model: Saturn, coords_z, device: str = "cpu"):
    """Per modality (lists of (N, K_m) arrays): full prediction, own-axis term
    (base + own pathway) and summed cross term; plus the (N, M) isodepths."""
    model = model.to(device).eval(); model.cross_on = True
    d, full, own, cross = model(torch.as_tensor(np.asarray(coords_z, np.float32), device=device))
    t = lambda L: [x.cpu().numpy() for x in L]
    return d.cpu().numpy(), t(full), t(own), t(cross)


def r2_split(model: Saturn, coords_z, targets, masks=None, modalities=None, device="cpu") -> Dict[str, dict]:
    """full / own_axis_only / other_axes_only R2 per modality on observed spots.
    own_axis_only freezes the cross term at its mean; other_axes_only freezes the own term."""
    _, full, own, cross = decompose(model, coords_z, device)
    names = modalities or [f"m{k}" for k in range(len(targets))]
    out = {}
    for k, nm in enumerate(names):
        ob = np.ones(len(targets[k]), bool) if masks is None or masks[k] is None else np.asarray(masks[k], bool)
        f, o, c, y = full[k][ob], own[k][ob], cross[k][ob], np.asarray(targets[k])[ob]
        out[nm] = dict(full=r2(y, f), own_axis_only=r2(y, f - c + c.mean(0)), other_axes_only=r2(y, f - o + o.mean(0)))
    return out


def orient(axes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """flip each column of `axes` so it correlates positively with the same column of `reference`
    (a marker score per axis, or the axes of a reference fit)"""
    A = np.array(axes, float, copy=True)
    R = np.asarray(reference, float).reshape(len(A), -1)
    for k in range(A.shape[1]):
        if np.corrcoef(A[:, k], R[:, min(k, R.shape[1] - 1)])[0, 1] < 0:
            A[:, k] *= -1
    return A


@dataclass
class SaturnFit:
    """Result of `fit`. `axes` columns follow `modalities`."""
    model: Saturn
    modalities: List[str]
    axes: np.ndarray
    coupling: np.ndarray
    r2: Dict[str, dict]
    best_seed: int
    seeds: List[int]
    fit_all_by_seed: List[float]
    seed_stability: Dict[str, float]
    axis_axis_corr: List[List[float]]
    axis_depth_corr: Dict[str, float] = field(default_factory=dict)
    all_axes: Optional[List[np.ndarray]] = None
    couplings: Optional[List[np.ndarray]] = None      # one (M, M) matrix per seed, in `seeds` order
    history: Optional[dict] = None

    def summary(self) -> dict:
        return dict(modalities=self.modalities, best_seed=self.best_seed, seeds=self.seeds,
                    fit_all_by_seed=self.fit_all_by_seed, R2_whole=self.r2, seed_stability=self.seed_stability,
                    coupling_best=np.round(self.coupling, 4).tolist(),
                    coupling_mean=np.round(np.mean(self.couplings, 0), 4).tolist() if self.couplings else None,
                    coupling_sd=np.round(np.std(self.couplings, 0), 4).tolist() if self.couplings else None,
                    axis_axis_corr=self.axis_axis_corr,
                    axis_depth_corr=self.axis_depth_corr)

    def save(self, out_dir: str, coords=None, barcodes=None):
        """writes saturn_best.pt (state_dict), axes.csv (x, y, d_<mod>) and summary.json"""
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{out_dir}/saturn_best.pt")
        cols = [f"d_{m}" for m in self.modalities]
        df = pd.DataFrame(self.axes, columns=cols, index=barcodes)
        if coords is not None:
            df.insert(0, "y", np.asarray(coords)[:, 1]); df.insert(0, "x", np.asarray(coords)[:, 0])
        df.to_csv(f"{out_dir}/axes.csv")
        json.dump(self.summary(), open(f"{out_dir}/summary.json", "w"), indent=1, default=float)


def load_model(path: str, K: Sequence[int], **kw) -> Saturn:
    """rebuild a saved Saturn from its state_dict; K = target width per modality, in fit order"""
    m = Saturn(list(K), **kw)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("iso.") for k in sd):
        sd = sd["state_dict"]
    m.load_state_dict(sd); m.eval(); m.cross_on = True
    return m


def fit(targets: Sequence[np.ndarray], coords_z: np.ndarray, modalities: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42,), masks=None, depth: Optional[Dict[str, np.ndarray]] = None,
        refit_all_epochs: int = REFIT_ALL, device: str = "cpu", threads: Optional[int] = None,
        verbose: int = 0, **train_kw) -> SaturnFit:
    """Fit Saturn with best-of-seeds selection. `targets` from saturn.preprocess (globally scaled),
    `coords_z` from saturn.preprocess.zscore_coords. Extra keywords go to train_saturn.

    Reproducibility: CPU fits depend on the thread count (reduction order). Pin `threads`
    (and OMP_NUM_THREADS) and record them; the same seed with another thread count can land in a
    different basin (measured: LPC10 rho_RNA r = 0.86 vs the published fit at 4 vs 8 threads)."""
    if threads is not None:
        torch.set_num_threads(int(threads))
    mods = list(modalities or [f"m{k}" for k in range(len(targets))])
    seeds = list(seeds)
    models, hists, axes, fits, couplings = [], [], [], [], []
    xy = torch.as_tensor(np.asarray(coords_z, np.float32), device=device)
    for s in seeds:
        m, h = train_saturn(targets, coords_z, seed=s, masks=masks, device=device,
                            refit_all_epochs=refit_all_epochs, verbose=verbose, **train_kw)
        with torch.no_grad():
            axes.append(m.isodepth(xy).cpu().numpy())
            couplings.append(m.coupling(xy))
        models.append(m); hists.append(h); fits.append(h["fit_all"])
        if verbose:
            print(f"seed {s}: whole-section fit {h['fit_all']:.4f}", flush=True)
    b = int(np.argmin(fits)); best, D = models[b], axes[b]
    aligned = [orient(a, D) for a in axes]
    M = D.shape[1]
    stab = {mods[k]: (float(np.mean([spearmanr(aligned[i][:, k], aligned[j][:, k]).statistic
                                     for i in range(len(seeds)) for j in range(i + 1, len(seeds))]))
                      if len(seeds) > 1 else float("nan")) for k in range(M)}
    dc = {}
    for k, nm in enumerate(mods):
        if depth is not None and nm in depth:
            dc[nm] = float(abs(spearmanr(D[:, k], np.log1p(np.asarray(depth[nm], float))).statistic))
    return SaturnFit(model=best, modalities=mods, axes=D, coupling=best.coupling(xy),
                     r2=r2_split(best, coords_z, targets, masks, mods, device), best_seed=seeds[b], seeds=seeds,
                     fit_all_by_seed=[float(f) for f in fits], seed_stability=stab,
                     axis_axis_corr=[[round(float(spearmanr(D[:, i], D[:, j]).statistic), 3) for j in range(M)]
                                     for i in range(M)],
                     axis_depth_corr=dc, all_axes=aligned, couplings=couplings, history=hists[b])


def heldout_r2(targets, coords_z, modalities=None, test_frac: float = 0.2, seed: int = 0, masks=None,
               device: str = "cpu", gaston: bool = False, **train_kw) -> pd.DataFrame:
    """Held-out R2 per modality on a random `test_frac` of spots (Saturn trained on the rest).
    gaston=True adds one single-modality GASTON per modality fitted on the same globally-scaled
    target (needs `pip install gaston-spatial`)."""
    mods = list(modalities or [f"m{k}" for k in range(len(targets))])
    N = len(coords_z); rng = np.random.default_rng(seed); perm = rng.permutation(N)
    te, tr = perm[:int(N * test_frac)], perm[int(N * test_frac):]
    sub = lambda mk, ix: None if mk is None else np.asarray(mk)[ix]
    mks = masks or [None] * len(targets)
    m, _ = train_saturn([np.asarray(y)[tr] for y in targets], coords_z[tr], seed=seed, device=device,
                        masks=[sub(mk, tr) for mk in mks], refit_all_epochs=train_kw.pop("refit_all_epochs", REFIT_ALL),
                        **train_kw)
    _, full, _, _ = decompose(m, coords_z[te], device)
    rows = []
    for k, nm in enumerate(mods):
        ob = np.ones(len(te), bool) if mks[k] is None else np.asarray(mks[k], bool)[te]
        rows.append(dict(modality=nm, method="Saturn", R2=r2(np.asarray(targets[k])[te][ob], full[k][ob])))
        if gaston:
            import contextlib, io
            from gaston.neural_net import train as gtrain
            obr = np.ones(len(tr), bool) if mks[k] is None else np.asarray(mks[k], bool)[tr]
            with contextlib.redirect_stdout(io.StringIO()):
                g, _ = gtrain(torch.tensor(coords_z[tr][obr]), torch.tensor(np.asarray(targets[k])[tr][obr], dtype=torch.float32),
                              S_hidden_list=[20, 20], A_hidden_list=[20, 20], epochs=10000, optim="adam", lr=1e-3,
                              seed=seed, device=device)
            P = g.cpu().eval()(torch.tensor(coords_z[te][ob])).detach().numpy()
            rows.append(dict(modality=nm, method="GASTON", R2=r2(np.asarray(targets[k])[te][ob], P)))
    return pd.DataFrame(rows)
