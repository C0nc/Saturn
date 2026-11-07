"""Saturn: GASTON extended to M modalities with cross-modal isodepth coupling.

GENERATIVE ASSUMPTION
    Each modality m has its own isodepth d_m(x). Every target feature j of every
    modality is explained ADDITIVELY by its own axis and by every other axis:

        Y_m,j(x) = base_m,j + sum_k  P_{m<-k, j}( d_k(x) ) + noise

    P_{m<-m} is the modality's own program; P_{m<-k}, k != m, is how modality m
    is shaped by modality k's axis (e.g. chromatin shaping RNA). With M=1 this is
    exactly GASTON: one isodepth, one piecewise-linear expression function.

DESIGN, AND WHY (each choice is a measured lesson, not a default)
    * Targets are low-dimensional continuous embeddings (RNA -> GLM-PCs,
      ATAC -> LSI with component 1 dropped), fitted with MSE -- GASTON's recipe.
      See saturn.preprocess.
    * Isodepth networks read RAW (z-scored) coordinates through small ReLU MLPs,
      so each d_m is a low-complexity, piecewise-linear surface. A random-Fourier
      encoding lets one axis become near-universal over space; measured on the
      earlier model, the shared axis grew to std 13 and absorbed every modality.
    * d_m is NOT standardised before decoding (standardize_d=False, as GASTON).
      Standardising was needed in the earlier shared-axis model, where one axis
      could grow until it absorbed every modality. Here each pathway reads one
      axis and its first linear layer absorbs any scale, while back-propagating
      through d's own std traps some initialisations. Measured on v5: with it,
      8/32 RNA-axis fits stuck near 0.2 (e.g. 0.234, 0.198, 0.462); without it
      the same configs recovered 0.840, 0.979, 0.986.
    * Every pathway P_{m<-k} reads ONE axis and outputs all K_m target columns
      through ReLU layers: a piecewise-linear curve per column, with bends shared
      across columns (GASTON's "domains are shared across genes", per axis).
    * Pathways are SUMMED. With an arbitrary function of all axes, the axes are
      identifiable only up to an invertible remixing (the head-collapse failure).
      Additivity removes that freedom (Lachapelle et al., NeurIPS 2023, which
      permits correlated latents). This is an imposed assumption; interactions
      can be tested per feature after the axes are fixed.

WHAT KEEPS "d_m BELONGS TO MODALITY m" MEANINGFUL
    Additivity leaves only label swaps. Two terms anchor the labels:
      own-anchor   d_m ALONE must reconstruct modality m
      cross cost   the k != m pathways must pay for the variance they carry
    Limit: this requires each modality to be explained more by its own axis
    than by any other. When cross-talk dominates, the label is not identifiable
    by any method.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .repro import set_all_seeds


def _mlp(sizes: Sequence[int], act=nn.ReLU) -> nn.Sequential:
    """Linear layers with `act` between them (none after the last)."""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Saturn(nn.Module):
    """Multi-modal isodepth model (M axes, M*M additive pathways).

    Parameters
    ----------
    K : list of int
        Target dimension per modality, e.g. [10, 15] for (RNA GLM-PCs, ATAC LSI).
    iso_hidden : hidden widths of each isodepth network  (x,y) -> d_m
    path_hidden : hidden widths of each pathway network  d_k -> K_m columns
    """

    def __init__(self, K: Sequence[int], iso_hidden: Sequence[int] = (20, 20),
                 path_hidden: Sequence[int] = (20, 20), standardize_d: bool = False):
        super().__init__()
        self.standardize_d = bool(standardize_d)
        self.K = list(K)
        self.M = len(self.K)
        relu = lambda sizes: _mlp(sizes, act=nn.ReLU)
        self.iso = nn.ModuleList(relu([2, *iso_hidden, 1]) for _ in self.K)
        self.path = nn.ModuleDict({
            self._key(m, k): relu([1, *path_hidden, self.K[m]])
            for m in range(self.M) for k in range(self.M)})
        self.base = nn.ParameterList(nn.Parameter(torch.zeros(k)) for k in self.K)
        # Cross pathways start at exactly zero output, so switching them on after
        # the warm-up continues from the per-modality GASTON solution instead of
        # perturbing it. (The last layer still receives gradient, so they learn.)
        for m in range(self.M):
            for k in range(self.M):
                if k != m:
                    last = self.path[self._key(m, k)][-1]
                    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.cross_on = True

    @staticmethod
    def _key(m: int, k: int) -> str:
        return f"{m}<-{k}"

    # ------------------------------------------------------------------ #
    def isodepth(self, xy: torch.Tensor, standardize: Optional[bool] = None) -> torch.Tensor:
        """(N, M) isodepths, one column per modality."""
        d = torch.cat([f(xy) for f in self.iso], 1)
        if standardize is None:
            standardize = self.standardize_d
        if standardize:
            d = (d - d.mean(0, keepdim=True)) / (d.std(0, keepdim=True) + 1e-6)
        return d

    def forward(self, xy: torch.Tensor):
        """Returns d, and per modality: full prediction, own-only prediction,
        and the summed cross contribution."""
        d = self.isodepth(xy)
        full, own, cross = [], [], []
        for m in range(self.M):
            o = self.path[self._key(m, m)](d[:, m:m + 1])
            cs = ([self.path[self._key(m, k)](d[:, k:k + 1])
                   for k in range(self.M) if k != m] if self.cross_on else [])
            c = torch.stack(cs).sum(0) if cs else torch.zeros_like(o)
            full.append(self.base[m] + o + c)
            own.append(self.base[m] + o)
            cross.append(c)
        return d, full, own, cross

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def coupling(self, xy: torch.Tensor) -> np.ndarray:
        """(M, M) matrix: entry [m, k] = share of modality m's predicted variance
        carried by the pathway from axis k. Rows sum to ~1 (ignoring covariance
        between pathways). Off-diagonal entries are the estimated cross-modal
        coupling; asymmetry gives the direction."""
        d = self.isodepth(xy)
        C = np.zeros((self.M, self.M))
        for m in range(self.M):
            v = [float(self.path[self._key(m, k)](d[:, k:k + 1]).var(0).sum())
                 for k in range(self.M)]
            C[m] = np.array(v) / (sum(v) + 1e-12)
        return C


# ---------------------------------------------------------------------- #
def _sel(t, mk):
    return t if mk is None else t[mk]


def _tail_penalty(d, k0: float = 4.0):
    """Hinge on each axis's kurtosis above k0 (Gaussian = 3, uniform ramp = 1.8).

    Real-data failure this targets: without per-axis standardisation an axis can
    spend almost its whole range on one small distinct region (a 'cliff'), which
    was measured as kurtosis 6.7-16.9 on the unstable axes (seed stability 0.36-
    0.53) vs 1.6-4.3 on every stable axis. Tissue-wide gradients sit below k0,
    so the penalty is inactive on them."""
    z = (d - d.mean(0, keepdim=True)) / (d.std(0, keepdim=True) + 1e-6)
    kurt = (z ** 4).mean(0)
    return torch.relu(kurt - k0).pow(2).mean()


def _loss_terms(model, xy, Y, lam_self, lam_cross, anchor_stage2, masks=None,
                lam_tail: float = 0.0):
    """One evaluation of the Saturn objective. Returns (loss, fit, anchor, share).

    masks[m] (bool, per spot) marks where modality m is OBSERVED; unobserved
    spots are left out of that modality's terms only.
    The isodepths are still defined -- and evaluated -- everywhere."""
    masks = masks or [None] * len(Y)
    d, full, own, cross = model(xy)
    fit = sum(F.mse_loss(_sel(f, mk), _sel(y, mk)) for f, y, mk in zip(full, Y, masks))
    if model.cross_on and anchor_stage2 == "residual":
        # own axis only has to explain what the cross pathways do NOT
        anc = sum(F.mse_loss(_sel(o, mk), _sel(y - c.detach(), mk))
                  for o, y, c, mk in zip(own, Y, cross, masks))
    elif model.cross_on and anchor_stage2 == "off":
        anc = fit * 0
    else:
        anc = sum(F.mse_loss(_sel(o, mk), _sel(y, mk)) for o, y, mk in zip(own, Y, masks))
    shr = (sum(_sel(c, mk).var(0).sum() / (_sel(f, mk).var(0).sum() + 1e-8)
               for c, f, mk in zip(cross, full, masks))
           if (model.M > 1 and model.cross_on) else fit * 0)
    tail = _tail_penalty(d) if lam_tail else fit * 0
    return fit + lam_self * anc + lam_cross * shr + lam_tail * tail, fit, anc, shr


@torch.no_grad()
def _fit_mse(model, xy, Y, masks=None) -> float:
    model.cross_on = True
    masks = masks or [None] * len(Y)
    _, full, _, _ = model(xy)
    return float(sum(F.mse_loss(_sel(f, mk), _sel(y, mk)) for f, y, mk in zip(full, Y, masks)))


def train_saturn(targets: Sequence[np.ndarray], coords: np.ndarray,
                 iso_hidden: Sequence[int] = (20, 20),
                 path_hidden: Sequence[int] = (20, 20),
                 lam_self: float = 1.0, lam_cross: float = 0.03,
                 epochs: int = 10000, warmup_frac: float = 0.3,
                 lr: float = 1e-3, seed: int = 0, standardize_d: bool = False,
                 anchor_stage2: str = "residual",
                 n_restarts: int = 4, restart_epochs: int = 4000,
                 restart_paths: bool = True, restart_tol: float = 0.005,
                 val_frac: float = 0.2, refit_all_epochs: int = 0,
                 masks: Optional[Sequence[np.ndarray]] = None, lam_tail: float = 0.0,
                 callback=None, device: str = "cuda", verbose: int = 0):
    """Fit Saturn. `targets` are globally scaled (N, K_m) arrays (saturn.preprocess);
    `coords` should be passed through saturn.preprocess.zscore_coords.

    Loss (per modality m, summed):
        MSE(full_m, Y_m)                       fit from all axes
      + lam_self  * anchor                     stage 1: MSE(own_m, Y_m)
                                               stage 2+: MSE(own_m, Y_m - cross_m)
      + lam_cross * var(cross_m)/var(full_m)   optional cost on cross pathways
    Full-batch Adam, as in GASTON. No explicit smoothness term.

    STAGE 1 (warmup_frac): cross pathways OFF -> M independent GASTON fits.
        Needed because additivity leaves a DISCRETE label swap; measured with a
        single stage: head R recovered the ATAC axis at 0.747, its own at 0.465.
    STAGE 2: cross pathways ON (zero-initialised). The anchor becomes RESIDUAL:
        the own axis only has to explain what the cross pathways do not. With the
        full anchor, d_m alone must reconstruct ALL of modality m -- the single-
        GASTON objective -- which REWARDS mixing the other axis into d_m and
        fights the cross pathway. Residual anchor + lam_cross=0 brought the
        estimated RNA<-ATAC share to 0.42-0.45 (true 0.41-0.44); the full anchor
        with lam_cross=0.1 gave 0.09.
    STAGE 3 (n_restarts rounds): for each head m, re-initialise its isodepth
        network (and, restart_paths, the pathways reading it), retrain
        restart_epochs with cross ON, and KEEP the restart only if the fit on
        HELD-OUT spots (val_frac, never trained on) drops by > restart_tol. Needed when a head locked onto ANOTHER modality's
        axis in stage 1: that duplicate leaves the modality's own structure
        unexplained, so the likelihood itself prefers the correct axis. Measured
        on a hard setting: stuck fit 0.144/0.167 -> restart 0.130/0.131 (oracle
        0.120/0.118); RNA axis 0.081 -> 0.792 and 0.065 -> 0.991.
    REFIT (refit_all_epochs > 0): after stage 3, train the chosen model on ALL
        spots, held-out ones included, so the reported fit uses every spot.
        This is NOT blind restart selection: unlike the earlier head-swap
        symmetry (equal likelihood), this failure has a genuine likelihood cost.
    hist["restarts"] records (round, head, fit_before, fit_after, accepted).
    """
    import copy
    set_all_seeds(seed, strict_gpu=(device == "cuda"))
    model = Saturn([y.shape[1] for y in targets], iso_hidden, path_hidden,
                   standardize_d=standardize_d).to(device)
    xy_all = torch.as_tensor(coords, dtype=torch.float32, device=device)
    Y_all = [torch.as_tensor(np.nan_to_num(np.asarray(y, dtype=np.float32)), device=device)
             for y in targets]
    Mk_all = ([None if mk is None else torch.as_tensor(np.asarray(mk, bool), device=device)
               for mk in masks] if masks is not None else [None] * len(targets))
    # Held-out spots decide restart acceptance (see STAGE 3). They are excluded
    # from every gradient step; isodepth is a function of coordinates, so the
    # final axes are still evaluated at every spot.
    N = xy_all.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).to(device)
    n_val = int(val_frac * N) if (val_frac > 0 and n_restarts > 0) else 0
    va, tr = perm[:n_val], perm[n_val:]
    xy, Y = xy_all[tr], [y[tr] for y in Y_all]
    Mk = [None if mk is None else mk[tr] for mk in Mk_all]
    xy_va, Y_va = (xy_all[va], [y[va] for y in Y_all]) if n_val else (xy, Y)
    Mk_va = [None if mk is None else mk[va] for mk in Mk_all] if n_val else Mk
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist: Dict[str, list] = {"loss": [], "fit": [], "own": [], "cross": [], "restarts": [],
                             "n_val": n_val}

    n_warm = int(warmup_frac * epochs)
    for ep in range(epochs):
        model.cross_on = ep >= n_warm
        opt.zero_grad()
        loss, fit, anc, shr = _loss_terms(model, xy, Y, lam_self, lam_cross, anchor_stage2, Mk, lam_tail)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            hist["loss"].append(loss.item()); hist["fit"].append(fit.item())
            hist["own"].append(anc.item()); hist["cross"].append(shr.item())
        if callback is not None:
            callback(ep, model, n_warm)
        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"  [{ep:5d}] loss={loss.item():.4f} fit={fit.item():.4f} "
                  f"own={anc.item():.4f} cross={shr.item():.4f}")

    # ---- stage 3: per-head restarts, kept only if the fit improves --------
    model.cross_on = True
    if model.M > 1:
        for r in range(n_restarts):
            for m in range(model.M):
                before = _fit_mse(model, xy_va, Y_va, Mk_va)
                cand = copy.deepcopy(model)
                torch.manual_seed(seed * 1000 + r * 10 + m + 1)
                for mod in cand.iso[m].modules():
                    if isinstance(mod, nn.Linear):
                        mod.reset_parameters()
                if restart_paths:
                    # also forget the curves learned ON the old axis: they would
                    # otherwise pull the fresh axis back into the old basin
                    for mm in range(cand.M):
                        net = cand.path[cand._key(mm, m)]
                        for mod in net.modules():
                            if isinstance(mod, nn.Linear):
                                mod.reset_parameters()
                        if mm != m:                       # cross paths restart at zero
                            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
                opt_c = torch.optim.Adam(cand.parameters(), lr=lr)
                cand.cross_on = True
                for _ in range(restart_epochs):
                    opt_c.zero_grad()
                    l_c, *_ = _loss_terms(cand, xy, Y, lam_self, lam_cross, anchor_stage2, Mk, lam_tail)
                    l_c.backward()
                    opt_c.step()
                after = _fit_mse(cand, xy_va, Y_va, Mk_va)
                # Accept on HELD-OUT spots. A size threshold on the training fit
                # cannot tell real from spurious: a measured genuine rescue
                # (RNA axis 0.808 -> 0.984) was a chain of <3% steps, while
                # spurious restarts that fit noise through the cross pathways
                # also gained 1-2% (and raised the other axis's leak). Only real
                # structure should improve fit at spots the model never saw.
                ok = after < before * (1 - restart_tol)
                hist["restarts"].append((r, m, before, after, ok))
                if verbose:
                    print(f"  restart r={r} head={m}: fit {before:.4f} -> {after:.4f} "
                          f"{'ACCEPT' if ok else 'reject'}")
                if ok:
                    model = cand
    model.cross_on = True
    # ---- final refit on ALL spots ----------------------------------------
    # The held-out spots only exist to make restart acceptance honest. Once the
    # basin is chosen, continue training on every spot so the reported model
    # (axes, pathways, coupling) is fitted to the whole section.
    if refit_all_epochs > 0 and n_val:
        opt_f = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(refit_all_epochs):
            opt_f.zero_grad()
            l_f, *_ = _loss_terms(model, xy_all, Y_all, lam_self, lam_cross, anchor_stage2, Mk_all, lam_tail)
            l_f.backward()
            opt_f.step()
    hist["refit_all_epochs"] = refit_all_epochs if n_val else 0
    hist["fit_all"] = _fit_mse(model, xy_all, Y_all, Mk_all)
    model.eval()
    return model, hist
